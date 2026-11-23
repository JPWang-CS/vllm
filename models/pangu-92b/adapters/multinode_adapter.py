"""Multi-host (cross-host rank) distributed adapter for the 92B matrix cases.

Covers the launch_contract mode "distributed_adapter_required" scenarios
(multi_host_tp: TP4 across 2 hosts x 2 NPUs; multi_host_dp: DP4 x TP1
across the same split) on the fork's RayDistributedExecutor surface
(vllm/v1/executor/ray_executor.py: VLLM_RAY_BUNDLE_INDICES /
VLLM_RAY_PER_WORKER_GPUS, placement groups formed PACK in ray_utils).

Contract implemented here:
  plan()             host-major rank/bundle map + cluster formation artifacts
  preflight()        TCP reachability gate; ANY unreachable -> blocked_external
                     (resource/infra verdict, never a runtime failure)
  cluster_scripts()  per-host ray start commands to run as fleet repair/*.sh jobs
  head_env()         env for the head-host serve process
  fixture_rendezvous()  CPU-only gloo proof of the rank map (fixture; per the
                     acceptance contract it does NOT count as device acceptance)
  rank_evidence()    parse service log + placement table for real cross-host
                     rank proof (>= 2 distinct hosts)
"""
import json
import os
import pathlib
import re
import socket
import sys
import time


def plan(hosts, devices_per_host, parallel, ray_port=29600):
    """Host-major rank map for the 2-host 2-NPU split.

    multi_host_tp (tp=4, dp=1): global rank == TP rank, ranks 0-1 on host 0,
    ranks 2-3 on host 1. multi_host_dp (tp=1, dp=4): global rank == DP rank
    with the same host-major placement. Ray bundles are ordered host-major to
    match (bundle 2i on hosts[i]); rank_evidence() verifies the real placement
    table instead of trusting PACK ordering.
    """
    if len(hosts) != 2 or devices_per_host != [2, 2]:
        raise ValueError("adapter covers the frozen 2 hosts x 2 NPUs contract")
    world = sum(devices_per_host)
    tp, dp = parallel["tp"], parallel["dp"]
    if tp * dp != world:
        raise ValueError(f"tp*dp={tp * dp} != world {world}")
    ranks = []
    base = 0
    for host, count in zip(hosts, devices_per_host):
        ranks.append({"host": host, "ranks": list(range(base, base + count)),
                      "bundles": list(range(base, base + count))})
        base += count
    return {
        "head": hosts[0], "ray_port": ray_port, "world": world,
        "tp": tp, "dp": dp,
        "hosts": [{"host": h, "devices_per_host": c} for h, c in zip(hosts, devices_per_host)],
        "rank_map": ranks,
        "vllm_ray_bundle_indices": ",".join(str(b) for h in ranks for b in h["bundles"]),
        "vllm_ray_per_worker_gpus": 1.0,
    }


def preflight(targets, timeout=5.0):
    """TCP reachability from the submit host. Evidence, not assumption: a
    failing probe writes a dated verdict that maps to blocked_external."""
    results = {}
    for host, port in targets:
        entry = {"host": host, "port": port}
        try:
            with socket.create_connection((host, port), timeout=timeout):
                entry["reachable"] = True
        except OSError as exc:
            entry.update(reachable=False, error=f"{type(exc).__name__}: {exc}")
        results[host] = entry
    return {"verdict": "reachable" if all(v["reachable"] for v in results.values()) else "unreachable",
            "probed_at": time.time(), "targets": results}


def cluster_scripts(plan_value):
    """Per-host ray formation commands (run as fleet repair/*.sh jobs, per
    the broker rule that forbids arbitrary shell in fleet requests)."""
    head = plan_value["head"]
    port = plan_value["ray_port"]
    scripts = {}
    for entry in plan_value["hosts"]:
        host = entry["host"]
        if host == head:
            scripts[host] = f"ray start --head --port {port} --disable-usage-stats"
        else:
            scripts[host] = f"ray start --address='{head}:{port}' --disable-usage-stats"
    scripts["_verify"] = "python3 -c \"import ray; ray.init(address='auto'); print(len(ray.nodes()))\""
    return scripts


def pd_plan(hosts, devices_per_host, tp, kv_port_base=7100):
    """Host-split PD rank map for the disaggregated contract: producer
    (prefill) TP group on hosts[0], consumer (decode) TP group on hosts[1].

    Mirrors the same-host split topology of run_pangu92b.py (producer devices
    = devices[:tp], consumer devices = devices[tp:]) with each role pinned to
    its own host. The KV endpoint is the piece the same-host adapter never
    exercised: producer and consumer now sit in DIFFERENT containers, so the
    connector address must be reachable cross-host; that is gated by the
    device-free container-network probes before any device phase starts.
    """
    if len(hosts) != 2 or devices_per_host != [2, 2]:
        raise ValueError("adapter covers the frozen 2 hosts x 2 NPUs contract")
    if tp != sum(devices_per_host) // 2:
        raise ValueError(f"pd tp={tp} does not match per-host devices {devices_per_host}")
    producer, consumer = hosts
    return {
        "head": producer,
        "world": 2 * tp,
        "rank_map": [{"host": producer, "ranks": list(range(tp)), "role": "prefill"},
                     {"host": consumer, "ranks": list(range(tp, 2 * tp)), "role": "decode"}],
        "producer": {"host": producer, "role": "prefill", "tp": tp},
        "consumer": {"host": consumer, "role": "decode", "tp": tp},
        "kv_port_base": kv_port_base,
        "relay_ports": {"prefill": 18333, "decode": 18334, "proxy": 18335},
        "cross_host_kv_endpoint": f"{producer}:{kv_port_base}",
        "endpoint_verified": False,
    }


def head_env(plan_value):
    return {
        "VLLM_RAY_BUNDLE_INDICES": plan_value["vllm_ray_bundle_indices"],
        "VLLM_RAY_PER_WORKER_GPUS": str(plan_value["vllm_ray_per_worker_gpus"]),
        "RAY_USAGE_STATS_ENABLED": "0",
    }


def _rank_worker(rank, plan_value, port, out_dir):
    import datetime
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      WORLD_SIZE=str(plan_value["world"]), RANK=str(rank),
                      GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist
    dist.init_process_group("gloo")
    mine = next(h for h in plan_value["rank_map"] if rank in h["ranks"])
    tensor = torch.tensor([rank], dtype=torch.float)
    dist.all_reduce(tensor)
    ok = int(tensor.item()) == sum(range(plan_value["world"]))
    (out_dir / f"rank{rank}.json").write_text(json.dumps(
        {"rank": rank, "host_group": mine["host"], "all_reduce_ok": ok,
         "at": datetime.datetime.now().isoformat(timespec="seconds")}))
    dist.barrier()
    dist.destroy_process_group()


def fixture_rendezvous(plan_value, out_dir, port=29671):
    """CPU gloo fixture: the host-major map from plan() is exercised by 4 real
    processes rendezvousing over loopback; ranks learn their host group from
    the SAME map the real launch uses. FIXTURE ONLY - per acceptance contract
    (fixture_counts_as_device_acceptance: false) this proves adapter logic,
    never device acceptance."""
    import multiprocessing as mp
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_worker, args=(r, plan_value, port, out_dir))
             for r in range(plan_value["world"])]
    started = time.time()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    ranks = []
    for r in range(plan_value["world"]):
        path = out_dir / f"rank{r}.json"
        ranks.append(json.loads(path.read_text()) if path.is_file() else {"rank": r, "missing": True})
    ok = all(entry.get("all_reduce_ok") and entry.get("host_group") is not None for entry in ranks)
    verdict = {"fixture": "multinode_rendezvous", "passed": bool(ok),
               "ranks": ranks, "elapsed": round(time.time() - started, 1),
               "counts_as_device_acceptance": False}
    (out_dir / "fixture-verdict.json").write_text(json.dumps(verdict, ensure_ascii=False))
    return verdict


def rank_evidence(service_log_text, placement_table_text):
    """Real cross-host rank proof for acceptance: >= 2 distinct worker hosts in
    the service log AND a placement group whose bundles span 2 nodes."""
    hosts = sorted(set(_HOST_RE.findall(service_log_text)))
    nodes = sorted(set(_HOST_RE.findall(placement_table_text)))
    return {"worker_hosts": hosts, "distinct_hosts": len(hosts),
            "placement_nodes": nodes,
            "has_cross_host_ranks": len(hosts) >= 2 and len(nodes) >= 2}


_HOST_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|7\.\d{1,3}\.\d{1,3}\.\d{1,3}|host-[A-Za-z0-9_-]+)\b")


if __name__ == "__main__":
    demo = plan(["host-a", "host-b"], [2, 2], {"tp": 4, "dp": 1})
    print(json.dumps(demo, ensure_ascii=False))
    sys.exit(0)


