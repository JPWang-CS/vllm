"""Pangu 7B: fixed RULER preparation, framework launch, scoring and export."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from fullnet.analysis import report
from fullnet.execution import plan, run_experiment
from fullnet.export import export_run
from fullnet.store import experiment, read_json, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--model-python", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--platform", choices=("a3", "a5"), required=True)
    p.add_argument("--device-index", type=int, required=True)
    p.add_argument("--npu-id", type=int, required=True)
    p.add_argument("--task", required=True)
    p.add_argument('--dataset', type=Path, help='Reuse previously frozen model-specific samples')
    p.add_argument('--keep-ready', action='store_true')
    p.add_argument('--port', type=int, default=18321)
    p.add_argument("--prepare-only", action="store_true", help="Prepare fixed evaluation samples without device computation")
    p.add_argument("--entrypoint-only", action="store_true", help="Check the real model entrypoint without loading weights")
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    task = (root / "runs" / a.task).resolve(strict=True)
    if not task.is_relative_to(root):
        raise ValueError("task outside project")
    env = os.environ.copy()
    runtime = a.runtime_root.resolve(strict=True)
    sources = [runtime / "source/image/opt/vllm", runtime / "source/image/workspace/omniinfer/components/omni-npu/src",
               runtime / "source/image/workspace/omniinfer/components/omni-models", runtime / "source/image/workspace/omniinfer/components/omni-cache",
               runtime / "source/image/workspace/omniinfer/components/omni-eplb", runtime / "source/image/workspace/omniinfer"]
    evaluation_deps = root / "envs/evaluation-deps"
    paths = [root / "src", evaluation_deps, *[v for v in sources if v.is_dir()]]
    cann_python = Path(env["ASCEND_HOME_PATH"]) / "python/site-packages"
    paths.append(cann_python)
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    native=runtime/'envs/pangu7b-native'
    if native.is_dir():
        env['LD_LIBRARY_PATH']=str(native)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    for key, suffix in {"HOME":"home", "XDG_CACHE_HOME":"cache", "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp",
                        "PIP_CACHE_DIR":"cache/pip", "HF_HOME":"cache/hf", "HF_MODULES_CACHE":"cache/hf-modules",
                        "ASCEND_PROCESS_LOG_PATH":"logs/ascend", "ASCEND_WORK_PATH":"tmp/ascend"}.items():
        target = task / suffix
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env.update(PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0", NO_PROXY="localhost,127.0.0.1,::1",
               ASCEND_RT_VISIBLE_DEVICES=str(a.device_index), ASCEND_VISIBLE_DEVICES=str(a.device_index),
               VLLM_USE_V1="1", VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_PLUGINS="omni-npu,omni_custom_models,omni_npu_patches",
               GLOO_SOCKET_IFNAME="lo", HCCL_SOCKET_IFNAME="lo")
    if a.platform == "a5":
        env["ASCEND_COMPUTE_UNIT"] = "ascend950"
    if env.get("FULLNET_CA_BUNDLE"):
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT"):
            env[key] = env["FULLNET_CA_BUNDLE"]
    steps = []
    def call(label, argv, timeout=300):
        with (task / (label + ".log")).open("w") as output:
            r = subprocess.run(argv, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        steps.append({"stage":label, "argv":argv, "exit":r.returncode})
        (task / "validation-steps.json").write_text(json.dumps(steps,ensure_ascii=False,indent=2))
        print(json.dumps(steps[-1],ensure_ascii=False),flush=True)
        if r.returncode:
            raise RuntimeError(label + " failed")
    def guard(label):
        r = subprocess.run([str(root / "tools/npu-smi"), "info"], env=env, capture_output=True, text=True, timeout=30)
        (task / (label + ".txt")).write_text(r.stdout+r.stderr)
        if r.returncode or not re.search(r"\|\s*"+str(a.npu_id)+r"\s+\|?\s*Ascend\w*\s*\|\s*OK\b",r.stdout):
            raise RuntimeError("target device health unconfirmed")
        if "No running processes found in NPU " + str(a.npu_id) + " " not in r.stdout:
            raise RuntimeError("target device is occupied or process visibility unconfirmed")
    status={"model":"pangu-7b","platform":a.platform,"model_inference":"not_run","quality_on_real_model":"not_run"}
    run=None
    try:
        config = read_json(a.weights / "config.json")
        if config.get("architectures") != ["PanguEmbeddedForCausalLM"] or config.get("hidden_size") != 4096 or config.get("num_hidden_layers") != 34:
            raise ValueError("expected real Pangu Embedded 7B, hidden 4096, 34 layers")
        if not a.prepare_only and not a.entrypoint_only and not list(a.weights.glob("*.safetensors")):
            raise ValueError("real model weight shards missing")
        if not a.prepare_only:
            guard("device-pre")
        if a.entrypoint_only:
            call("entrypoint", [str(a.model_python), "-B", str(root/"models/pangu-7b/adapters/serve.py"),
                "--model", str(a.weights), "--help"], timeout=180)
            guard("device-post-entrypoint")
            status.update(entrypoint="passed", model_inference="not_run")
            return
        ruler = root / "cache/RULER"
        call("ruler-source",[sys.executable,"-B",str(root/"scripts/evaluation/get_ruler.py"),"--output",str(ruler)])
        check = task / "check-evaluation-deps.py"
        check.write_text('import importlib.util,json\nprint(json.dumps([n for n in ["nltk","wonderwords","tenacity"] if importlib.util.find_spec(n) is None]))\n')
        r = subprocess.run([str(a.model_python),"-B",str(check)],env=env,capture_output=True,text=True,check=True)
        missing=json.loads(r.stdout.strip())
        if missing:
            versions={"nltk":"nltk==3.9.1","wonderwords":"wonderwords==2.2.0","tenacity":"tenacity==9.1.2"}
            call("evaluation-deps",[sys.executable,"-B","-m","pip","install","--target",str(evaluation_deps),"--no-deps",*[versions[n] for n in missing]])
        data=task/"data"
        if a.dataset:
            dataset=a.dataset.resolve(strict=True)
            if not dataset.is_relative_to(root):raise ValueError('dataset outside project')
        else:
            call("ruler-data",[str(a.model_python),"-B",str(root/"scripts/evaluation/prepare_ruler_subset.py"),
              "--ruler-root",str(ruler),"--model",str(a.weights),"--output",str(data),"--lengths","4096","8192","16384",
              "--samples","8","--seed","42","--model-template-token","64","--chat-template-kwargs",'{}',
              "--user-suffix","\n/no_think"],timeout=900)
            dataset=data/"all.jsonl"
            dataset.write_text("".join(v.read_text() for v in sorted(data.glob("*-*.jsonl"))))
        if len(rows(dataset)) != 48:
            raise ValueError("fixed RULER subset must contain 48 samples")
        if a.prepare_only:
            status.update(evaluation_data="prepared", samples=48, dataset=str(dataset))
            return
        model_env={k:v for k,v in env.items() if k in ("PYTHONPATH","LD_LIBRARY_PATH","VLLM_USE_V1","VLLM_WORKER_MULTIPROC_METHOD","VLLM_PLUGINS","ASCEND_COMPUTE_UNIT","GLOO_SOCKET_IFNAME","HCCL_SOCKET_IFNAME")}
        model_env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        c={"schema_version":1,"model":"pangu-7b","platform":a.platform,"devices":[a.device_index],"changes":[],"resolved_changes":[],
           "identity":{"architecture":"PanguEmbeddedForCausalLM","weights":str(a.weights),"dtype":"bfloat16","quantization":"none",
                       "npu_id":a.npu_id,"runtime_root":str(runtime),"async_scheduling":False,"prefix_caching":False,"api_server_count":1,
                       "thinking_control":{"mode":"fast","user_suffix":"\n/no_think"}},
           "env":model_env,"service":{"argv":[str(a.model_python),"-B",str(root/"models/pangu-7b/adapters/serve.py"),
                       "--model",str(a.weights),"--served-model-name","pangu","--trust-remote-code","--host","127.0.0.1","--port",str(a.port),
                       "--dtype","bfloat16","--max-model-len","16384","--tensor-parallel-size","1","--max-num-batched-tokens","512",
                       "--max-num-seqs","4","--gpu-memory-utilization","0.7","--enforce-eager","--seed","0"],
                       "base_url":f"http://127.0.0.1:{a.port}/v1","served_model":"pangu","ready_timeout":1800,"request_timeout":240,"shutdown_timeout":30},
           "evaluation":{"dataset":str(dataset),"scope":"Pangu 7B / RULER niah_single_1 + vt; 48 fixed samples", "metric":"ruler_official",
                       "official_metrics":str(ruler/"scripts/eval/synthetic/constants.py"),"context_limit":16384,"seed":42,"temperature":0,"top_p":1,
                       "chat_template_kwargs":{},"add_special_tokens":False,"minimum_score":None}}
        guard("device-before-launch")
        os.environ.update(env)
        run=plan(root,c,a.task)
        def completed(path,config,result):
            status.update(model_inference='requests_completed',quality_on_real_model=result['quality_status'],
                          score=result['score'],protocol_pass=result['protocol_pass'],retention_requested=a.keep_ready)
            report(path)
            export_run(root,path,task/'export')
            (task/'validation.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        result=run_experiment(root,"pangu-7b",a.task,after_evaluation=completed,keep_ready=a.keep_ready)
        report(run)
        status.update(model_inference="requests_completed",quality_on_real_model=result["quality_status"],score=result["score"],protocol_pass=result["protocol_pass"])
        if not result["protocol_pass"] or result["quality_status"] == "failed":
            raise RuntimeError("model evaluation failed; retain raw responses and result details")
    except Exception as exc:
        status.update(error=type(exc).__name__,detail=str(exc))
        raise
    finally:
        if run and run.exists():
            # Keep the original failure even when a report/export step also fails.
            for label, action in (("report", lambda: report(run)),
                                  ("export", lambda: export_run(root,run,task/"export") if not (task/'export').exists() else None)):
                try:
                    action()
                except Exception as exc:
                    status.setdefault("artifact_errors", {})[label] = str(exc)
            service_log=run/"logs/service.log"
            if service_log.exists():(task/"service.log").write_text(service_log.read_text(errors="replace"))
            try:
                device=subprocess.run([str(root/"tools/npu-smi"),"info"],env=env,capture_output=True,text=True,timeout=30)
                (task/"device-post.txt").write_text(device.stdout+device.stderr)
                cleanup=read_json(run/"state.json").get("phases",{}).get("cleanup",{})
                status["cleanup"]={"service_exit":cleanup.get("service_exit"),
                    "target_device_empty":device.returncode==0 and "No running processes found in NPU "+str(a.npu_id)+" " in device.stdout}
            except Exception as exc:
                status["cleanup"]={"error":str(exc),"target_device_empty":None}
        status["numerical_parity"]="not_run"
        (task/"validation.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
