"""Export earlier fixture evidence and confirm its HTTP process has exited."""
import argparse
import json
import socket
from pathlib import Path

from fullnet.export import export_run
from fullnet.store import experiment, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--previous", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="workflow-fixture")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    task = (root / "runs" / a.task).resolve(strict=True)
    if not task.is_relative_to(root):
        raise ValueError("task escapes project")
    source = experiment(root, a.model, a.previous)
    receipt = export_run(root, source, task / "export")
    state = read_json(source / "state.json")
    from urllib.parse import urlparse
    url = urlparse(read_json(source / "config.json")["service"]["base_url"])
    with socket.socket() as probe:
        probe.settimeout(2)
        port_closed = probe.connect_ex((url.hostname, url.port)) != 0
    outcome = {"export": receipt, "fixture_only": True, "service_port_closed": port_closed,
               "service_exit": state["phases"]["cleanup"]["service_exit"],
               "model_inference": "not_run", "quality_on_real_model": "not_run"}
    target = task / "finish.json"
    target.write_text(json.dumps(outcome, ensure_ascii=False, indent=2))
    print(json.dumps(outcome, ensure_ascii=False))
    if not port_closed or outcome["service_exit"] is None:
        raise RuntimeError("fixture cleanup not confirmed")


if __name__ == "__main__":
    main()
