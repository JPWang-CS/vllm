"""Stop one identified project download before a supervised, resumable restart."""
import argparse
import json
import os
import signal
import time
from pathlib import Path


def command(pid, proc_root=Path('/proc')):
    try:
        return (proc_root / str(pid) / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return []


def parent(pid):
    try:
        for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
            if line.startswith('PPid:'):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass


def download_supervisors(root, task, proc_root=Path('/proc')):
    expected = [os.fsencode(root/'scripts/ops/prepare_pangu_assets.py'), b'--project', os.fsencode(root)]
    candidates = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = command(int(proc.name), proc_root)
            if args and Path(os.fsdecode(args[0])).name.startswith('python') and any(
                    args[i:i+len(expected)] == expected for i in (1, 2)):
                if b'--task' in args and args[args.index(b'--task')+1] == os.fsencode(task):
                    candidates.append(int(proc.name))
        except (PermissionError, IndexError):
            continue
    return candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--previous-task', required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    previous = (root / 'runs' / a.previous_task).resolve(strict=True)
    task = (root / 'runs' / a.task).resolve(strict=True)
    if not previous.is_relative_to(root) or not task.is_relative_to(root) or task == previous:
        raise ValueError('invalid task boundaries')
    candidates = download_supervisors(root, a.previous_task)
    if len(candidates) != 1:
        raise RuntimeError('expected exactly one live download supervisor; inspect state')
    supervisor = candidates[0]
    children = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if parent(pid) != supervisor:
            continue
        args = command(pid)
        if b'fullnet' in args and b'assets' in args and b'fetch' in args and b'pangu-7b' in args:
            children.append((pid, args))
    if len(children) != 1:
        raise RuntimeError('expected one fetch child; no signal sent')
    pid, args = children[0]
    receipt = {'previous_task': a.previous_task, 'supervisor_pid': supervisor,
               'fetch_pid': pid, 'reason': 'replace short download timeout with download-and-model pipeline'}
    (task/'superseded-download.json').write_text(json.dumps(receipt, indent=2))
    if command(pid) != args or parent(pid) != supervisor:
        raise RuntimeError('download process identity changed; no signal sent')
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+25
    while time.monotonic() < deadline:
        state = json.loads((previous/'job-state.json').read_text())
        if not command(pid) and not command(supervisor) and state.get('status') != 'running':
            receipt.update(fetch_exited=True, supervisor_exited=True, previous_exit=state.get('exit'))
            (task/'superseded-download.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt))
            return
        time.sleep(.5)
    raise RuntimeError('download shutdown unconfirmed; do not start another writer')


if __name__ == '__main__':
    main()
