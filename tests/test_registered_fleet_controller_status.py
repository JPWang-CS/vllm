import os
from pathlib import Path

from scripts.ops.registered_fleet_bridge import controller_process_alive


def put_process(proc, pid, *args):
    folder = proc / str(pid)
    folder.mkdir(parents=True)
    (folder / 'cmdline').write_bytes(b'\0'.join(os.fsencode(arg) for arg in args) + b'\0')


def test_shared_controller_without_active_pid_is_found(tmp_path):
    task = '/project/runs/current'
    put_process(tmp_path, 12, 'python3', '-B', '/management/glm_task.py', task, '/project')
    assert controller_process_alive(task, proc_root=tmp_path)
    assert controller_process_alive(task, pid=999, proc_root=tmp_path)


def test_other_tasks_and_command_mentions_do_not_count(tmp_path):
    task = '/project/runs/current'
    put_process(tmp_path, 12, 'python3', '/management/glm_task.py', task + '-other')
    put_process(tmp_path, 13, 'python3', '/project/worker.py', task)
    put_process(tmp_path, 14, 'sh', '-c', 'python3 /management/glm_task.py ' + task)
    assert not controller_process_alive(task, proc_root=tmp_path)


def test_controller_from_another_uid_does_not_count(tmp_path):
    put_process(tmp_path, 12, 'python3', '/management/glm_task.py', '/project/runs/current')
    assert not controller_process_alive('/project/runs/current', proc_root=tmp_path,
                                        uid=(tmp_path / '12').stat().st_uid + 1)
