import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts/ops/glm_launcher_state.py"
SPEC = importlib.util.spec_from_file_location("glm_launcher_state_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repair_rule_normalizes_to_single_registered_prefix():
    project = "/data/projects/vllm"
    command = f"python3 -B {project}/scripts/ops/recover_framework.py"
    assert MODULE.canonical_repair_bash_rule(project, command) == command + ":*"
    assert MODULE.canonical_repair_bash_rule(project, command + ":*") == command + ":*"


@pytest.mark.parametrize(
    "command",
    [
        "docker exec w00939120-vllm python3 -I -c 'print(1)'",
        "cd /data/projects/vllm/runs/task/repair && python3 -I script.py",
        "python3 -m py_compile /data/projects/vllm/src/fullnet/measurements.py",
    ],
)
def test_repair_rule_rejects_shell_equivalents_and_raw_commands(command):
    with pytest.raises(ValueError, match="recover_framework.py"):
        MODULE.canonical_repair_bash_rule("/data/projects/vllm", command)


def test_probe_argv_accepts_equivalent_quoting():
    expected = ["python3", "-B", "/data/projects/vllm/scripts/ops/recover_framework.py",
                "--project", "/data/projects/vllm", "--task", "resume-task", "--action", "inspect"]
    assert MODULE.bash_matches_argv(
        "python3 -B /data/projects/vllm/scripts/ops/recover_framework.py --project '/data/projects/vllm' --task \"resume-task\" --action inspect",
        expected,
    )


@pytest.mark.parametrize(
    "command",
    [
        "cd /data/projects/vllm && python3 -B /data/projects/vllm/scripts/ops/recover_framework.py --project /data/projects/vllm --task resume-task --action inspect",
        "python3 -B /data/projects/vllm/scripts/ops/recover_framework.py --project /data/projects/vllm --task resume-task --action inspect; ps",
    ],
)
def test_probe_argv_rejects_chained_commands(command):
    expected = ["python3", "-B", "/data/projects/vllm/scripts/ops/recover_framework.py",
                "--project", "/data/projects/vllm", "--task", "resume-task", "--action", "inspect"]
    assert not MODULE.bash_matches_argv(command, expected)
