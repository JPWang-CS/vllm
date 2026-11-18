import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/ops/glm_launcher_state.py"
spec = importlib.util.spec_from_file_location("glm_launcher_state", MODULE_PATH)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def terminal(**changes):
    value = {
        "exit": 0,
        "model": "glm-5.3-flash",
        "runtime": "Claude Code 2.1.314",
        "fifo_removed": True,
        "child_alive": False,
        "credentials_persisted": False,
    }
    value.update(changes)
    return value


def test_restart_note_in_state_file_is_not_terminal():
    payload = json.dumps({"reason": "broker restart requested", "at": 123.0})
    assert state.decode_terminal_state(payload) is None
    assert state.should_stop_proxy(terminal_state=None, controller_submitted=True) is False


def test_partial_atomic_write_is_not_terminal():
    assert state.decode_terminal_state(b'{"exit": 0, "model":') is None


def test_only_complete_controller_schema_is_terminal():
    value = terminal()
    assert state.parse_terminal_state(value) == value
    assert state.decode_terminal_state(json.dumps(value).encode()) == value


def test_rejects_exit_bool_or_missing_controller_fields():
    assert state.parse_terminal_state(terminal(exit=True)) is None
    assert state.parse_terminal_state({"exit": 0}) is None
    assert state.parse_terminal_state(terminal(child_alive=True)) is None
    assert state.parse_terminal_state(terminal(fifo_removed=False)) is None


def test_proxy_lifetime_tracks_controller_terminal_state():
    assert state.should_stop_proxy(terminal_state=None, controller_submitted=False) is True
    assert state.should_stop_proxy(terminal_state=terminal(exit=7), controller_submitted=True) is True
    assert state.should_stop_proxy(terminal_state=None, controller_submitted=True) is False
