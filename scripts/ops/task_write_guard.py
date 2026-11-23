"""Claude write hook: allow only the reviewed project paths and task edits."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
scopes = [Path(value).resolve(strict=False) for value in sys.argv[2:]]
protected = {
    root / 'scripts/ops/submit_registered_glm.py',
    root / 'scripts/ops/glm_launcher_state.py',
    root / 'scripts/ops/task_write_guard.py',
}
allowed = False
try:
    payload = json.load(sys.stdin)
    supplied = Path(payload.get('tool_input', {}).get('file_path', ''))
    if supplied.is_absolute():
        target = supplied.resolve(strict=False)
        allowed = target.is_relative_to(root) and target not in protected and any(
            target == scope or (scope.is_dir() and target.is_relative_to(scope)) for scope in scopes)
except (ValueError, OSError):
    pass
print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
    'permissionDecision': 'allow' if allowed else 'deny',
    'permissionDecisionReason': 'Reviewed project orchestration or task candidate only; original kernels and shared environments are read-only'}}))
