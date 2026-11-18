"""Export a run from a root container to its host-owned project directory."""
import os
from pathlib import Path

from .store import inside


def export_run(root, run, destination):
    root, run = Path(root).resolve(), inside(root, run)
    destination = inside(root, destination)
    if destination == root or destination.is_relative_to(run):
        raise ValueError("export to a separate task directory")
    parent = destination.parent.resolve(strict=True)
    ownership = parent.stat()
    destination.mkdir(mode=0o700)
    names = ("config.json", "dataset.jsonl", "changes.json", "state.json", "process.json", "results.json",
             "report.md", "final-report.md", "commands.jsonl", "raw/predictions.jsonl", "evaluation-progress.json", "smoke.json",
             "performance/summary.json", "performance/profile-artifacts.json", "validation.json")
    for name in names:
        source = inside(run, run / name)
        if source.is_file():
            target = inside(destination, destination / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(source.read_bytes())
            os.chmod(target, 0o600)
    # This changes only the new export, never source files, environments or other runs.
    if os.name == "posix" and os.geteuid() == 0:
        for path in sorted(destination.rglob("*"), reverse=True):
            os.chown(path, ownership.st_uid, ownership.st_gid, follow_symlinks=False)
        os.chown(destination, ownership.st_uid, ownership.st_gid)
    return {"export": str(destination), "source": str(run), "files": [v for v in names if (destination / v).is_file()]}
