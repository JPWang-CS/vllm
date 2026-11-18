"""Stage a read-only host diagnostic ELF in the project's private tools directory."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    project = a.project.resolve(strict=True)
    source = a.source.resolve(strict=True)
    if project.name != "vllm" or source.name != "npu-smi":
        raise ValueError("unexpected project or tool")
    with source.open("rb") as data:
        if data.read(4) != b"\x7fELF":
            raise ValueError("source is not the host npu-smi ELF")
    directory = (project / "tools").resolve()
    if not directory.is_relative_to(project):
        raise ValueError("tool destination escapes project")
    directory.mkdir(exist_ok=True)
    target = directory / "npu-smi"
    if target.exists():
        raise ValueError("private tool exists; inspect before replacing")
    with source.open("rb") as reader, target.open("xb") as output:
        shutil.copyfileobj(reader, output)
    target.chmod(0o700)
    print(json.dumps({"source": str(source), "target": str(target), "bytes": target.stat().st_size,
                      "original_modified": False, "device_operation": "not_started"}))


if __name__ == "__main__":
    main()
