"""Fetch the official files needed for the existing two-task RULER protocol."""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


FILES = ("LICENSE", "scripts/synthetic.yaml", "scripts/data/tokenizer.py", "scripts/data/manifest_utils.py",
         "scripts/data/synthetic/constants.py", "scripts/data/synthetic/niah.py",
         "scripts/data/synthetic/variable_tracking.py", "scripts/eval/synthetic/constants.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="main")
    p.add_argument("--output", type=Path, default=Path("cache/RULER"))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = a.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("source must be saved inside this project")
    receipt = output / "source.json"
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data["revision"] != a.revision:
            raise ValueError("use a separate source directory for a different revision")
    urls = []
    for filename in FILES:
        target = output / filename
        url = "https://raw.githubusercontent.com/NVIDIA/RULER/" + urllib.parse.quote(a.revision, safe="") + "/" + filename
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            with target.open("xb") as stream:
                stream.write(content)
        urls.append(url)
    receipt.write_text(json.dumps({"repository": "NVIDIA/RULER", "revision": a.revision, "files": urls}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
