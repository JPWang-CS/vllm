"""Install the framework in a private venv, including Python installations without pip."""
import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="envs/fullnet")
    p.add_argument("--system-site-packages", action="store_true")
    p.add_argument("--wheelhouse", type=Path)
    p.add_argument("--virtualenv-archive", type=Path)
    p.add_argument("--extras", default="")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = (root / a.env).resolve()
    if not target.is_relative_to(root / "envs"):
        raise ValueError("environment must be under this project's envs directory")
    if a.extras and any(x not in ("test", "download", "evaluation", "metrics") for x in a.extras.split(",")):
        raise ValueError("unknown extras")
    env = os.environ.copy()
    for key, suffix in {"PIP_CACHE_DIR":"cache/pip", "TMPDIR":"tmp", "TEMP":"tmp", "TMP":"tmp",
                        "VIRTUALENV_OVERRIDE_APP_DATA":"cache/virtualenv"}.items():
        path = root / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        args = [sys.executable, "-B", "-m", "venv", str(target)]
        if a.system_site_packages:
            args.append("--system-site-packages")
        result = subprocess.run(args, cwd=root, env=env)
    else:
        result = subprocess.run([str(python), "-B", "-m", "pip", "--version"], cwd=root, env=env)
    if result.returncode:
        archive = a.virtualenv_archive or root / "cache/bootstrap/virtualenv.pyz"
        if not archive.is_file():
            archive.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen("https://bootstrap.pypa.io/virtualenv.pyz", timeout=60) as response:
                archive.write_bytes(response.read())
        args = [sys.executable, "-B", str(archive), str(target), "--no-periodic-update", "--no-download",
                "--app-data", str(root / "cache/virtualenv")]
        if a.system_site_packages:
            args.append("--system-site-packages")
        subprocess.run(args, cwd=root, env=env, check=True)
    # Check the interpreter belongs to this env before issuing any package installation.
    subprocess.run([str(python), "-B", "-c", "import pathlib,sys; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve()", str(target)], check=True, env=env)
    requirement = "." + ("[" + a.extras + "]" if a.extras else "")
    args = [str(python), "-B", "-m", "pip", "install", "-e", requirement]
    if a.wheelhouse:
        args += ["--no-index", "--find-links", str(a.wheelhouse.resolve())]
    subprocess.run(args, cwd=root, env=env, check=True)
    print("Framework interpreter:", python)


if __name__ == "__main__":
    main()
