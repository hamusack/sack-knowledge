import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "deploy" / "systemd"
NAMES = ("sack-knowledge-refresh", "sack-knowledge-backup")
TIMERS = tuple(name + ".timer" for name in NAMES)
UNITS = tuple(name + ".service" for name in NAMES) + TIMERS


def destination():
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config / "systemd" / "user"


def render(template):
    directory = str(ROOT).replace("%", "%%")
    if "\n" in directory or "\r" in directory or directory.endswith("\\") or directory.rstrip() != directory:
        raise SystemExit("The repository path is not supported by the systemd templates.")
    executable = json.dumps(directory.replace("$", "$$") + "/.venv/bin/python", ensure_ascii=False)
    return template.replace("@REPOSITORY@", directory).replace("@PYTHON@", executable)


def systemctl(*args):
    result = subprocess.run(["systemctl", "--user", *args], capture_output=True, check=False)
    if result.returncode:
        raise SystemExit("systemctl --user " + " ".join(args) + " failed.")
