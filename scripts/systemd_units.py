import json
import os
import stat
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "deploy" / "systemd"
NAMES = ("sack-knowledge-refresh", "sack-knowledge-backup")
TIMERS = tuple(name + ".timer" for name in NAMES)
UNITS = tuple(name + ".service" for name in NAMES) + TIMERS
MARKER = b"# sack-knowledge systemd installer sha256="


def destination():
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config / "systemd" / "user"


def render(template):
    directory = str(ROOT).replace("%", "%%")
    if "\n" in directory or "\r" in directory or directory.endswith("\\") or directory.rstrip() != directory:
        raise SystemExit("The repository path is not supported by the systemd templates.")
    executable = json.dumps(directory.replace("$", "$$") + "/.venv/bin/python", ensure_ascii=False)
    return template.replace("@REPOSITORY@", directory).replace("@PYTHON@", executable)


def unit_contents(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            return None
        return file.read()


def marked(body):
    return MARKER + sha256(body).hexdigest().encode("ascii") + b"\n" + body


def owned(content):
    if content is None:
        return False
    _, _, body = content.partition(b"\n")
    return content == marked(body)


def unit_stamp(path):
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino, status.st_mode, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def write_unit(path, body, expected):
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(marked(body))
            file.close()
            if unit_stamp(path) != expected:
                raise SystemExit("Unit changed during installation: " + path.name)
            if expected is None:
                os.link(temporary, path)
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def systemctl(*args):
    result = subprocess.run(["systemctl", "--user", *args], capture_output=True, check=False)
    if result.returncode:
        raise SystemExit("systemctl --user " + " ".join(args) + " failed.")
