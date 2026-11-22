import json
import os
import re
import secrets
import sqlite3
import stat
import subprocess
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

from .errors import LocalCommandError
from .operations import operation_lock


RESTIC = "/usr/bin/restic"
TAG = "sack-knowledge"
PENDING_TAG = "sack-knowledge-pending"
STAGE = ".backup-database.sqlite3"
STAGE_PREFIX = ".backup-stage-"
WORK = ".backup-work"
STATE = "backup-state.json"
PASSWORD_NOTICE = "Store a separate copy of the backup password file outside the backup repository."
DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def backup_host():
    identity = Path("/etc/machine-id").read_bytes().strip()
    if not re.fullmatch(rb"[a-f0-9]{32}", identity):
        raise LocalCommandError("A machine ID is required to scope the backup source.")
    return TAG + "-" + sha256(b"sack-knowledge-backup:" + identity).hexdigest()[:24]


def mount_identity(fd):
    with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as info:
        mount_id = next(line.split()[1] for line in info if line.startswith("mnt_id:"))
    entry = os.fstat(fd)
    return mount_id, entry.st_dev, entry.st_ino


@dataclass
class PinnedRepository:
    mount: Path
    name: str
    mount_fd: int
    repo_fd: int

    @property
    def path(self):
        return Path(f"/proc/self/fd/{self.repo_fd}")

    def check(self):
        try:
            if not self.mount.is_mount():
                raise OSError
            current = os.open(self.mount, DIRECTORY)
            try:
                if mount_identity(current) != mount_identity(self.mount_fd):
                    raise OSError
                repo = os.open(self.name, DIRECTORY, dir_fd=current)
                try:
                    if mount_identity(repo) != mount_identity(self.repo_fd):
                        raise OSError
                finally:
                    os.close(repo)
            finally:
                os.close(current)
        except OSError:
            raise LocalCommandError("The backup destination mount changed.") from None


@contextmanager
def mounted_repository(options, *, initialize=False):
    mount = Path(options["mount_point"]).resolve()
    repository = Path(options["repository"]).absolute()
    if repository.parent.resolve() != mount or repository.name in ("", ".", ".."):
        raise LocalCommandError("The backup repository must be a directory directly under the configured mount.")
    if not mount.is_mount():
        raise LocalCommandError("The backup destination is not mounted.")
    mount_fd = os.open(mount, DIRECTORY)
    try:
        if not mount.is_mount() or not os.path.samestat(os.fstat(mount_fd), mount.stat()):
            raise LocalCommandError("The backup destination mount changed.")
        if initialize:
            try:
                os.mkdir(repository.name, mode=0o700, dir_fd=mount_fd)
            except FileExistsError:
                pass
        try:
            repo_fd = os.open(repository.name, DIRECTORY, dir_fd=mount_fd)
        except FileNotFoundError:
            raise LocalCommandError("Initialize the backup repository first.") from None
        try:
            # Restic runs from the pinned directory, including if the mount is detached.
            pinned = PinnedRepository(mount, repository.name, mount_fd, repo_fd)
            pinned.check()
            yield pinned
        finally:
            os.close(repo_fd)
    finally:
        os.close(mount_fd)


def check_password(path, *, create=False):
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="ascii") as file:
                file.write(secrets.token_urlsafe(48) + "\n")
                file.flush()
                os.fsync(file.fileno())
    if not path.exists():
        raise LocalCommandError("The backup password file is missing.")
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
        raise LocalCommandError("The backup password file must be regular and have mode 0600.")


class Restic:
    def __init__(self, repository, password, root):
        self.repository = repository
        self.password = password
        self.root = root
        self.host = backup_host()

    def run(self, command, *args):
        env = {key: value for key, value in os.environ.items() if not key.startswith("RESTIC_")}
        self.repository.check()
        try:
            result = subprocess.run(
                [RESTIC, "--repo", ".", "--password-file", str(self.password), "--no-cache", "--json",
                 command, *args], cwd=self.repository.path, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, check=False,
            )
        finally:
            self.repository.check()
        if result.returncode:
            raise LocalCommandError(f"Restic {command} failed (exit {result.returncode}).")
        return result.stdout

    def scope(self, tag):
        return ("--tag", tag, "--host", self.host, "--path", str(self.root))

    def owns(self, snapshot, tag):
        return (snapshot.get("hostname") == self.host and snapshot.get("paths") == [str(self.root)]
                and tag in (snapshot.get("tags") or []))

    def snapshots(self, tag=TAG):
        rows = json.loads(self.run("snapshots", *self.scope(tag))) or []
        return [row for row in rows if self.owns(row, tag)]

    def retain(self):
        groups = json.loads(self.run(
            "forget", "--dry-run", *self.scope(TAG), "--group-by", "host,paths,tags",
            "--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "12"
        )) or []
        expired = [snapshot["id"] for group in groups for snapshot in (group.get("remove") or [])
                   if self.owns(snapshot, TAG)]
        expired.extend(snapshot["id"] for snapshot in self.snapshots(PENDING_TAG))
        if expired:
            self.run("forget", *sorted(set(expired)))
        self.run("prune")


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)


def stage_database(database, stage):
    with closing(readonly(database)) as source, closing(sqlite3.connect(stage)) as target:
        source.execute("BEGIN")
        source.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()
        source.backup(target, pages=256)
        # Keep the staged copy self-contained even if the live database uses WAL.
        target.execute("PRAGMA journal_mode=DELETE")


def literal_pattern(path):
    return re.sub(r"([\\*?\[\]])", r"\\\1", str(path))


def excluded_paths(config, root, database, stage=None):
    return {database, *(database.with_name(database.name + suffix)
                         for suffix in ("-wal", "-shm", "-journal")),
            root / WORK, root / STAGE, root / ".backup.lock", root / ".refresh.lock",
            Path(config.get("embedding", {}).get("cache", "local/models")).resolve(),
            *(path for path in root.glob(f"{STAGE_PREFIX}*") if path != stage)}


def exclusions(config, root, database, stage):
    return [argument for path in sorted(excluded_paths(config, root, database, stage))
            for argument in ("--exclude", literal_pattern(path))] + [
        "--exclude", ".venv",
    ]


def require_backed_up_files(config, config_path, root, database, password):
    excluded = excluded_paths(config, root, database)
    required = {"--config": config_path, "backup password file": password}

    def collect(table, prefix=""):
        for key, value in table.items():
            name = prefix + key
            if isinstance(value, dict):
                collect(value, name + ".")
            elif key.endswith(("password_file", "secret_file", "token_file", "credentials_file",
                               "private_key_file", "api_key_file")):
                required[name] = value

    collect(config)
    for setting, value in required.items():
        path = Path(value).absolute() if isinstance(value, (str, Path)) and value else None
        if (path is None or not path.is_relative_to(root) or path.resolve() != path or not path.is_file()
                or ".venv" in path.relative_to(root).parts or any(path.is_relative_to(item) for item in excluded)):
            raise LocalCommandError(f"{setting} must point to a regular file inside the backup set.")


def validate_database_path(database, root, repository, password):
    outputs = [root / name for name in (WORK, STATE, STAGE, ".backup.lock", ".refresh.lock")]
    outputs += [*root.glob(f"{STAGE_PREFIX}*"), password, Path(repository)]
    paths = (database, *(database.with_name(database.name + suffix) for suffix in ("-wal", "-shm", "-journal")))
    for path in paths:
        if ((path.is_relative_to(root) and any(part.startswith(STAGE_PREFIX)
                                              for part in path.relative_to(root).parts))
                or any(path.is_relative_to(output.resolve()) or output.resolve().is_relative_to(path)
                       or (path.exists() and output.exists() and path.samefile(output)) for output in outputs)):
            raise LocalCommandError("The live database overlaps a backup output path.")


def directory_bytes(path):
    return sum((Path(folder) / name).stat().st_size for folder, _, names in os.walk(path)
               for name in names)


def make_snapshot(config, root, database, restic):
    with TemporaryDirectory(prefix=STAGE_PREFIX, dir=root) as directory:
        stage = Path(directory)
        pending = stage / STAGE
        pending.touch(mode=0o600, exist_ok=False)
        stage_database(database, pending)
        output = restic.run("backup", "--host", restic.host, "--tag", PENDING_TAG,
                            *exclusions(config, root, database, stage), str(root))
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("message_type") == "summary" and event.get("snapshot_id"):
            original = event["snapshot_id"]
            for tag_line in restic.run("tag", "--set", TAG, original).splitlines():
                # Restic can print an informational line before its JSON tag events.
                if not tag_line.lstrip().startswith("{"):
                    continue
                tagged = json.loads(tag_line)
                if tagged.get("old_snapshot_id", "").startswith(original) and tagged.get("new_snapshot_id"):
                    return event | {"snapshot_id": tagged["new_snapshot_id"]}
            break
    raise LocalCommandError("Restic did not report a completed snapshot.")


def counts(database, *, integrity=False):
    with closing(readonly(database)) as db:
        db.execute("BEGIN")
        if integrity and db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise LocalCommandError("The restored database failed its integrity check.")
        values = db.execute("SELECT (SELECT count(*) FROM records), (SELECT count(*) FROM chunks)").fetchone()
        return dict(records=values[0], chunks=values[1])


def verify_snapshot(root, database, restic, snapshots):
    if not snapshots:
        raise LocalCommandError("No snapshot is available to verify.")
    latest = max(snapshots, key=lambda snapshot: datetime.fromisoformat(snapshot["time"]))
    paths = latest["paths"]
    if len(paths) != 1 or not Path(paths[0]).is_absolute() or ".." in Path(paths[0]).parts:
        raise LocalCommandError("The snapshot does not have a recognized backup root.")
    with TemporaryDirectory(dir=root / WORK) as directory:
        restic.run("restore", f"{latest['id']}:{paths[0]}", "--target", directory)
        target = Path(directory)
        staged = [path / STAGE for path in target.glob(f"{STAGE_PREFIX}*")
                  if path.is_dir() and not path.is_symlink()]
        staged = [path for path in staged if path.is_file() and not path.is_symlink()]
        legacy = target / STAGE
        if not staged and legacy.is_file() and not legacy.is_symlink():
            staged = [legacy]
        if len(staged) != 1:
            raise LocalCommandError("The backup does not contain exactly one staged database.")
        restored = counts(staged[0], integrity=True)
    live = counts(database)
    return dict(restored=restored, live=live, matches_live=restored == live)


def write_state(root, state):
    destination = root / STATE
    if destination.is_symlink() or (destination.exists() and not owned_state(destination)):
        raise LocalCommandError("The backup state path is occupied by an unmanaged file.")
    with NamedTemporaryFile(dir=root / WORK, mode="w", encoding="utf-8", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(state, file, separators=(",", ":"))
            file.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def owned_state(path):
    if not path.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(state, dict) and set(state) ==
            {"snapshot_id", "processed_bytes", "snapshots", "repository_bytes", "elapsed_ms"}
            and isinstance(state["snapshot_id"], str) and
            re.fullmatch(r"[0-9a-f]{64}", state["snapshot_id"]) is not None)


def backup(config, *, mode="backup", root=Path("local"), config_path=None):
    started = perf_counter()
    root = Path(root).resolve()
    options = config.get("backup")
    if not options:
        raise LocalCommandError("Configure the backup mount and repository first.")
    configured_database = Path(config["storage"]["database"]).absolute()
    database = configured_database.resolve()
    password = root / "secrets" / "restic-password"
    validate_database_path(configured_database, root, options["repository"], password)
    if database != configured_database:
        validate_database_path(database, root, options["repository"], password)
    if mode == "backup":
        require_backed_up_files(config, root / "config.toml" if config_path is None else config_path,
                                root, database, password)
        if (root / STATE).is_symlink() or ((root / STATE).exists() and not owned_state(root / STATE)):
            raise LocalCommandError("The backup state path is occupied by an unmanaged file.")
    with operation_lock(root, "backup") as acquired:
        if not acquired:
            return {"status": "busy"}
        with mounted_repository(options, initialize=mode == "init") as repository:
            restic = Restic(repository, password, root)
            if (root / WORK).is_symlink():
                raise LocalCommandError("The backup work directory must not be a symlink.")
            (root / WORK).mkdir(mode=0o700, exist_ok=True)
            snapshot = None
            if mode == "init":
                initialized = (repository.path / "config").is_file()
                check_password(password, create=not initialized)
                if not initialized:
                    restic.run("init")
                result = {"notice": PASSWORD_NOTICE}
            else:
                check_password(password)
                if mode == "verify":
                    result = verify_snapshot(root, database, restic, restic.snapshots())
                else:
                    snapshot = make_snapshot(config, root, database, restic)
                    restic.retain()
                    result = {"processed_bytes": snapshot["total_bytes_processed"]}
            result.update(snapshots=len(restic.snapshots()), repository_bytes=directory_bytes(repository.path),
                          elapsed_ms=round((perf_counter() - started) * 1000, 2))
            repository.check()
            if snapshot:
                write_state(root, result | {"snapshot_id": snapshot["snapshot_id"]})
            return result
