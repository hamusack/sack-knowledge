import json
import os
import re
import secrets
import sqlite3
import stat
import subprocess
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

from .errors import LocalCommandError
from .operations import operation_lock


RESTIC = "/usr/bin/restic"
TAG = "sack-knowledge"
PENDING_TAG = "sack-knowledge-pending"
STAGE = ".backup-database.sqlite3"
WORK = ".backup-work"
STATE = "backup-state.json"
PASSWORD_NOTICE = "Store a separate copy of the backup password file outside the backup repository."


@contextmanager
def mounted_repository(options, *, initialize=False):
    mount = Path(options["mount_point"]).resolve()
    repository = Path(options["repository"]).absolute()
    if repository.parent.resolve() != mount or repository.name in ("", ".", ".."):
        raise LocalCommandError("The backup repository must be a directory directly under the configured mount.")
    if not mount.is_mount():
        raise LocalCommandError("The backup destination is not mounted.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    mount_fd = os.open(mount, flags)
    try:
        if not mount.is_mount() or not os.path.samestat(os.fstat(mount_fd), mount.stat()):
            raise LocalCommandError("The backup destination mount changed.")
        if initialize:
            try:
                os.mkdir(repository.name, mode=0o700, dir_fd=mount_fd)
            except FileExistsError:
                pass
        try:
            repo_fd = os.open(repository.name, flags, dir_fd=mount_fd)
        except FileNotFoundError:
            raise LocalCommandError("Initialize the backup repository first.") from None
        try:
            # Restic runs from the pinned directory, including if the mount is detached.
            yield Path(f"/proc/self/fd/{repo_fd}")
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
    def __init__(self, repository, password):
        self.repository = repository
        self.password = password

    def run(self, command, *args):
        env = {key: value for key, value in os.environ.items() if not key.startswith("RESTIC_")}
        result = subprocess.run(
            [RESTIC, "--repo", ".", "--password-file", str(self.password), "--no-cache", "--json",
             command, *args], cwd=self.repository, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise LocalCommandError(f"Restic {command} failed (exit {result.returncode}).")
        return result.stdout

    def snapshots(self, tag=TAG):
        return json.loads(self.run("snapshots", "--tag", tag)) or []


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


def exclusions(config, root, database):
    paths = {database, *(database.with_name(database.name + suffix)
                         for suffix in ("-wal", "-shm", "-journal")),
             root / WORK, root / ".backup.lock", root / ".refresh.lock",
             Path(config.get("embedding", {}).get("cache", "local/models")).resolve()}
    return [argument for path in sorted(paths) for argument in ("--exclude", literal_pattern(path))] + [
        "--exclude", ".venv",
    ]


def directory_bytes(path):
    return sum((Path(folder) / name).stat().st_size for folder, _, names in os.walk(path)
               for name in names)


def make_snapshot(config, root, database, restic):
    stage = root / STAGE
    with TemporaryDirectory(dir=root / WORK) as directory:
        pending = Path(directory) / STAGE
        pending.touch(mode=0o600, exist_ok=False)
        stage_database(database, pending)
        pending.replace(stage)
        try:
            output = restic.run("backup", "--host", TAG, "--tag", PENDING_TAG,
                                *exclusions(config, root, database), str(root))
        finally:
            stage.unlink(missing_ok=True)
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
    recorded_stage = Path(paths[0]) / STAGE
    with TemporaryDirectory(dir=root / WORK) as directory:
        restic.run("restore", latest["id"], "--target", directory)
        restored = counts(Path(directory) / recorded_stage.relative_to("/"), integrity=True)
    live = counts(database)
    return dict(restored=restored, live=live, matches_live=restored == live)


def write_state(root, state):
    with NamedTemporaryFile(dir=root / WORK, mode="w", encoding="utf-8", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(state, file, separators=(",", ":"))
            file.close()
            temporary.replace(root / STATE)
        finally:
            temporary.unlink(missing_ok=True)


def backup(config, *, mode="backup", root=Path("local")):
    started = perf_counter()
    root = Path(root).resolve()
    options = config.get("backup")
    if not options:
        raise LocalCommandError("Configure the backup mount and repository first.")
    database = Path(config["storage"]["database"]).resolve()
    if database == root / STAGE:
        raise LocalCommandError("The live database cannot use the backup staging path.")
    password = root / "secrets" / "restic-password"
    with operation_lock(root, "backup") as acquired:
        if not acquired:
            return {"status": "busy"}
        with mounted_repository(options, initialize=mode == "init") as repository:
            restic = Restic(repository, password)
            (root / WORK).mkdir(mode=0o700, exist_ok=True)
            snapshot = None
            if mode == "init":
                initialized = (repository / "config").is_file()
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
                    abandoned = [item["id"] for item in restic.snapshots(PENDING_TAG)]
                    if abandoned:
                        restic.run("forget", *abandoned)
                    restic.run("forget", "--tag", TAG, "--group-by", "tags",
                               "--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "12")
                    restic.run("prune")
                    result = {"processed_bytes": snapshot["total_bytes_processed"]}
            result.update(snapshots=len(restic.snapshots()), repository_bytes=directory_bytes(repository),
                          elapsed_ms=round((perf_counter() - started) * 1000, 2))
            if snapshot:
                write_state(root, result | {"snapshot_id": snapshot["snapshot_id"]})
            return result
