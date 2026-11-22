import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import pytest

from sack_knowledge import backup, refresh
from sack_knowledge.errors import LocalCommandError
from sack_knowledge.operations import operation_lock


def create_database(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE records (id INTEGER PRIMARY KEY);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY);
            INSERT INTO records DEFAULT VALUES;
            INSERT INTO chunks DEFAULT VALUES;
        """)


def test_refresh_lock_skips_an_overlapping_run(tmp_path, monkeypatch):
    def unexpected(config):
        pytest.fail("An overlapping refresh must not run sync or embed")

    monkeypatch.setattr(refresh, "sync", unexpected)
    monkeypatch.setattr(refresh, "embed", unexpected)
    with operation_lock(tmp_path, "refresh"):
        assert refresh.refresh({}, tmp_path) == {"status": "busy"}
    with operation_lock(tmp_path, "refresh") as acquired:
        assert acquired


@pytest.mark.parametrize("phase", ["sync", "embed"])
def test_refresh_records_safe_failure_and_next_run_retries(tmp_path, monkeypatch, phase):
    database = tmp_path / "knowledge.sqlite3"
    config = {"storage": {"database": database}, "discord": {}}
    calls = []
    failing = True

    def sync(config):
        calls.append("sync")
        if failing and phase == "sync":
            raise RuntimeError("synthetic source details")
        return dict(inserted=1, updated=0, deleted=0, chunks_built=1, chunks_removed=0,
                    deletions_held=[{"missing": 4}])

    def embed(config):
        calls.append("embed")
        if failing and phase == "embed":
            raise RuntimeError("synthetic model details")
        return {"embedded": 1}

    monkeypatch.setattr(refresh, "sync", sync)
    monkeypatch.setattr(refresh, "embed", embed)
    assert refresh.refresh(config, tmp_path)["status"] == "failed"
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT last_error FROM sync_state").fetchone()[0] == f"{phase}:RuntimeError"
    failing = False
    result = refresh.refresh(config, tmp_path)
    assert (result["status"], result["embedded"], result["deletions_held"]) == ("success", 1, 4)
    assert calls == (["sync"] if phase == "sync" else ["sync", "embed"]) + ["sync", "embed"]
    with closing(sqlite3.connect(database)) as db:
        success, error = db.execute("SELECT last_success, last_error FROM sync_state").fetchone()
        assert success and error is None


def test_backup_refuses_unmounted_destination_before_creating_repository(tmp_path):
    root = tmp_path / "local"
    mount = tmp_path / "mount"
    mount.mkdir()
    repository = mount / "backup"
    config = {"storage": {"database": root / "knowledge.sqlite3"},
              "backup": {"mount_point": mount, "repository": repository}}
    with pytest.raises(LocalCommandError, match="not mounted"):
        backup.backup(config, root=root, mode="init")
    assert not repository.exists() and not (root / "secrets").exists()


def test_staging_is_consistent_with_an_active_writer(tmp_path):
    live = tmp_path / "live.sqlite3"
    copy = tmp_path / "copy.sqlite3"
    create_database(live)
    with closing(sqlite3.connect(live)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO records DEFAULT VALUES")
        writer.execute("DELETE FROM chunks")
        backup.stage_database(live, copy)
        writer.commit()
    assert backup.counts(copy, integrity=True) == {"records": 1, "chunks": 1}
    assert backup.counts(live) == {"records": 2, "chunks": 0}
    assert not (tmp_path / "copy.sqlite3-wal").exists()


@pytest.fixture
def backup_config(tmp_path, monkeypatch):
    root, repository = tmp_path / "local", tmp_path / "mounted-repository"
    root.mkdir()
    repository.mkdir()
    database = root / "knowledge.sqlite3"
    create_database(database)
    backup.check_password(root / "secrets" / "restic-password", create=True)

    @contextmanager
    def mounted(options, *, initialize=False):
        yield repository

    monkeypatch.setattr(backup, "mounted_repository", mounted)
    config = {"storage": {"database": database}, "backup": {"configured": True},
              "embedding": {"cache": root / "models"}}
    return config, root, repository, database


def test_backup_restic_commands_exclude_live_files_and_keep_staged_copy(backup_config, monkeypatch):
    config, root, repository, database = backup_config
    commands = []

    def run(args, **options):
        commands.append(args)
        assert options["cwd"] == repository and options["capture_output"]
        if "backup" in args:
            assert backup.counts(root / backup.STAGE) == {"records": 1, "chunks": 1}
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(
                message_type="summary", total_bytes_processed=100, snapshot_id="0123456789abcdef"
            )))
        if "snapshots" in args:
            return SimpleNamespace(returncode=0, stdout="[]" if backup.PENDING_TAG in args else "[{}]")
        if "tag" in args:
            return SimpleNamespace(returncode=0, stdout="informational prelude\n" + json.dumps(dict(
                old_snapshot_id="0123456789abcdef", new_snapshot_id="fedcba9876543210"
            )))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(backup.subprocess, "run", run)
    result = backup.backup(config, root=root)
    assert result["snapshots"] == 1 and result["processed_bytes"] == 100
    assert commands[0][-1] == str(root)
    excluded = {commands[0][index + 1] for index, value in enumerate(commands[0]) if value == "--exclude"}
    assert {str(database), *(str(database) + suffix for suffix in ("-shm", "-wal", "-journal")),
            str(root / backup.WORK), ".venv", str(root / "models")} <= excluded
    assert str(root / backup.STAGE) not in excluded and not (root / backup.STAGE).exists()
    assert commands[0][commands[0].index("--tag") + 1] == backup.PENDING_TAG
    assert commands[1][-4:] == ["tag", "--set", backup.TAG, "0123456789abcdef"]
    assert commands[3][-6:] == ["--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "12"]
    assert commands[4][-1] == "prune"
    assert json.loads((root / backup.STATE).read_text())["snapshot_id"] == "fedcba9876543210"


def test_partial_backup_is_never_promoted(backup_config, monkeypatch):
    config, root, _, _ = backup_config
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=3, stdout='{"snapshot_id":"fixture"}')

    monkeypatch.setattr(backup.subprocess, "run", run)
    with pytest.raises(LocalCommandError, match="exit 3"):
        backup.backup(config, root=root)
    assert len(calls) == 1 and backup.PENDING_TAG in calls[0]
    assert not (root / backup.STATE).exists() and not (root / backup.STAGE).exists()


def test_backup_verifies_latest_restore_with_counts_and_cleans_temporary_files(backup_config, monkeypatch):
    config, root, _, database = backup_config
    old = dict(id="old", paths=[str(root)], time="2025-01-01T00:00:00Z")
    new = dict(id="new", paths=[str(root)], time="2025-01-02T00:00:00Z")
    targets = []

    def run(self, command, *args):
        if command == "snapshots":
            return json.dumps([new, old])
        assert command == "restore" and args[0] == "new"
        assert args[1] == "--target" and len(args) == 3
        targets.append(Path(args[2]))
        target = targets[-1] / root.relative_to("/") / backup.STAGE
        target.parent.mkdir(parents=True)
        copyfile(database, target)

    monkeypatch.setattr(backup.Restic, "run", run)
    result = backup.backup(config, mode="verify", root=root)
    assert result["restored"] == result["live"] == {"records": 1, "chunks": 1}
    assert result["matches_live"] and not targets[0].exists()
