import json
import importlib
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import pytest

from sack_knowledge import backup, refresh
from sack_knowledge.errors import LocalCommandError
from sack_knowledge.operations import operation_lock


@pytest.fixture(autouse=True)
def mock_backup_host(monkeypatch):
    monkeypatch.setattr(backup, "backup_host", lambda: "fixture-host")


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
    (root / "config.toml").write_text("[backup]\n")
    backup.check_password(root / "secrets" / "restic-password", create=True)

    @contextmanager
    def mounted(options, *, initialize=False):
        yield SimpleNamespace(path=repository, check=lambda: None)

    monkeypatch.setattr(backup, "mounted_repository", mounted)
    config = {"storage": {"database": database}, "backup": {"repository": repository},
              "embedding": {"cache": root / "models"}}
    return config, root, repository, database


def test_backup_restic_commands_exclude_live_files_and_keep_staged_copy(backup_config, monkeypatch):
    config, root, repository, database = backup_config
    commands = []
    old = root / (backup.STAGE_PREFIX + "older")
    old.mkdir()
    (old / backup.STAGE).touch()
    active = dict(id="a" * 64, hostname=backup.backup_host(), paths=[str(root)], tags=[backup.TAG])

    def run(args, **options):
        commands.append(args)
        assert options["cwd"] == repository and options["capture_output"]
        if "backup" in args:
            staged = [path / backup.STAGE for path in root.glob(backup.STAGE_PREFIX + "*") if path != old]
            assert len(staged) == 1 and backup.counts(staged[0]) == {"records": 1, "chunks": 1}
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(
                message_type="summary", total_bytes_processed=100, snapshot_id="0123456789abcdef"
            )))
        if "snapshots" in args:
            return SimpleNamespace(returncode=0, stdout="[]" if backup.PENDING_TAG in args else json.dumps([active]))
        if "--dry-run" in args:
            return SimpleNamespace(returncode=0, stdout="[]")
        if "tag" in args:
            return SimpleNamespace(returncode=0, stdout="informational prelude\n" + json.dumps(dict(
                old_snapshot_id="0123456789abcdef", new_snapshot_id=active["id"]
            )))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(backup.subprocess, "run", run)
    result = backup.backup(config, root=root)
    assert result["snapshots"] == 1 and result["processed_bytes"] == 100
    assert commands[0][-1] == str(root)
    excluded = {commands[0][index + 1] for index, value in enumerate(commands[0]) if value == "--exclude"}
    assert {str(database), *(str(database) + suffix for suffix in ("-shm", "-wal", "-journal")),
            str(root / backup.WORK), ".venv", str(root / "models"), str(old)} <= excluded
    assert list(root.glob(backup.STAGE_PREFIX + "*")) == [old]
    assert commands[0][commands[0].index("--tag") + 1] == backup.PENDING_TAG
    assert commands[1][-4:] == ["tag", "--set", backup.TAG, "0123456789abcdef"]
    assert commands[2][-6:] == ["--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "12"]
    assert commands[4][-1] == "prune"
    assert json.loads((root / backup.STATE).read_text())["snapshot_id"] == active["id"]


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
    assert not (root / backup.STATE).exists() and not list(root.glob(backup.STAGE_PREFIX + "*"))


def test_backup_verifies_latest_restore_with_counts_and_cleans_temporary_files(backup_config, monkeypatch):
    config, root, _, database = backup_config
    own = dict(paths=[str(root)], hostname=backup.backup_host(), tags=[backup.TAG])
    old = dict(own, id="old", time="2025-01-01T00:00:00Z")
    new = dict(own, id="new", time="2025-01-02T00:00:00Z")
    foreign = dict(new, id="foreign", paths=[str(root), "/unrelated"], time="2025-01-03T00:00:00Z")
    targets = []

    def run(self, command, *args):
        if command == "snapshots":
            return json.dumps([new, old, foreign])
        assert command == "restore" and args[0] == f"new:{root}"
        assert args[1] == "--target" and len(args) == 3
        targets.append(Path(args[2]))
        target = targets[-1] / (backup.STAGE_PREFIX + "restored") / backup.STAGE
        target.parent.mkdir(parents=True)
        copyfile(database, target)

    monkeypatch.setattr(backup.Restic, "run", run)
    result = backup.backup(config, mode="verify", root=root)
    assert result["restored"] == result["live"] == {"records": 1, "chunks": 1}
    assert result["matches_live"] and not targets[0].exists()


def test_retention_scopes_expiry_and_pending_removal_to_exact_source(tmp_path, monkeypatch):
    restic = backup.Restic(None, None, tmp_path)
    owned = dict(id="owned", hostname=restic.host, paths=[str(tmp_path)], tags=[backup.TAG])
    pending = dict(owned, id="pending", tags=[backup.PENDING_TAG])
    foreign = [
        dict(owned, id="untagged", tags=[]),
        dict(owned, id="different-host", hostname="foreign"),
        dict(owned, id="different-root", paths=["/foreign"]),
        dict(owned, id="multiple-roots", paths=[str(tmp_path), "/foreign"]),
    ]
    calls = []

    def run(command, *args):
        calls.append((command, *args))
        if "--dry-run" in args:
            return json.dumps([{"remove": [owned, *foreign]}])
        if command == "snapshots":
            return json.dumps([pending, dict(pending, id="foreign-pending", paths=["/foreign"])])
        return ""

    monkeypatch.setattr(restic, "run", run)
    restic.retain()
    assert calls[0][:10] == ("forget", "--dry-run", "--tag", backup.TAG, "--host", restic.host,
                            "--path", str(tmp_path), "--group-by", "host,paths,tags")
    assert calls[1] == ("snapshots", "--tag", backup.PENDING_TAG, "--host", restic.host, "--path", str(tmp_path))
    assert calls[2:] == [("forget", "owned", "pending"), ("prune",)]


@pytest.mark.parametrize("relative", [backup.STATE, backup.WORK + "/live", backup.STAGE,
                                     backup.STAGE_PREFIX + "previous/live", ".backup.lock", "."])
def test_live_database_may_not_overlap_backup_paths(backup_config, relative):
    config, root, _, _ = backup_config
    path = root / relative
    config["storage"]["database"] = path
    if relative == backup.STATE:
        create_database(path)
    with pytest.raises(LocalCommandError, match="overlaps a backup output path"):
        backup.backup(config, root=root)
    if relative == backup.STATE:
        assert backup.counts(path) == {"records": 1, "chunks": 1}


@pytest.mark.parametrize("setting", ["--config", "discord.password_file"])
def test_required_symlinks_are_rejected_before_backup(backup_config, setting):
    config, root, _, _ = backup_config
    regular = root / "regular"
    regular.touch()
    if setting == "--config":
        link = root / "config.toml"
        link.unlink()
    else:
        link = root / "secret-link"
        config["discord"] = {"password_file": link}
    link.symlink_to(regular)
    with pytest.raises(LocalCommandError) as error:
        backup.backup(config, root=root)
    assert setting in str(error.value)
    assert str(root) not in str(error.value)


def test_repository_must_still_be_pinned_after_each_restic_call(tmp_path, monkeypatch):
    root, mount = tmp_path / "local", tmp_path / "mount"
    root.mkdir()
    mount.mkdir()
    repo = mount / "backup"
    repo.mkdir()
    monkeypatch.setattr(Path, "is_mount", lambda path: path == mount)
    db = root / "live.sqlite3"
    create_database(db)
    (root / "config.toml").touch()
    backup.check_password(root / "secrets" / "restic-password", create=True)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["cwd"].samefile(repo)
        repo.rename(mount / "detached")
        repo.mkdir()
        return SimpleNamespace(returncode=0, stdout='{"message_type":"summary","snapshot_id":"fixture"}')

    monkeypatch.setattr(backup.subprocess, "run", run)
    config = {"storage": {"database": db}, "backup": {"mount_point": mount, "repository": repo}}
    with pytest.raises(LocalCommandError, match="mount changed"):
        backup.backup(config, root=root)
    assert len(calls) == 1 and "backup" in calls[0]
    assert not (root / backup.STATE).exists()


def test_install_migrates_legacy_and_preserves_foreign_or_modified_units(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    units = importlib.import_module("systemd_units")
    install = importlib.import_module("install_systemd")
    uninstall = importlib.import_module("uninstall_systemd")
    target = tmp_path / "units"
    target.mkdir()
    calls = []
    for module in (install, uninstall):
        monkeypatch.setattr(module, "destination", lambda: target)
        monkeypatch.setattr(module, "systemctl", lambda *args: calls.append(args))
    foreign = tmp_path / "foreign"
    foreign.write_text("foreign fixture")
    service = target / units.UNITS[0]
    service.symlink_to(foreign)
    with pytest.raises(SystemExit, match="--force"):
        install.main()
    assert service.is_symlink() and not calls
    install.main(force=True)
    assert not service.is_symlink() and foreign.read_text() == "foreign fixture"
    for name in units.UNITS:
        (target / name).write_text(units.render((units.SOURCE / name).read_text()))
    install.main()
    assert all(units.owned((target / name).read_bytes()) for name in units.UNITS)
    modified = target / units.TIMERS[0]
    modified.write_bytes(modified.read_bytes() + b"# edited\n")
    calls.clear()
    uninstall.main()
    assert modified.exists() and sorted(path.name for path in target.iterdir()) == [modified.name]
    assert calls == [("disable", "--now", units.TIMERS[1]), ("daemon-reload",)]
    assert modified.name in capsys.readouterr().out


def test_install_refuses_a_foreign_unit_created_after_preflight(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    units = importlib.import_module("systemd_units")
    install = importlib.import_module("install_systemd")
    path = tmp_path / units.UNITS[0]
    monkeypatch.setattr(install, "destination", lambda: tmp_path)
    monkeypatch.setattr(install, "systemctl", lambda *args: path.write_text("foreign fixture"))
    with pytest.raises(SystemExit, match="changed during installation"):
        install.main()
    assert path.read_text() == "foreign fixture"
