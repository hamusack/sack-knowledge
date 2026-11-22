import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["", "grant", "password"])
def test_role_script_without_database(tmp_path, monkeypatch, failure):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "create_discord_reader.sh"
    script.write_bytes((Path(__file__).resolve().parents[1] / "scripts" / script.name).read_bytes())
    docker = tmp_path / "docker"
    docker.write_text("""#!/usr/bin/env python3
import os
import sys
from pathlib import Path
statement = sys.stdin.read()
failure = os.environ["TEST_ROLE_FAILURE"]
if "CREATE ROLE" in statement:
    assert "PASSWORD" not in statement
    assert 'GRANT CONNECT ON DATABASE :"reader_database" TO :"reader_role"' in statement
    assert "has_table_privilege" in statement and "has_any_column_privilege" in statement
    assert "ROLLBACK;" in statement
    step = "grant"
elif " PASSWORD " in statement:
    step = "password"
    print("synthetic output from password statement")
    print("synthetic output from password statement", file=sys.stderr)
else:
    step = "cleanup"
    assert 'DROP ROLE :"reader_role"' in statement
with Path(os.environ["TEST_ROLE_STEPS"]).open("a") as journal:
    journal.write(step + "\\n")
sys.exit(bool(failure and step == failure))
""")
    docker.chmod(0o700)
    steps = tmp_path / "steps"
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    monkeypatch.setenv("TEST_ROLE_STEPS", str(steps))
    monkeypatch.setenv("TEST_ROLE_FAILURE", failure)
    result = subprocess.run(["bash", str(script), "fixture", "fixture", "fixture", "fixture"],
                            capture_output=True, text=True, check=False)
    assert "synthetic output from password statement" not in result.stdout + result.stderr
    files = list((tmp_path / "local" / "secrets").iterdir())
    if failure:
        assert result.returncode != 0
        assert not files
        assert steps.read_text().splitlines() == (
            ["grant"] if failure == "grant" else ["grant", "password", "cleanup"]
        )
    else:
        assert result.returncode == 0
        assert steps.read_text().splitlines() == ["grant", "password"]
        assert len(files) == 1 and files[0].stat().st_mode & 0o777 == 0o600
