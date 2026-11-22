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
    docker.write_text(r"""#!/usr/bin/env python3
import base64
import hashlib
import hmac
import os
import re
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
    secrets = Path(os.environ["TEST_ROLE_STEPS"]).parent / "local" / "secrets"
    password = next(secrets.iterdir()).read_bytes().rstrip()
    assert password not in statement.encode()
    match = re.fullmatch(r'''ALTER ROLE :"reader_role" PASSWORD 'SCRAM-SHA-256\$(\d+):([^$]+)\$([^:]+):([^']+)';\n''', statement)
    assert match
    count, salt, stored, server = match.groups()
    key = hashlib.pbkdf2_hmac("sha256", password, base64.b64decode(salt), int(count))
    client = hmac.new(key, b"Client Key", "sha256").digest()
    assert base64.b64decode(stored) == hashlib.sha256(client).digest()
    assert base64.b64decode(server) == hmac.new(key, b"Server Key", "sha256").digest()
    print("synthetic output from password statement")
    print("synthetic output from password statement", file=sys.stderr)
else:
    step = "cleanup"
    assert 'DROP ROLE :"reader_role"' in statement
with Path(os.environ["TEST_ROLE_STEPS"]).open("a") as journal:
    journal.write(step + "\n")
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
