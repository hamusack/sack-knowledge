#!/usr/bin/env bash
# Create a read-only login role in the Discord collector's PostgreSQL.
#
# The password is generated here and written to local/secrets (git-ignored, 0600).
# Only its SCRAM verifier is sent to PostgreSQL, over stdin.
#
# Usage: create_discord_reader.sh <container> <admin_role> <database> <role>
# The role may read only the two application tables granted below.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <container> <admin_role> <database> <role>" >&2
  exit 2
fi

readonly container="$1"
readonly admin="$2"
readonly database="$3"
readonly role="$4"

for name in "$admin" "$database" "$role"; do
  if [[ ! "$name" =~ ^[a-z_][a-z0-9_]*$ ]]; then
    echo "Expected plain identifiers for the admin, database and role." >&2
    exit 2
  fi
done

root="$(cd "$(dirname "$0")/.." && pwd)"
readonly secret="${root}/local/secrets/${role}_password"

if [[ -e "$secret" ]]; then
  echo "Password file already exists." >&2
  exit 1
fi

install -d -m 0700 "$root/local/secrets"
umask 077
temp="$(mktemp "${secret}.XXXXXX")"
created=0

psql() {
  docker exec -i "$container" psql -X -U "$admin" -d "$database" -q -v ON_ERROR_STOP=1 \
    -v "reader_role=$role" -v "reader_database=$database"
}

cleanup() {
  rm -f -- "$temp"
  if [[ "$created" == 1 ]]; then
    if ! psql <<'SQL'
BEGIN;
DROP OWNED BY :"reader_role";
DROP ROLE :"reader_role";
COMMIT;
SQL
    then
      echo "Could not remove the incomplete role." >&2
    fi
  fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
openssl rand -hex 24 > "$temp"

psql <<'SQL'
BEGIN;
CREATE ROLE :"reader_role" LOGIN
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE :"reader_role" SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE :"reader_database" TO :"reader_role";
GRANT USAGE ON SCHEMA public TO :"reader_role";
GRANT SELECT ON public.messages, public.channels TO :"reader_role";

WITH relations AS (
    SELECT c.oid, n.nspname = 'public' AND c.relname IN ('messages', 'channels') AS intended
    FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema')
      AND n.nspname !~ '^pg_(temp|toast)'
)
SELECT count(*) FILTER (WHERE intended) = 2 AND bool_and(
    CASE WHEN intended THEN
        has_table_privilege(:'reader_role', oid, 'SELECT')
        AND NOT has_table_privilege(:'reader_role', oid, 'INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
        AND NOT has_any_column_privilege(:'reader_role', oid, 'INSERT, UPDATE, REFERENCES')
    ELSE
        NOT has_any_column_privilege(:'reader_role', oid, 'SELECT')
    END
) AS reader_grants_ok FROM relations
\gset
\if :reader_grants_ok
COMMIT;
\else
ROLLBACK;
\warn Read-only role grants failed verification.
SELECT 1 / 0;
\endif
SQL
created=1

verifier="$(python3 - "$temp" <<'PY'
import base64
import hashlib
import hmac
import secrets
import sys
from pathlib import Path

# Hex passwords are ASCII and need no SASLprep normalization.
password = Path(sys.argv[1]).read_bytes().rstrip(b"\n")
salt = secrets.token_bytes(16)
iterations = 4096
salted = hashlib.pbkdf2_hmac("sha256", password, salt, iterations)
client_key = hmac.digest(salted, b"Client Key", "sha256")
stored_key = hashlib.sha256(client_key).digest()
server_key = hmac.digest(salted, b"Server Key", "sha256")

def encode(value):
    return base64.b64encode(value).decode("ascii")

print(f"SCRAM-SHA-256${iterations}:{encode(salt)}${encode(stored_key)}:{encode(server_key)}")
PY
)"
if ! psql >/dev/null 2>&1 <<SQL
ALTER ROLE :"reader_role" PASSWORD '$verifier';
SQL
then
  echo "Could not set the role password." >&2
  exit 1
fi
mv -T --no-clobber -- "$temp" "$secret"
if [[ -e "$temp" ]]; then
  echo "Password file already exists." >&2
  exit 1
fi
created=0
echo "Read-only role created and password file saved."
