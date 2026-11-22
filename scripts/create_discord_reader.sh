#!/usr/bin/env bash
# Create a read-only login role in the Discord collector's PostgreSQL.
#
# The password is generated here, written to local/secrets (git-ignored, 0600)
# and sent to psql over stdin, so it never appears on a command line or in
# this script's output.
#
# Usage: create_discord_reader.sh <container> <admin_role> <database> <role>
# The role may read only the tables listed in TABLES below.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <container> <admin_role> <database> <role>" >&2
  exit 2
fi

readonly container="$1"
readonly admin="$2"
readonly database="$3"
readonly role="$4"
readonly TABLES="public.messages, public.channels"

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
openssl rand -hex 24 > "$secret"
trap 'rm -f "$secret"' ERR

# The password is hex, so it needs no SQL escaping.
docker exec -i "$container" psql -U "$admin" -d "$database" -q -v ON_ERROR_STOP=1 <<SQL
BEGIN;
CREATE ROLE $role LOGIN PASSWORD '$(cat "$secret")'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
ALTER ROLE $role SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE $database TO $role;
GRANT USAGE ON SCHEMA public TO $role;
GRANT SELECT ON $TABLES TO $role;
COMMIT;
SQL

trap - ERR
echo "Read-only role created and password file saved."
