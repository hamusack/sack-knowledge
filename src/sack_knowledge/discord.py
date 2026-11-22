import ipaddress
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect(config):
    import psycopg
    from psycopg.rows import dict_row

    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", "--", config["container"]],
        capture_output=True, check=True, timeout=15,
    )
    host = str(ipaddress.ip_address(json.loads(result.stdout)[config["network"]]["IPAddress"]))
    with psycopg.connect(
        host=host, port=5432, dbname=config["database"], user=config["role"],
        password=Path(config["password_file"]).read_text().rstrip("\r\n"),
        connect_timeout=10, row_factory=dict_row,
    ) as connection:
        connection.read_only = True
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        yield connection


class DiscordSource:
    def __init__(self, connection):
        self.connection = connection

    def manifest(self, guild):
        rows = self.connection.execute("""
            SELECT m.message_id::text AS source_id,
                   md5(ROW(m.channel_id, m.author_user_id, m.author_display_name,
                           m.reply_to_message_id, m.content, m.mentioned_user_ids,
                           m.created_at, m.edited_at, c.name)::text) AS fingerprint
            FROM public.messages m
            LEFT JOIN public.channels c USING (guild_id, channel_id)
            WHERE m.guild_id::text = %s
        """, (guild,))
        return {row["source_id"]: row["fingerprint"] for row in rows}

    def messages(self, guild, ids):
        for offset in range(0, len(ids), 5000):
            yield from self.connection.execute("""
                SELECT m.message_id::text AS source_id, m.channel_id::text AS channel_id,
                       m.author_user_id::text AS author_id, m.author_display_name AS author_name,
                       m.reply_to_message_id::text AS reply_id, m.content AS body,
                       m.mentioned_user_ids AS mentions, m.created_at, m.edited_at, c.name AS title
                FROM public.messages m
                LEFT JOIN public.channels c USING (guild_id, channel_id)
                WHERE m.guild_id::text = %s AND m.message_id::text = ANY(%s)
            """, (guild, ids[offset:offset + 5000]))
