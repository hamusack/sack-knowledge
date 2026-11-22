import argparse
import sys
from pathlib import Path

from .config import load_config
from .errors import LocalCommandError
from .query import Query, encode
from .sync import sync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local/config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--target", choices=("summary", "body"))
    search.add_argument("--position", type=int, default=0)
    search.add_argument("--mode", choices=("fts", "vector", "hybrid"))
    get = commands.add_parser("get")
    get.add_argument("ids", nargs="+")
    get.add_argument("--neighbors", type=int, default=0)
    count = commands.add_parser("count")
    count.add_argument("group_by", choices=("month", "person", "project", "source"))
    for command in (search, count):
        for field in ("start", "end", "person", "project", "source"):
            command.add_argument(f"--{field}")
    mcp = commands.add_parser("mcp")
    for command in (search, get, count, mcp):
        command.add_argument("--profile")
    commands.add_parser("embed")
    commands.add_parser("refresh")
    backup_command = commands.add_parser("backup")
    backup_mode = backup_command.add_mutually_exclusive_group()
    backup_mode.add_argument("--init", action="store_const", const="init", dest="mode")
    backup_mode.add_argument("--verify", action="store_const", const="verify", dest="mode")
    commands.add_parser("eval").add_argument("questions", nargs="?", default="local/questions.jsonl")
    sync_command = commands.add_parser("sync")
    sync_command.add_argument("source", choices=("discord",))
    sync_command.add_argument("--confirm-deletions", action="store_true")
    options = vars(parser.parse_args())
    command = options.pop("command")
    try:
        config_path = options.pop("config")
        config = load_config(config_path)
        if command == "mcp":
            from .mcp_server import server

            server(config, profile_name=options["profile"]).run(transport="stdio")
            return 0
        if command == "sync":
            result = sync(config, confirm_deletions=options["confirm_deletions"])
        elif command == "embed":
            from .embedding import embed

            result = embed(config)
        elif command == "refresh":
            from .refresh import refresh

            result = refresh(config)
        elif command == "backup":
            from .backup import backup

            result = backup(config, mode=options["mode"] or "backup", config_path=config_path)
        elif command == "eval":
            from .evaluation import evaluate, table

            questions = Path(options["questions"]).resolve()
            if not questions.is_relative_to(Path("local").resolve()):
                raise ValueError("Use a question file under local/")
            print(table(evaluate(config, questions)))
            return 0
        else:
            result = getattr(Query(config, profile_name=options.pop("profile")), command)(**options)
        print(encode(result))
        if command == "sync" and result["deletions_held"]:
            return 1
        if command == "refresh" and result["status"] == "failed":
            return 1
        if command == "backup" and result.get("matches_live") is False:
            return 1
    except LocalCommandError as error:
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"{command} failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
