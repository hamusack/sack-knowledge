import argparse
import sys
from pathlib import Path

from .config import load_config
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
    commands.add_parser("mcp")
    commands.add_parser("embed")
    commands.add_parser("eval").add_argument("questions", nargs="?", default="local/questions.jsonl")
    sync_command = commands.add_parser("sync")
    sync_command.add_argument("source", choices=("discord",))
    sync_command.add_argument("--confirm-deletions", action="store_true")
    options = vars(parser.parse_args())
    command = options.pop("command")
    try:
        config = load_config(options.pop("config"))
        if command == "mcp":
            from .mcp_server import server

            server(config).run(transport="stdio")
            return 0
        if command == "sync":
            result = sync(config, confirm_deletions=options["confirm_deletions"])
        elif command == "embed":
            from .embedding import embed

            result = embed(config)
        elif command == "eval":
            from .evaluation import evaluate, table

            questions = Path(options["questions"]).resolve()
            if not questions.is_relative_to(Path("local").resolve()):
                raise ValueError("Use a question file under local/")
            print(table(evaluate(config, questions)))
            return 0
        else:
            result = getattr(Query(config), command)(**options)
        print(encode(result))
        if command == "sync" and result["deletions_held"]:
            return 1
    except Exception as error:
        print(f"{command} failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
