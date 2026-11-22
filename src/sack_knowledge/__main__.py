import argparse
import sys

from .config import load_config
from .query import Query, encode
from .sync import sync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local/config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--target", choices=("summary", "body"), default="summary")
    search.add_argument("--position", type=int, default=0)
    get = commands.add_parser("get")
    get.add_argument("ids", nargs="+")
    get.add_argument("--neighbors", type=int, default=0)
    count = commands.add_parser("count")
    count.add_argument("group_by", choices=("month", "person", "project", "source"))
    for command in (search, count):
        for field in ("start", "end", "person", "project", "source"):
            command.add_argument(f"--{field}")
    commands.add_parser("mcp")
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
