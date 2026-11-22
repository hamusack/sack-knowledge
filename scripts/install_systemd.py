import argparse
import os

from systemd_units import SOURCE, TIMERS, UNITS, destination, owned, render, systemctl, unit_contents, unit_stamp, write_unit


def main(*, force=False):
    target = destination()
    bodies = {name: render((SOURCE / name).read_text()).encode("utf-8") for name in UNITS}
    stamps = {name: unit_stamp(target / name) for name in UNITS}
    existing = {name: unit_contents(target / name) for name in UNITS}
    # Only the complete exact set rendered by the previous installer may migrate without markers.
    legacy = existing == bodies
    foreign = [name for name in UNITS if os.path.lexists(target / name) and not owned(existing[name])]
    if foreign and not (force or legacy):
        raise SystemExit("Use --force to replace unmanaged or modified units: " + ", ".join(foreign))
    systemctl("show-environment")
    target.mkdir(parents=True, exist_ok=True)
    for name in UNITS:
        write_unit(target / name, bodies[name], stamps[name])
    systemctl("daemon-reload")
    systemctl("enable", "--now", *TIMERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    try:
        main(force=parser.parse_args().force)
    except OSError:
        raise SystemExit("Unable to install the user units.") from None
