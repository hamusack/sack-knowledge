import os

from systemd_units import TIMERS, UNITS, destination, owned, systemctl, unit_contents, unit_stamp


def main():
    target = destination()
    stamps = {name: unit_stamp(target / name) for name in UNITS}
    installed = {name for name in UNITS if owned(unit_contents(target / name))}
    left = [name for name in UNITS if name not in installed and os.path.lexists(target / name)]
    timers = [name for name in TIMERS if name in installed and unit_stamp(target / name) == stamps[name]]
    if timers:
        systemctl("disable", "--now", *timers)
    for name in sorted(installed):
        path = target / name
        if unit_stamp(path) == stamps[name] and owned(unit_contents(path)):
            path.unlink(missing_ok=True)
        elif os.path.lexists(path):
            left.append(name)
    if installed:
        systemctl("daemon-reload")
    if left:
        print("Left unmanaged or modified units: " + ", ".join(left))


if __name__ == "__main__":
    try:
        main()
    except OSError:
        raise SystemExit("Unable to uninstall the user units.") from None
