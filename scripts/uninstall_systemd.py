from systemd_units import TIMERS, UNITS, destination, systemctl


def main():
    systemctl("disable", "--now", *TIMERS)
    target = destination()
    for name in UNITS:
        (target / name).unlink(missing_ok=True)
    systemctl("daemon-reload")


if __name__ == "__main__":
    try:
        main()
    except OSError:
        raise SystemExit("Unable to uninstall the user units.") from None
