from systemd_units import SOURCE, TIMERS, UNITS, destination, render, systemctl


def main():
    systemctl("show-environment")
    target = destination()
    target.mkdir(parents=True, exist_ok=True)
    for name in UNITS:
        (target / name).write_text(render((SOURCE / name).read_text()), encoding="utf-8")
    systemctl("daemon-reload")
    systemctl("enable", "--now", *TIMERS)


if __name__ == "__main__":
    try:
        main()
    except OSError:
        raise SystemExit("Unable to install the user units.") from None
