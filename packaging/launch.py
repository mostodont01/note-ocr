"""Entry point of the packaged app: no arguments (or a file) -> GUI; a command -> CLI."""

import sys

COMMANDS = {"recognize", "app", "retrain", "install-desktop", "install-firefox", "-h", "--help"}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from notocr.cli import main

        main()
    else:
        from notocr.app.backend import main

        sys.exit(main())
