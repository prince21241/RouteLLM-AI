"""python -m app.evaluation"""

import sys

from app.evaluation.runner import main


def console_main(argv: list[str] | None = None) -> int:
    """Process entry. Cancelling the run exits 130 without a traceback."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("Evaluation cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(console_main())
