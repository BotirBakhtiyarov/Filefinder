"""Entry shim for ``flet build`` (app path = ``src``).

Ensures the ``filefinder`` package is importable when the app runs as a
plain script inside the packaged bundle, then delegates to the real entry
point.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from filefinder.main import main, run  # noqa: F401,E402

if __name__ == "__main__":
    run()
