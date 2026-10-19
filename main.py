from __future__ import annotations
"""
main.py
--------
Entry point for EyeGuard AI. Creates the main application window and
starts the CustomTkinter event loop. All setup that can fail gracefully
(missing camera, missing API key, missing optional dependencies) is
handled inside the modules themselves so a partial-failure environment
still launches a usable app.
"""


import sys
import traceback


def main() -> int:
    try:
        from eyeguard_ui import EyeGuardApp
    except ImportError as exc:
        print("EyeGuard AI failed to start because a required package is missing.")
        print(f"Details: {exc}")
        print("Run 'pip install -r requirements.txt' and try again.")
        return 1
    
    try:
        app = EyeGuardApp()
        app.mainloop()
    except Exception:
        print("EyeGuard AI encountered an unexpected error and had to close:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
    