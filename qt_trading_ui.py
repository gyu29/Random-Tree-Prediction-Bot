"""Thin compatibility shim -- the Qt UI now lives in ui/. Kept so `python
qt_trading_ui.py` and any packaging scripts that reference this filename by
name still work; prefer `python main.py` going forward.
"""
from ui.app_window import launch_qt_app

if __name__ == "__main__":
    launch_qt_app()
