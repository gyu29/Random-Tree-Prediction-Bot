"""Entry point for the swing-trading desktop app."""


def launch_desktop_app():
    from ui.app_window import launch_qt_app

    return launch_qt_app()


if __name__ == "__main__":
    launch_desktop_app()
