"""Desktop app launcher:  python app.py [video]   (or double-click run_app.bat)"""
import logging
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(HERE, "logs", "app.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", encoding="utf-8")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from src.app.main_window import MainWindow

    app = QApplication(argv[:1])

    def hook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        logging.error("unhandled exception:\n%s", text)
        QMessageBox.critical(None, "Video Censor - unexpected error", f"{v}\n\nDetails were written to logs/app.log")

    sys.excepthook = hook
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.open_video(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
