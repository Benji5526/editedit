"""Desktop app launcher:  python app.py [video]   (or double-click run_app.bat, or VideoCensor.exe when packaged)

--selftest-file <path.json>   show the window, write an environment report to the file and exit (used to verify a build)
"""
import json
import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.paths import app_root, is_frozen, worker_command  # noqa: E402

HERE = app_root()           # project folder, or the folder of the exe when packaged


def _selftest_report(win) -> dict:
    import shutil
    from PySide6 import QtCore
    from src.app.process_dialog import find_default_checkpoint
    from src.video_io import VideoError, find_tool

    rep = {"frozen": is_frozen(), "app_root": HERE, "title": win.windowTitle(), "qt": QtCore.qVersion(),
           "worker": worker_command()[0], "worker_exists": os.path.isfile(worker_command()[0]),
           "default_checkpoint": find_default_checkpoint() or None, "python": sys.version.split()[0]}
    for tool in ("ffmpeg", "ffprobe"):
        try:
            rep[tool] = find_tool(tool)
        except VideoError as e:
            rep[tool] = f"NOT FOUND: {e}"
    rep["gpu_libs_in_gui_process"] = "torch" in sys.modules          # the GUI must stay light: torch belongs to the worker
    rep["ffmpeg_on_path"] = shutil.which("ffmpeg")
    return rep


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    for stream in ("stdout", "stderr"):                              # a windowed exe has no console: libraries must not crash on print()
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w", encoding="utf-8"))
    selftest = job_selftest = None
    if "--selftest-file" in argv:
        i = argv.index("--selftest-file")
        selftest = argv[i + 1]
        del argv[i:i + 2]
    if "--selftest-job" in argv:
        i = argv.index("--selftest-job")
        job_selftest = argv[i + 1]
        del argv[i:i + 2]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(HERE, "logs", "app.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", encoding="utf-8")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from src.app.main_window import MainWindow

    app = QApplication(argv[:1])
    from src.app.theme import apply_theme
    apply_theme(app)

    def hook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        logging.error("unhandled exception:\n%s", text)
        QMessageBox.critical(None, "Video Censor - 예기치 않은 오류", f"{v}\n\n자세한 내용은 logs/app.log에 기록했습니다.")

    sys.excepthook = hook
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.open_video(argv[1])
    if job_selftest:
        from src.app import selftest as selftest_mod
        selftest_mod.install(app, win, argv[1], job_selftest)
    if selftest:
        def finish():
            try:
                rep = _selftest_report(win)
            except Exception as e:  # noqa: BLE001
                rep = {"error": f"{type(e).__name__}: {e}"}
            with open(selftest, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)
            app.quit()
        QTimer.singleShot(2500, finish)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
