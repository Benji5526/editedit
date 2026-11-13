"""Dark, editor-style look for the desktop app (one stylesheet + palette, applied to the QApplication)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#17181c"          # window
PANEL = "#1f2126"       # docks, panels
RAISED = "#2a2d34"      # buttons, inputs
BORDER = "#34373f"
TEXT = "#e6e7ea"
MUTED = "#9aa0aa"
ACCENT = "#4c8dff"
ACCENT_HOVER = "#6aa1ff"
OK = "#5cc46a"
WARN = "#e0a030"
ERROR = "#ff6b6b"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 10pt; }}
QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}

QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {RAISED}; border-radius: 4px; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 26px 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: #5d626c; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; padding: 5px 8px; spacing: 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 6px 12px; }}
QToolButton:hover {{ background: {RAISED}; }}
QToolButton:checked {{ background: #24344f; border-color: {ACCENT}; color: white; }}
QToolButton:disabled {{ color: #5d626c; }}
QToolButton#primary {{ background: {ACCENT}; color: white; font-weight: bold; padding: 6px 18px; }}
QToolButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QToolButton#primary:disabled {{ background: #2c3a55; color: #7f8794; }}

QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; font-weight: bold; }}
QDockWidget::title {{ background: {PANEL}; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
QDockWidget > QWidget {{ background: {PANEL}; }}
QDockWidget QLabel, QDockWidget QCheckBox, QDockWidget QGroupBox {{ background: transparent; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 7px 16px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: #343842; }}
QPushButton:pressed {{ background: #24272d; }}
QPushButton:disabled {{ color: #5d626c; background: #202227; }}
QPushButton#play {{ background: {ACCENT}; color: white; font-weight: bold; min-width: 76px; }}
QPushButton#play:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#play:disabled {{ background: #2c3a55; color: #7f8794; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 6px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: #5d626c; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 12px; padding: 10px 8px 8px 8px; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #555a66; border-radius: 3px; background: {RAISED}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QListWidget, QTableWidget {{ background: #1a1c20; border: 1px solid {BORDER}; border-radius: 6px; outline: none; }}
QListWidget::item {{ padding: 6px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: #24344f; color: white; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 5px 6px; }}
QTableWidget {{ gridline-color: {BORDER}; }}

QProgressBar {{ background: {RAISED}; border: none; border-radius: 6px; height: 16px; text-align: center; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

QSlider::groove:horizontal {{ height: 8px; background: {RAISED}; border-radius: 4px; }}
QSlider::sub-page:horizontal {{ background: #35507f; border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 6px; margin: -9px 0; background: {TEXT}; border-radius: 3px; }}
QSlider::handle:horizontal:disabled {{ background: #5d626c; }}

QFrame#timeline {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#timeline QLabel {{ background: transparent; color: {MUTED}; }}
QLabel#timecode {{ color: {TEXT}; font-weight: bold; }}
QLabel#hint {{ color: {MUTED}; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3a3e48; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #3a3e48; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, colour in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, "#1a1c20"), (QPalette.ColorRole.AlternateBase, PANEL),
                         (QPalette.ColorRole.Button, RAISED), (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Text, TEXT),
                         (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.ToolTipBase, RAISED), (QPalette.ColorRole.ToolTipText, TEXT),
                         (QPalette.ColorRole.Highlight, ACCENT), (QPalette.ColorRole.HighlightedText, "#ffffff"),
                         (QPalette.ColorRole.PlaceholderText, MUTED), (QPalette.ColorRole.Link, ACCENT)):
        pal.setColor(role, QColor(colour))
    app.setPalette(pal)
    font = QFont("Malgun Gothic")           # Windows' Korean UI font; Qt falls back when it is missing
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLE)
