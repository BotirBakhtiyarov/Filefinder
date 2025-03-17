import sys
import os
import subprocess
from assets.ui.main_app import MainApp
import pystray
from PIL import Image
import threading

def create_tray_icon(app):
    icon_path = os.path.join(os.path.dirname(__file__), "myicon.ico")
    icon_image = Image.open(icon_path)
    menu = (
        pystray.MenuItem("Open", lambda: app.show_window()),
        pystray.MenuItem("Quit", lambda: app.quit_app())
    )
    icon = pystray.Icon("SmartFileFinder", icon_image, "Smart File Finder", menu)
    return icon

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = MainApp()
    tray_icon = create_tray_icon(app)
    threading.Thread(target=tray_icon.run, daemon=True).start()
    app.mainloop()