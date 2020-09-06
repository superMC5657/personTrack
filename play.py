# -*- coding: utf-8 -*-
# !@time: 2020/9/6 下午6:10
# !@author: superMC @email: 18758266469@163.com
# !@fileName: play.py
import os

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from gui_utils.gui import Gui
from gui_utils.guiEvent import GuiEvent

# opencv 与 pyqt5 冲突 所以在调用pyqt5 作为前端时 使用pyqt5的plugins
os.environ[
    "QT_QPA_PLATFORM_PLUGIN_PATH"] = '/home/supermc/anaconda3/envs/personTrack/lib/python3.7/site-packages/PyQt5/Qt/plugins'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = Gui()

    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)

    display = GuiEvent(ui, mainWnd)

    mainWnd.show()

    sys.exit(app.exec_())
