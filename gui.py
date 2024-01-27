from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem
from PyQt6 import QtCore, uic
from PyQt6.QtGui import QMovie, QPixmap, QColor, QPainter, QPen, QBrush, QPainterPath

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector

import sys
import time
import videoProcessor as video
import audioProcessor as audio
import pyautogui as pg
import multiprocessing as mp
import logging
import datetime
import os
tracker = video.videoManager()
pg.FAILSAFE = False
debug = False

if not os.path.exists(f'logs\\{datetime.date.today()}'):
    os.makedirs(f'logs\\{datetime.date.today()}')
if debug:
    logging.basicConfig(filename=f'logs\\{datetime.date.today()}\\data.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
else:
    logging.basicConfig(filename=f'logs\\{datetime.date.today()}\\data.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info('Program Started')

class gestureControlSetup(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('GUI/gestureControlSetup.ui', self)
        self.updateScene()
        self.nextButton.clicked.connect(self.calibrate)
        self.toggleButton.clicked.connect(self.toggleButtonClicked)
        self.calibrateWindow = calibrateWindow()
    
    def updateScene(self):
        self.show()

    def calibrate(self):
        self.calibrateWindow.show()

    def toggleButtonClicked(self):
        logging.debug('Toggle Button Clicked')
        if self.toggleButton.isChecked():
            self.toggleButton.setText("OFF")
            tracker.videoProcessing = False
        else:
            self.toggleButton.setText("ON")
            tracker.videoProcessing = True
            tracker.processVideo()

class calibrateWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('GUI/calibrate.ui', self)
        self.nextButton.clicked.connect(self.nextButtonClicked)
        self.corner = 0
        screenSize = pg.size()
        self.setFixedWidth(screenSize[0])
        self.setFixedHeight(screenSize[1])
    def nextButtonClicked(self):
        if self.corner == 0:
            tracker.calibrateGaze(0)
            time.sleep(1)
            self.corner = 1
            self.label.setText("Please point your head at the TOP RIGHT of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            self.topLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        elif self.corner == 1:
            tracker.calibrateGaze(1)
            time.sleep(1)
            self.corner = 2
            self.label.setText("Please point your head at the BOTTOM LEFT of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            
        elif self.corner == 2:
            tracker.calibrateGaze(2)
            time.sleep(1)
            self.corner = 3
            self.label.setText("Please point your head at the BOTTOM RIGHT of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            self.nextButton.setText("Finish")
        elif self.corner == 3:
            tracker.calibrateGaze(3)
            time.sleep(1)
            tracker.calibrateTracker()
            self.corner = 0
            self.label.setText("Please point your head at the TOP LEFT of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            self.nextButton.setText("Next")
            self.hide()


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongGame(Widget):
    pass


class PongApp(App):
    def build(self):
        return PongGame()


if __name__ == '__main__':
    PongApp().run()



# if __name__ == '__main__':
#     filePath = mp.Queue()
#     audioText = mp.Array('c', 100)
#     audioToggle = mp.Value('i', 0)
#     audioProcess = mp.Process(target=audio.rollingAudio, args=(filePath, audioToggle,))
#     transcribeProcess = mp.Process(target=audio.transcribeAudio, args=(filePath, audioText,))
#     # audioProcess.start()
#     # time.sleep(5)
#     # transcribeProcess.start()
#     # while True:
#     #     print(audioText.value.decode('utf-8'))
#     #     time.sleep(1)
#     app = QApplication(sys.argv)
#     mainWindow = gestureControlSetup()
#     sys.exit(app.exec())