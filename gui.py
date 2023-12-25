from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem
from PyQt6 import QtCore, uic
from PyQt6.QtGui import QMovie, QPixmap, QColor, QPainter, QPen, QBrush, QPainterPath
import sys
import time
import videoProcessor as video
import audioProcessor as audio
import pyautogui as pg
import multiprocessing as mp
tracker = video.videoManager()
pg.FAILSAFE = False

class gestureControlSetup(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('GUI/gestureControlSetup.ui', self)
        self.updateScene()
        self.nextButton.clicked.connect(self.calibrate)
        self.toggleButton.clicked.connect(self.toggleButtonClicked)
        self.calibrateWindow = calibrateWindow()
    
    def updateScene(self):
        print("updateScene")
        self.show()

    def calibrate(self):
        print("nextButtonClicked")
        self.calibrateWindow.show()

    def toggleButtonClicked(self):
        print("toggleButtonClicked")
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
            self.label.setText("Please point your head at the top right of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
        elif self.corner == 1:
            tracker.calibrateGaze(1)
            time.sleep(1)
            self.corner = 2
            self.label.setText("Please point your head at the bottom left of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
        elif self.corner == 2:
            tracker.calibrateGaze(2)
            time.sleep(1)
            self.corner = 3
            self.label.setText("Please point your head at the bottom right of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            self.nextButton.setText("Finish")
        elif self.corner == 3:
            tracker.calibrateGaze(3)
            time.sleep(1)
            tracker.calibrateTracker()
            self.corner = 0
            self.label.setText("Please point your head at the top left of your screen then click the button below. Be sure to turn your whole head and not just your eyes.")
            self.nextButton.setText("Next")
            self.hide()

if __name__ == '__main__':
    filePath = mp.Queue()
    audioText = mp.Array('c', 100)
    audioToggle = mp.Value('i', 0)
    audioProcess = mp.Process(target=audio.rollingAudio, args=(filePath, audioToggle,))
    transcribeProcess = mp.Process(target=audio.transcribeAudio, args=(filePath, audioText,))
    # audioProcess.start()
    # time.sleep(5)
    # transcribeProcess.start()
    # while True:
    #     print(audioText.value.decode('utf-8'))
    #     time.sleep(1)
    app = QApplication(sys.argv)
    mainWindow = gestureControlSetup()
    sys.exit(app.exec())