import PySide6.QtGui
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt
from PySide6 import QtGui
import sys

class TextLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint,True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint,True)
        self.setGeometry(300,200,500,200)
        self.setStyleSheet("color: rgb(255, 0, 0);")
        self.setFont(QtGui.QFont("Microsoft YaHei", 24))
        self.setText("沥青路面，模式：城市工况")
        self.mode=0
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.oldPos is not None:
            delta = event.globalPos() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPos()
    
    def keyPressEvent(self, ev) -> None:
        if self.mode ==0:
            self.mode=1
            self.setText("水泥路面，模式：城市工况")
        else:
            self.mode=0
            self.setText("沥青路面，模式：城市工况")
        
        return super().keyPressEvent(ev)

    def mouseReleaseEvent(self, event):
        self.oldPos = None
   

app=QApplication()
text_label=TextLabel()
text_label.show()

sys.exit(app.exec())
