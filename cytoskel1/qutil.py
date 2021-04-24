from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class MRad2(QDialog):
    def __init__(self,parent=None,markers=[]):
        super().__init__(parent)
        p = self.parent()

        self.markers = markers

        lay = QVBoxLayout()

        gbox = self.create_buttons()

        lay.addWidget(gbox)

        setButton = QPushButton("Apply")
        setButton.clicked.connect(self.apply)
        lay.addWidget(setButton)
        
        self.setLayout(lay)    


    def create_buttons(self):
        markers = self.markers

        gbox = QGroupBox()

        glay = QGridLayout()

        radio_buttons = []
        
        for i,m in enumerate(markers):
            cb = QRadioButton(m)
            if self.parent().mode == m:
                cb.setChecked(True)
            radio_buttons.append(cb)
            glay.addWidget(cb,i,0)

        gbox.setLayout(glay)

        self.radio_buttons = radio_buttons
        return gbox
        
    def apply(self):
        #check the buttons for cname

        m = None
        for i,box in enumerate(self.radio_buttons):
            if box.isChecked():
                m = self.markers[i]

        self.parent().mode = m
        self.close()
