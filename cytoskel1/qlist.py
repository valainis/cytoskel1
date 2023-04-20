import sys, os
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from  PyQt5.QtCore import Qt

class ListWidget1(QListWidget):

    msig = pyqtSignal(str)

    def __init__(self, type, parent=None):
        super().__init__()
        self.setIconSize(QtCore.QSize(124, 124))
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):

        row = self.currentRow()
        if row >= 0:
            item = self.item(row)
            self.msig.emit(item.text())
        super(ListWidget1, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super(ListWidget1, self).dragMoveEvent(event)

    def dropEvent(self, event):
        #ok this seems easiest way to kepp this list from changing
        #and it has to be here
        return


class ListWidget2(QListWidget):

    msig2 = pyqtSignal()

    def __init__(self, type, src, parent=None):
        super(ListWidget2, self).__init__(parent)
        self.src = src
        self.setIconSize(QtCore.QSize(124, 124))
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)

        self.src.msig.connect(self.set_incoming)
        self.msig2.connect(self.set_internal)
        self.incoming = False
        self.internal = False

    def set_incoming(self,itext):
        #what does this do?
        self.itext = itext
        self.incoming = True

    def set_internal(self):
        if not self.incoming:
            self.internal = True

    def dragEnterEvent(self, event):
        if not self.incoming:
            self.msig2.emit()
        else:
            #print("drag incoming")
            pass
        super(ListWidget2, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super(ListWidget2, self).dragMoveEvent(event)

    def dropEvent(self, event):

        items = []
        for index in range(self.count()):
            items.append(self.item(index).text())

        if self.incoming:
            if self.itext in items:
                #print("already there",self.text)
                pass
            else:
                event.setDropAction(QtCore.Qt.MoveAction)
                super(ListWidget2, self).dropEvent(event)                    
            self.incoming = False                
        else:
            event.setDropAction(QtCore.Qt.MoveAction)
            super(ListWidget2, self).dropEvent(event)
            self.incoming = False
            self.internal = False

        #ok so item does not get added until here
        items = []
        for index in range(self.count()):
            items.append(self.item(index).text())


    def mousePressEvent(self, event):
        self._mouse_button = event.button()
        super(ListWidget2, self).mousePressEvent(event)

        modP = QApplication.keyboardModifiers()        
        
        #this is how to delete list items
        if (event.buttons() == Qt.RightButton) and ((modP & Qt.ShiftModifier) == Qt.ShiftModifier):
            row = self.currentRow()
            self.takeItem(row)


class Dialog_01(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)

        if hasattr(self.parent().mwin,'subway_names'):
            print("qlist subway names",self.parent().mwin.subway_names)
        
        self.listItems={}

        self.csk = self.parent().csk

        glay = QGridLayout()


        myBoxLayout = QHBoxLayout()
        #self.setLayout(myBoxLayout)
        self.setLayout(glay)

        self.items0 = []

        self.listWidgetA = ListWidget1(self)

        self.lwA = self.listWidgetA

        #qitem = QListWidgetItem("curvature", self.lwA)
        #self.items0.append(qitem)

        #for m in self.csk.markers:
        for m in self.csk.df_avg2.columns:
            qitem = QListWidgetItem( m, self.listWidgetA )
            self.items0.append(qitem)
        #myBoxLayout.addWidget(self.listWidgetA)

        glay.addWidget(self.listWidgetA,0,0)

        self.listWidgetB = ListWidget2(self,self.listWidgetA)
        myBoxLayout.addWidget(self.listWidgetB)

        glay.addWidget(self.listWidgetB,0,1)

        self.listWidgetB.currentItemChanged.connect(self.item_clicked)        
        self.listWidgetB.itemDoubleClicked.connect(self.idoub)

        snames = list(self.parent().mwin.csk.df_avg2.columns)

        for m in self.parent().mwin.subway_names[::-1]:
            qitem = QListWidgetItem( m, self.listWidgetB )
            self.items0.append(qitem)
        

        self.listWidgetA.iden = "A"
        self.listWidgetB.iden = "B"
        self.listWidgetB.text = ""

        setButton = QPushButton("Apply")
        setButton.clicked.connect(self.apply)
        glay.addWidget(setButton,1,0,1,2)
        

        self.listWidgetA.partner = self.listWidgetB

    def apply(self):
        print("applied")
        n_items = self.listWidgetB.count()

        lwb = self.listWidgetB        

        items = []
        for index in range(n_items):
            items.append(lwb.item(index).text())

        if len(items) == 0:
            self.parent().mwin.txt.setPlainText("subway empty list")
            return

        eb = self.parent()

        eb.mwin.subway_names = items[::-1]

        print("qlist apply",eb.csk.cg.start)
        eb.subway_canvas.compute_initial_figure()        
        

    def idoub(self,value):
        print(value)
        row = self.listWidgetB.currentRow()
        self.listWidgetB.takeItem(row)

    def items_dropped(self, arg):
        print('items_dropped', arg)
        pass

    def item_clicked(self, arg):
        print("clicked",arg)


class Dialog_02(QDialog):
    def __init__(self,parent=None,name_list=[],apply_func=None):
        super().__init__(parent)

        self.name_list = name_list

        self.apply_func = apply_func
        
        self.listItems={}

        self.csk = self.parent().csk

        glay = QGridLayout()


        myBoxLayout = QHBoxLayout()
        #self.setLayout(myBoxLayout)
        self.setLayout(glay)

        self.items0 = []

        self.listWidgetA = ListWidget1(self)

        self.lwA = self.listWidgetA

        #qitem = QListWidgetItem("curvature", self.lwA)
        #self.items0.append(qitem)

        #for m in self.csk.markers:
        for m in self.csk.df_avg2.columns:
            qitem = QListWidgetItem( m, self.listWidgetA )
            self.items0.append(qitem)
        #myBoxLayout.addWidget(self.listWidgetA)

        glay.addWidget(self.listWidgetA,0,0)

        self.listWidgetB = ListWidget2(self,self.listWidgetA)
        myBoxLayout.addWidget(self.listWidgetB)

        glay.addWidget(self.listWidgetB,0,1)

        self.listWidgetB.currentItemChanged.connect(self.item_clicked)        
        self.listWidgetB.itemDoubleClicked.connect(self.idoub)

        snames = list(self.parent().mwin.csk.df_avg2.columns)

        for m in self.name_list[::-1]:
            qitem = QListWidgetItem( m, self.listWidgetB )
            self.items0.append(qitem)
        

        self.listWidgetA.iden = "A"
        self.listWidgetB.iden = "B"
        self.listWidgetB.text = ""

        setButton = QPushButton("Draw")
        setButton.clicked.connect(self.draw0)
        glay.addWidget(setButton,1,0,1,2)
        

        self.listWidgetA.partner = self.listWidgetB

    def draw0(self):
        print("applied")
        n_items = self.listWidgetB.count()

        lwb = self.listWidgetB        

        items = []
        for index in range(n_items):
            items.append(lwb.item(index).text())

        if len(items) == 0:
            self.parent().mwin.txt.setPlainText("subway empty list")
            return

        eb = self.parent()

        self.name_list.clear()
        self.name_list.extend(items[::-1])

        print("qlist apply",eb.csk.cg.start)
        if self.apply_func is not None:
            #eb.subway_canvas.compute_initial_figure()
            self.apply_func()

    def idoub(self,value):
        print(value)
        row = self.listWidgetB.currentRow()
        self.listWidgetB.takeItem(row)

    def items_dropped(self, arg):
        print('items_dropped', arg)
        pass

    def item_clicked(self, arg):
        print("clicked",arg)




                

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog_1 = Dialog_01()
    dialog_1.show()
    dialog_1.resize(300,360)
    sys.exit(app.exec_())
