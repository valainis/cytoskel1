#!/usr/bin/python3


import sys
import copy

import matplotlib
matplotlib.use('qt5agg')

import os.path
from functools import partial

from collections import OrderedDict

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
import PyQt5.QtCore as QtCore

from PyQt5.QtWidgets import *

from PyQt5.QtGui import QKeySequence,QFont, QColor,QDoubleValidator


import PyQt5.QtWidgets as QtWidgets

import numpy as np
from numpy.linalg import norm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import cytoskel1 as csk1

from cytoskel1.subway2 import subway_canvas3
from cytoskel1.qlist import *

#from qutil import *
from cytoskel1.qutil import *

import pandas as pd


class examine_bdata(QMainWindow):
    def __init__(self,parent=None):
        #parent should be mwin?
        #super(QDialog,self).__init__(parent)
        super().__init__()

        self.mwin = parent
        mwin = self.mwin
        mwin.subway_names = []
        #self.bd = self.mwin.bd
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose) #???


        hbox = QHBoxLayout()
        self.vwidget = QWidget()
        vbox = QVBoxLayout()

        self.mode = "Base"
        self.modes = ["Base","Select Segments","Cell Info","Set Start","Split"]      
        sc = subway_canvas3(self, mwin,width=14, height=7.5, dpi=100)

        #sc.setFocusPolicy( QtCore.Qt.ClickFocus )
        #ok strong focus works, the above had some trouble
        sc.setFocusPolicy( QtCore.Qt.StrongFocus )
        sc.setFocus()
        
        vbox.addWidget(sc)
        #vbox.addWidget(ntb)
        self.vwidget.setLayout(vbox)        


        hbox.addWidget(self.vwidget)

        self.subway_canvas = sc

        self.central = QWidget()
        self.central.setLayout(hbox)
        self.setCentralWidget(self.central)

        self.createActions()
        self.createMenus()       

    def save_figure(self):
        #this generates worning on catalina, fix later
        print("saving figure")

        fname,ext = QFileDialog.getSaveFileName(self, 'Save Fig', self.mwin.dir_name, 'png')

        self.subway_canvas.fig.savefig(fname)

    def do_save_adj(self):

        print("saving adj")

        sc = self.subway_canvas

        print("start",self.mwin.csk.cg.start)

        self.mwin.csk.dump_cg()

    def apply(self):

        ch = []
        for i,box in enumerate(self.check_boxes):
            if box.isChecked():
                ch.append(self.check_markers[i])

        if len(ch) == 0:
            self.mwin.txt.setPlainText("No markers chosen")
            return

        self.mwin.subway_names = ch
        self.subway_canvas.compute_initial_figure()

        #this seems to be essential, draw inside of compute_initial_figure does not work
        self.subway_canvas.draw()


    def __call__(self,event):
        print("examine event",event)

        super(examine_bdata,self).__call__(event)

    def create_check_boxes(self):
        csk = self.mwin.csk
        check_markers = csk.markers
        print(check_markers)

        self.check_markers = check_markers

        gbox = QGroupBox()

        glay = QGridLayout()

        check_boxes = []
        
        for i,m in enumerate(check_markers):
            cb = QCheckBox(m)
            check_boxes.append(cb)
            glay.addWidget(cb,i//2,i%2)

            cb.setStyleSheet("color : red ; font : bold 12px")

        gbox.setLayout(glay)

        self.check_boxes = check_boxes

        return gbox



    def createActions(self):
        self.qlist_action = QAction("&Qlist..", self,
                                        statusTip="Qlist", triggered=self.do_qlist)

        self.save_figure_action = QAction("&Save Fig..", self,
                                        statusTip="", triggered=self.save_figure)


        self.save_adj_action = QAction("&Save Adj..", self,
                                        statusTip="", triggered=self.do_save_adj)

        self.reset_adj_action = QAction("&Reset adj..", self,
                                        statusTip="", triggered=self.do_reset_adj)

        self.set_mode_action =  QAction("&Set Mode..", self,
                                        statusTip="", triggered=self.do_set_mode)
        
    def createMenus(self):
        self.menuBar().setNativeMenuBar(False)
        
        self.actionMenu = self.menuBar().addMenu("&Actions")
        self.actionMenu.addAction(self.qlist_action)

        self.actionMenu.addAction(self.set_mode_action)        


        self.fileMenu = self.menuBar().addMenu("&Files")        
        self.fileMenu.addAction(self.save_figure_action)
        self.fileMenu.addAction(self.reset_adj_action)
        self.fileMenu.addAction(self.save_adj_action)        

    def do_set_mode(self):

        dlg = MRad2(self,self.modes)
        dlg.setWindowTitle("Mode")
        dlg.exec_()

        print(self.mode)


    def do_reset_adj(self):
        #make a new subway canvas
        print("reset")
        #need to copy to keep cg_save from being changed
        new_cg = copy.deepcopy(self.mwin.cg_save)
        self.mwin.csk.cg = new_cg
        self.subway_canvas.cg0 = new_cg

        self.subway_canvas.compute_initial_figure()        
        
    def do_qlist(self):
        qlis = Dialog_01(self)

        if hasattr(self.mwin,"subway_names"):
            print("subway names",self.mwin.subway_names)

        qlis.resize(300,500)
        qlis.setModal(False)
        #qlis.exec_() #for modal
        qlis.show()
    


class MWin(QMainWindow):
    def __init__(self,argv):
        super(MWin, self).__init__()
        font = QFont()


        font.setFamily("Courier")
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(16)

        txt = QPlainTextEdit()
        txt.setFont(font)
        self.setCentralWidget(txt)
        #txt.setPlainText("hello central")
        self.txt = txt

        self.createActions()
        self.createMenus()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.current_setup = None
        self.current_fcs_data = None

        self.added = False

        if len(argv) > 1:
            self.open_bdata(argv[1])
        

    def createActions(self):
        self.open_bdata_action = QAction("&Open Branch Data...", self,triggered=self.open_bdata)



        self.add_data_action = QAction("Add Data", self,triggered=self.add_data)

        
        self.bdata_action1 = QAction("&Subway..", self,
                               statusTip="", triggered=self.do_subway)

        
    def createMenus(self):
        self.menuBar().setNativeMenuBar(False)
        
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.open_bdata_action)

        #this may be useful later but does nothing now        
        #self.fileMenu.addAction(self.add_data_action)        

        
        self.actionMenu = self.menuBar().addMenu("&Actions")
        self.actionMenu.addAction(self.bdata_action1)

    def add_data(self):
        print("data added")

        file,_ = QFileDialog.getOpenFileName(self,filter="*.csv")

        df_add = pd.read_csv(file)

        print(df_add.shape)



    def do_qlist(self):
        #not needed currently
        qlis = Dialog_01(self)

        qlis.resize(300,500)        
        qlis.exec_()


    def do_subway(self):
        if not hasattr(self,"csk"):
            print("no cytoskel object")
            return

        eb = examine_bdata(self)
        #doing this lets subway stay up with global pca
        self.eb = eb
        eb.resize(1600,700)
        self.eb.csk = self.csk
        eb.show()

        

    def open_bdata(self,file=None):

        if not file:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        #check if valid

        self.dir_name = file

        if not os.path.isfile(file+"/br.adj"):
            s = "Invalid directory:" + file
            self.txt.setPlainText(s)
            return
        else:
            s = "Reading data from: " + file
            self.txt.setPlainText(s)

        self.csk = csk1.cytoskel(file)
        self.csk.open()
        #for adding data like inverse density
        self.csk.df_avg2 = self.csk.df_avg.copy()

        self.add_density()

        pcells = list(self.csk.br_adj.keys())

        #should just use this in subway
        self.csk.df_pcells = self.csk.df_avg2.loc[pcells,:]

        #save cg for reset
        self.cg_save = copy.deepcopy(self.csk.cg)
        self.pca_branches = []

    def show_cell_info0(self,icell):
        s = "Cell: %d : " % icell
        s2 = str(self.csk.br_adj[icell])
        self.txt.setPlainText(s+s2)


    def show_cell_info(self,cell):
        info = ["cell",str(cell)]

        df_info = self.csk.df_avg2

        markers = df_info.columns        
        #marker name lengths
        mlens = [len(x) for x in markers]
        mw = "%"+str(max(mlens)+1)+"s"
        fw = "%9.1f"
        iw = "%9d"
        sw = "%9s"

        info.append("\n")

        shead = mw % "marker" + " " + sw % "value"

        info.append(shead)
        cell_data = df_info.loc[cell,:].values
        
        for i,m in enumerate(markers):
            info.append(mw % m + " " + "%9.3f" % cell_data[i])

        qstr = "\n".join(info)

        self.txt.setPlainText(qstr)
        

    def add_density(self):
        if hasattr(self.csk,"csr_mst"):
            csr_dist = self.csk.csr_mst
        else:
            return
        dist = csr_dist.mean(axis=1)
        N = self.csk.csr_mst.shape[1]
        dist = np.array(dist)*N

        zzz = dist == 0.0

        nnz0 = np.array(self.csk.csr_mst.getnnz(axis=1))
        print(dist.shape,nnz0.shape)

        #need to do it like this, otherwise numpy confusion?
        dist /= nnz0[:,None]

        #np.savetxt("dist.txt",dist)
        self.csk.df_avg2.loc[:,'inv density'] = dist



if __name__ == '__main__':



    app = QApplication(sys.argv)
    mwin = MWin(sys.argv)
    mwin.show()
    mwin.setGeometry(50,50,350,600);    
    sys.exit(app.exec_())



