#!/usr/bin/env python3
import os

import vtk
import numpy as np
import time

import ast
from sortedcontainers import SortedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore, QtGui, uic, QtWidgets

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt

from PyQt5.QtWidgets import *
"""
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QFrame,
        QPlainTextEdit, QLabel, QMainWindow, QMenu, QMessageBox, QSizePolicy,
        QVBoxLayout, QWidget, QHBoxLayout, QOpenGLWidget, QSlider,QDialog,
        QFileDialog,QGridLayout,QGroupBox,QCheckBox,QPushButton,QLineEdit,QComboBox,
        QRadioButton,QDockWidget)
"""

from PyQt5.QtGui import QKeySequence,QFont, QColor,QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
import time
import pandas as pd

import numpy as np
import numpy.linalg as la

import cytoskel1 as csk1

from vtk.util import numpy_support as ns
from sklearn.decomposition import NMF




class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    cell_picked = pyqtSignal(int)

    def __init__(self, vpts,pobj):
        self.r0 = vpts.r0
        self.glyphs = vpts.glyph
        #self.renderer = renderer
        self.renderer = pobj.ren
        #self.AddObserver("LeftButtonPressEvent", self._left_button_press_event)
        self.AddObserver("RightButtonPressEvent", self._right_button_press_event)
        self.map0 = vpts.map0

        self.actor = None
        self.pobj = pobj

    #def _left_button_press_event(self, obj, event):
    def _right_button_press_event(self, obj, event):

        win = self.pobj.vtkWidget.GetRenderWindow()

        #this gives window coords, lower left is 0,0, x is left_right
        click_pos = self.GetInteractor().GetEventPosition()

        print("click type",type(click_pos))

        #scale = self.renderer.getDevicePixelRatioCompensation()
        #scale = 2
        #click_pos = [ p*scale for p in click_pos ]


        if self.actor != None:
            self.renderer.RemoveActor(self.actor)
            self.actor = None
            print("removed")
            self.OnRightButtonDown()            
            return


        size0 = win.GetSize()
        print("pos",click_pos,size0)

        #this is work around to solve problem with retina
        #which depends on vtk install
        # not clear if will fail on ubuntu or windows
        # user should be able to set something instead
        if (size0[0] < 1024) or (size0[1] < 1024):
            sfac = .5
        else:
            sfac = 1.0

        #it seems sfac = .5 is now always correct, is this a retina issue?
        sfac = .5

        print("sfac",sfac)
        """
        #picking can be slow with lots of points
        self.OnLeftButtonDown()
        return
        """

        cell_picker = vtk.vtkCellPicker()
        #cell_picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())
        #cell_picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        cell_picker.Pick(sfac*click_pos[0], sfac*click_pos[1], 0, self.renderer)

        input_ids = self.glyphs.GetOutput().GetPointData().GetArray("InputPointIds")

        if input_ids:
            vtk_cell = self.glyphs.GetOutput().GetCell(cell_picker.GetCellId())
            print("cell pick",vtk_cell)
            if vtk_cell and vtk_cell.GetNumberOfPoints() > 0:
                input_id = vtk_cell.GetPointId(0)
                selected_id = input_ids.GetTuple1(input_id)
                if selected_id >= 0:
                    highlight_sphere = vtk.vtkSphereSource() 
                    #highlight_sphere.SetRadius(1.16*self.r0)
                    print("highlight")
                    highlight_sphere.SetRadius(self.r0)
                    highlight_sphere.SetThetaResolution(8)
                    highlight_sphere.SetPhiResolution(8)
                    #highlight_sphere.SetCenter(self.glyphs.GetOutput().GetPoint(int(selected_id)))
                    print("selected",selected_id)

                    sid = int(selected_id)

                    #cytoskel cell
                    cell = self.map0[sid]

                    self.pobj.cell_info(cell)

                    #so sid is the cell # in br_adj - map0[sid] is full data cell number
                    print("pobj",self.pobj.vpts.data[sid])

                    #self.cell_picked.emit(cell)

                    xpnt = self.glyphs.GetInput().GetPoint(int(selected_id))

                    print("xpnt",xpnt) #this matches pobj.vpts.data[sid]

                    print("click_pos",click_pos)

                    highlight_sphere.SetCenter(self.glyphs.GetInput().GetPoint(int(selected_id)))                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(highlight_sphere.GetOutputPort())

                    highlight_actor = vtk.vtkActor()
                    highlight_actor.SetMapper(mapper)
                    highlight_actor.GetProperty().SetColor(1.0,0.0,1.0)
                    self.renderer.AddActor(highlight_actor)
                    self.actor = highlight_actor

                    print("Pick!")
        #self.OnLeftButtonDown()
        self.OnRightButtonDown()
        return


        

def mk_maps(df,pcells):

    map0 = np.array(df.loc[pcells,:].index)
    npcells = len(pcells)

    idx0 = np.arange(npcells)

    N = df.shape[0]

    rmap0 = np.full((N,),-1,dtype=int)

    rmap0[map0] = idx0


    return map0,rmap0


class MRad00(QDialog):
    def __init__(self,parent=None,markers=[],set_func=None,get_func=None):
        super().__init__(parent)
        p = self.parent()

        self.set_func = set_func
        self.get_func = get_func

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

        if self.get_func:
            m0 = self.get_func()
        else:
            m0 = None
        
        for i,m in enumerate(markers):
            cb = QRadioButton(m)
            if m == m0:
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

        self.set_func(m)
        self.close()


class MRad(QDialog):
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
        p = self.parent()

        gbox = QGroupBox()

        glay = QGridLayout()

        radio_buttons = []

        ncol = len(markers)//25 + 1
        
        for i,m in enumerate(markers):
            cb = QRadioButton(m)
            if m == p.mcolor:
                cb.setChecked(True)
            radio_buttons.append(cb)
            glay.addWidget(cb,i//ncol,i%ncol)

        gbox.setLayout(glay)

        self.radio_buttons = radio_buttons
        return gbox
        
    def apply(self):
        #check the buttons for cname
        p = self.parent()
        m = None
        for i,box in enumerate(self.radio_buttons):
            if box.isChecked():
                m = self.markers[i]

        self.parent().m = m
        self.parent().mcolor = m
        self.close()


class MCheck(QDialog):
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

        p = self.parent()

        gbox = QGroupBox()

        glay = QGridLayout()

        radio_buttons = []

        for i,m in enumerate(markers):
            cb = QCheckBox(m)
            radio_buttons.append(cb)
            glay.addWidget(cb,i,0)

            if m in p.mlist:
                cb.setChecked(True)
            else:
                cb.setChecked(False)

        gbox.setLayout(glay)

        self.radio_buttons = radio_buttons
        return gbox
        
    def apply(self):

        p = self.parent()

        mlist = []
        for i,box in enumerate(self.radio_buttons):
            if box.isChecked():
                m = self.markers[i]
                mlist.append(m)

        p.mlist = mlist
        self.close()

        p.do_color(mode_changed=True)



class MSlide(QDialog):
    def __init__(self,parent=None,markers=[]):
        super().__init__(parent)
        self.mwin = self.parent()

        self.markers = markers

        lay = QVBoxLayout()

        self.l1 = QLabel("Hello")
        self.l1.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.l1)
        

        self.sl = QSlider(Qt.Horizontal)

        self.sl.setMinimum(10)
        self.sl.setMaximum(30)
        self.sl.setValue(20)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(5)

        lay.addWidget(self.sl)
        self.sl.valueChanged.connect(self.valuechange)        


        setButton = QPushButton("Apply")
        setButton.clicked.connect(self.apply)
        lay.addWidget(setButton)
        
        self.setLayout(lay)


    def valuechange(self):
      size = self.sl.value()
      self.l1.setFont(QFont("Arial",size))

      rad = size * .12 / 20

      mwin = self.parent()
      #mwin.do_radius(mwin.vpts,rad)
      do_radius2(mwin,mwin.vpts,rad)
        
    def apply(self):
        self.close()



class canvas0(FigureCanvas):
    def __init__(self,parent=None):
        fig = Figure()
        super().__init__(fig)

        ax = fig.add_subplot(111)

        #this does not work unless there is a subplot added
        #with add_subplot
        fig.subplots_adjust(bottom=0.05,right=.5)        
        
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.ax.axis('off')

    def cbar(self,vmin,vmax):
        self.ax.clear()
        cmap = mpl.cm.get_cmap('jet')
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)        
        self.cb = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=self.ax, orientation='vertical')


    def sizeHint(self):
        return QtCore.QSize(150,600)


    
def do_radius2(mwin,vpts,rad=.12):
    iren2 = mwin.vtkWidget.GetRenderWindow().GetInteractor()
    r0 = rad

    npnts = vpts.data.shape[0]        
    rscale = np.full(npnts,r0)

    srad = ns.numpy_to_vtk(num_array=rscale, deep=True, array_type=vtk.VTK_FLOAT)
    srad.SetName('scales')        

    vpts.p_data.GetPointData().RemoveArray('scales')

    vpts.p_data.GetPointData().AddArray(srad)
    vpts.p_data.GetPointData().SetActiveScalars("scales") #radius first        

    mwin.vtkWidget.GetRenderWindow().Render()





def do_color2(mwin,mode_changed=False):

    do_vpts2 = mwin.do_vpts2
    
    iren2 = mwin.vtkWidget.GetRenderWindow().GetInteractor()

    if not hasattr(mwin,"vpts"):
        print("no data")
        return

    map0 = mwin.vpts.map0
    dfm = mwin.df_info.loc[map0,:]

    print("do_color2")


    if not mode_changed:
        dlg = MRad(mwin,list(dfm.columns))
        dlg.setWindowTitle("marker")
        dlg.move(50,50)
        dlg.exec_()


    m = mwin.m
    colors = dfm.loc[:,m].values



    cmap = mpl.cm.get_cmap('jet')

    vmin = np.amin(colors)
    vmax = np.amax(colors)

    mwin.c0.cbar(vmin,vmax)
    mwin.c0.cb.set_label(m)
    mwin.c0.draw()


    #colors = colors/np.amax(colors)

    #fixed as in vpara
    colors = (colors - vmin)/(vmax-vmin)    

    colors8 = cmap(colors,bytes=True)

    scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    #this works
    mwin.vpts.p_data.GetPointData().RemoveArray('col8')

    scol8.SetName('col8')
    mwin.vpts.p_data.GetPointData().AddArray(scol8)


    if do_vpts2:
        df2 = mwin.df_skip
        colors2 = df2.loc[:,m].values
        colors2 = colors2/vmax
        colors8_2 = cmap(colors2,bytes=True)

        colors8_2 = colors8_2

        #this must be the alpha ?
        #colors8_2[:,3] = 32
        #colors8_2[:,3] = 16
        colors8_2[:,3] = 8
        
        scol8_2 = ns.numpy_to_vtk(num_array=colors8_2, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        mwin.vpts2.p_data.GetPointData().RemoveArray('col8')
        scol8_2.SetName('col8')
        mwin.vpts2.p_data.GetPointData().AddArray(scol8_2)
        mwin.ren.AddActor(mwin.vpts2.actor)
        mwin.vpts2_actor = True
        print("added scol8_2")

    else:
        if mwin.vpts2_actor:
            mwin.ren.RemoveActor(mwin.vpts2.actor)
            mwin.vpts2_actor = False
        
    #ok this works, can add and remove actors at will
    #test
    #mwin.ren.RemoveActor(mwin.vpts2.actor)

    mwin.vtkWidget.GetRenderWindow().Render()
    


class vpoints:
    def __init__(self,data,rscale):
        self.data = data
        sphere = vtk.vtkSphereSource()
        ndiv = 8
        sphere.SetThetaResolution(ndiv)
        sphere.SetPhiResolution(ndiv)
        #this sets the original radius of
        #the sphere for the glyph
        sphere.SetRadius(.5)

        points = vtk.vtkPoints()
        
        p_data = vtk.vtkPolyData() #note polydata
        
        self.p_data = p_data

        npnts = data.shape[0]
        points.SetNumberOfPoints(npnts)


        data2 = ns.numpy_to_vtk(num_array=data, deep=True, array_type=vtk.VTK_FLOAT)

        points.SetData(data2)

        #this will be used to scale the glyph sphere radii
        #could vary on sphere by sphere basis
        #rad = np.full(npnts,1.0) #note .2        
        srad = ns.numpy_to_vtk(num_array=rscale, deep=True, array_type=vtk.VTK_FLOAT)
        srad.SetName('scales')
        
        p_data.SetPoints(points)
        p_data.GetPointData().AddArray(srad)
        p_data.GetPointData().SetActiveScalars("scales") #radius first

        glyph = vtk.vtkGlyph3D()
        glyph.GeneratePointIdsOn()
        glyph.SetInputData(p_data)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        
        mapper.SetScalarModeToUsePointFieldData() # without, color are displayed regarding radius and not color label
        mapper.SelectColorArray("col8") # !!!to set color (nevertheless you will have nothing)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.actor = actor
        self.glyph = glyph
        self.points = points
        self.p_data = p_data

    def add_lines(self,br_adj,rmap0,r0):
        points = self.points
        lines = vtk.vtkCellArray()

        line_id = 0
        for v in br_adj:
            iv = rmap0[v]
            row = br_adj[v]
            for v2 in row:
                iv2 = rmap0[v2]
                line0 = vtk.vtkLine()
                line0.GetPointIds().SetId(0,iv)
                line0.GetPointIds().SetId(1,iv2)
                lines.InsertNextCell(line0)
                
        # Create a polydata to store everything in
        linesPolyData = vtk.vtkPolyData()

        # Add the points to the dataset
        linesPolyData.SetPoints(points)

        # Add the lines to the dataset
        linesPolyData.SetLines(lines)

        tubes = vtk.vtkTubeFilter()
        tubes.SetInputData(linesPolyData)
        #tubes.CappingOn()
        tubes.SidesShareVerticesOff()
        tubes.SetNumberOfSides(16)
        tubes.SetRadius(.15*r0)
        tubes.Update()

        # Visualize
        line_mapper = vtk.vtkPolyDataMapper()
        #line_mapper.SetInputData(linesPolyData)
        line_mapper.SetInputData(tubes.GetOutput())        

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(1,1,1)
        self.line_actor = line_actor


class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)

        self.frame = QFrame()
        self.vl = QHBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)



        #set up text widget
        font = QFont()

        #font.setFamily("Courier")
        font.setFamily("Arial")
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(16)

        txt = QPlainTextEdit()
        txt.setFont(font)
        txt.setMaximumWidth(400)        
        self.txt = txt
        #color: is font color
        txt.setStyleSheet("QPlainTextEdit {background-color: #c0c0c0; color: #000000;}")
        c0 = canvas0()        
        self.c0 = c0
        
        self.vl.addWidget(txt)
        self.vl.addWidget(self.vtkWidget)
        self.vl.addWidget(c0)        


        self.c0.draw()

        #vtk window setup
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.ren.SetBackground(.8,.8,.8)

        self.renderer = self.ren
        self.frame.setLayout(self.vl)
        
        self.setCentralWidget(self.frame)

        self.iren.Start()

        #self.statusBar().showMessage('vtk editor')
        self.createActions()
        self.createMenus()        

        self.vtkWidget.GetRenderWindow().Render()

        self.mlist = []
        self.mcolor = ""
        
        #this is necessary
        self.show()


    def createMenus(self):
        self.menuBar().setNativeMenuBar(False)
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.open_project_action)
        
        self.actionMenu = self.menuBar().addMenu("&Action")
        self.actionMenu.addAction(self.color_action)
        self.actionMenu.addAction(self.slide_action)
        self.actionMenu.addAction(self.camera_info_action)
        self.actionMenu.addAction(self.get_view_action)                
        

        self.modeMenu = self.menuBar().addMenu("&Mode")
        self.modeMenu.addAction(self.mode1_action)
        
    def createActions(self):
        self.open_project_action = QAction("&Open Project...", self,triggered=self.open_project)
        
        self.color_action = QAction("&Color", self, triggered=self.do_color)
        self.slide_action = QAction("Size", self, triggered=self.do_slide)

        self.mode1_action = QAction("View", self, triggered=self.do_mode1)
        self.camera_info_action = QAction("Camera", self, triggered=self.camera_info)
        self.get_view_action = QAction("Get View", self, triggered=self.get_view)

    def set_view0(self,m):
        self.view = m

    def get_view0(self):
        if hasattr(self,"mode"):
            m0 = self.mode
        else:
            self.mode = None
            m0 = None
        return m0
        

    def get_view(self):
        vdir = self.csk.pca_views        
        print("hello",vdir)
        vlist = os.listdir(vdir)
        print(vlist)
        
        dlg = MRad00(self,vlist,self.set_view0)
        dlg.setWindowTitle("Mode")
        dlg.exec_()

        vfile = self.csk.pca_views + self.view

        df_cam = pd.read_csv(vfile)

        df_cam.set_index(df_cam.columns[0],inplace=True)
        V = df_cam.values

        print(df_cam)

        cam = self.ren.GetActiveCamera()

        cam.SetPosition(df_cam.loc['pos',:])
        cam.SetViewUp(df_cam.loc['up',:])
        cam.SetFocalPoint(df_cam.loc['focal',:])

        self.ren.ResetCamera()        
        
    def camera_info(self):
        mwin = self

        cam = mwin.ren.GetActiveCamera()
        size = mwin.ren.GetRenderWindow().GetSize()
        #tp = mwin.ren.GetProjectionTransformMatrix()

        #tp = cam.GetProjectionTransformMatrix(mwin.ren)    

        n,f = cam.GetClippingRange()
        print("nf",n,f)    

        aspect = size[0]/size[1]

        P = cam.GetCompositeProjectionTransformMatrix( aspect, n, f )


        cam_pos = cam.GetPosition()
        cam_up = cam.GetViewUp()
        cam_focal = cam.GetFocalPoint()
        
        print("position",cam_pos)        
        print("viewup",cam_up)
        print("focal",cam_focal)

        head = ["pca0","pca1","pca2"]

        index = ["focal","pos","up"]

        data = np.array([cam_focal,cam_pos,cam_up])

        df = pd.DataFrame(data,columns=head,index=index)

        if hasattr(mwin,"csk"):

            pdir = mwin.csk.project_dir

            vdir = pdir + "pca_views/"

            if not os.path.exists(vdir):
                os.mkdir(vdir)                
            

            text, ok = QInputDialog.getText(self, 'pca view', 'view name:')

            if ok:
                df.to_csv(vdir+text)


    def camera_info0(self):
        #capture the initial camera state
        mwin = self

        cam = mwin.ren.GetActiveCamera()
        size = mwin.ren.GetRenderWindow().GetSize()

        n,f = cam.GetClippingRange()

        cam_pos = cam.GetPosition()
        cam_up = cam.GetViewUp()
        cam_focal = cam.GetFocalPoint()

        cam_angle = cam.GetViewAngle()
        
        print("position",cam_pos)        
        print("viewup",cam_up)
        print("focal",cam_focal)
        print("angle",cam_angle)

        head = ["pca0","pca1","pca2"]

        index = ["focal","pos","up"]

        data = np.array([cam_focal,cam_pos,cam_up])

        df = pd.DataFrame(data,columns=head,index=index)

        if hasattr(mwin,"csk"):

            pdir = mwin.csk.project_dir
            vdir = pdir + "pca_views/"

            if not os.path.exists(vdir):
                os.mkdir(vdir)                

            text = "initial_view"

            if not os.path.exists(vdir+text):
                df.to_csv(vdir+text)
                

    def do_mode1(self):
        mch = MCheck(self,['cloud'])

        #modifies self.mlist
        mch.exec_()

    def do_slide(self):
        dlg = MSlide(self)
        dlg.setWindowTitle("HELLO!")
        dlg.exec_()

    def do_color(self,mode_changed=False):
        if 'cloud' in self.mlist:
            self.do_vpts2 = True           
        else:
            self.do_vpts2 = False            
        do_color2(self,mode_changed)
        

    def do_color0(self):
        iren2 = self.vtkWidget.GetRenderWindow().GetInteractor()

        if not hasattr(self,"vpts"):
            print("no data")
            return

        map0 = self.vpts.map0
        dfm = self.df_info.loc[map0,:]

        dlg = MRad(self,list(dfm.columns))
        dlg.setWindowTitle("HELLO!")
        dlg.exec_()
        

        m = self.m
        colors = dfm.loc[:,m].values

        cmap = mpl.cm.get_cmap('jet')

        vmin = 0.0
        vmax = np.amax(colors)

        self.c0.cbar(vmin,vmax)
        self.c0.cb.set_label(m)
        self.c0.draw()

        colors = colors/np.amax(colors)

        colors8 = cmap(colors,bytes=True)

        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        #this works
        self.vpts.p_data.GetPointData().RemoveArray('col8')

        scol8.SetName('col8')
        self.vpts.p_data.GetPointData().AddArray(scol8)

        #ok this works, can add and remove actors at will
        #test
        self.ren.RemoveActor(self.vpts2.actor)

        self.vtkWidget.GetRenderWindow().Render()


    def cell_info(self,cell):
        info = ["cell",str(cell)]

        markers = self.df_info.columns        
        #marker name lengths
        mlens = [len(x) for x in markers]
        mw = "%"+str(max(mlens)+1)+"s"
        fw = "%9.1f"
        iw = "%9d"
        sw = "%9s"

        info.append("\n")

        shead = mw % "marker" + " " + sw % "value"

        info.append(shead)
        cell_data = self.df_info.loc[cell,:].values
        
        for i,m in enumerate(markers):
            cd = cell_data[i]
            if isinstance(cd,(int,float)):
                info.append(mw % m + " " + "%9.3f" % cell_data[i])
            else:
                info.append(mw % m + " : " + str(cd))

        qstr = "\n".join(info)

        self.txt.setPlainText(qstr)


    def open_project(self):
        pdir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if pdir == "" or pdir == None: return

        self.dir_name = pdir

        csk = csk1.cytoskel(pdir)
        #if not csk.open():
        if not csk.open2(): #make this the new version
            print("Invalid project")
            return
        else:
            self.csk = csk

        dset(csk,self)

        qstr = pdir
        self.txt.setPlainText(qstr)
        
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.ren.RemoveAllViewProps()
        self.ren.SetBackground(.8,.8,.8)


        
        self.use_data(self.vpts)

        self.vtkWidget.GetRenderWindow().Render()                

        #ok this solved the view scaling problem
        self.ren.ResetCamera()

        #
        self.camera_info0()

        #this solved the problem of not displaying
        #until window click
        self.vtkWidget.update()



    def use_data(self,vpts=None):
        self.vpts = vpts

        vpts2 = self.vpts2

        npnts2 = vpts2.data.shape[0]

        colors8 = np.full( (npnts2,4),255,dtype=np.uint8)
        colors8[:,3] = 32
        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)


        vpts2.p_data.GetPointData().RemoveArray('col8')
        scol8.SetName('col8')
        vpts2.p_data.GetPointData().AddArray(scol8)
        

        self.ren.AddActor(vpts.actor)


        self.ren.AddActor(vpts2.actor)

        self.vpts2_actor = True
        
        self.ren.AddActor(vpts.line_actor)
        style = InteractorStyle(vpts,self)

        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)
        
        #this is necessary
        self.iren.Start()


def write_x(x,fname):
    f = open(fname,"w")

    s = str(x)

    s = s.split('],')
    s = '],\n'.join(s)
    
    f.write(s)
    f.close()

def read_x(fname):
    f = open(fname,"r")
    x = f.read()
    f.close()

    x = ast.literal_eval(x)

    return x
        

def dset(csk,mwin):

    pcells = list(csk.br_adj.keys())
    map0,rmap0 = mk_maps(csk.df_avg,pcells)

    ncells = len(map0)

    tcols = csk.traj_markers
    
    tdata0 = csk.df_avg.loc[pcells,tcols].values
    #tdata0 = csk.df.loc[pcells,tcols].values #to see the original trajectories
    tdata = csk.df.loc[:,tcols].values    
    n = tdata0.shape

    print("do pca")

    #use the original data to get the pca transform
    #so that it is independent of MST etc
    pca = csk1.pca_coords(tdata,fix=True)
    ux,urad = pca.get_coords(tdata0-pca.mu)

    print("urad",urad)

    scale = 5.0/urad

    #data = ux[:,:3]
    #do scaling based on urad
    data = ux[:,:3]*scale
    

    hdata = ["pca00","pca01","pca02"]
    df_data = pd.DataFrame(data,columns=hdata,index=map0)

    br_adj = csk.br_adj

    r0 = .12

    npnts = data.shape[0]        
    rscale = np.full(npnts,r0)

    vpts = vpoints(data,rscale)
    points = vpts.points

    vpts.add_lines(br_adj,rmap0,r0)

    vpts.map0 = map0
    print("map0 set")    
    vpts.r0 = r0


    mwin.df_info = csk.df_avg
    mwin.vpts = vpts

    #add some more points

    skip = 1
    mwin.skip = skip
    tdata2 = csk.df.loc[::skip,tcols].values

    mwin.df_skip = csk.df.loc[::skip,:]

    """
    #try this
    #interesting result shows potential extra branches?
    tdata3 = csk.csr_avg @ tdata2

    for i in range(4):
        tdata3 = csk.csr_avg @ tdata3
    ux2,urad2 = pca.get_coords(tdata3-pca.mu)
    """
    ux2,urad2 = pca.get_coords(tdata2-pca.mu)


    print("ux2",ux2.shape)

    ux2 = ux2 * scale
    rscale2 = np.full(ux2.shape[0],.1)
    vpts2 = vpoints(ux2[:,:3],rscale2)

    mwin.vpts2 = vpts2
    


    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    #window.setGeometry(200,200,1000,600);
    window.setGeometry(50,100,1500,900)
    sys.exit(app.exec_())
