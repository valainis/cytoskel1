#!/usr/bin/env python3

import vtk
import numpy as np
import time

import os

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

from cytoskel1.gu import fgraph,critical0

from vtk.util import numpy_support as ns
from sklearn.decomposition import NMF

from sklearn.metrics import euclidean_distances
from sklearn import manifold


def get_sparsity(csr_dist):
    #csr_dist is sparse adjacency
    dist = csr_dist.sum(axis=1)
    N = csr_dist.shape[1]
    dist = np.array(dist)

    nnz0 = np.array(csr_dist.getnnz(axis=1))
    #need to do it like this, otherwise numpy confusion?
    dist /= nnz0[:,None]

    #dist is measure of inverse density
    return dist.flatten(),nnz0



class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    cell_picked = pyqtSignal(int)

    def __init__(self, vpts_dict,pobj):
        self.vpts_dict = vpts_dict
        self.renderer = pobj.ren
        #self.AddObserver("LeftButtonPressEvent", self._left_button_press_event)
        self.AddObserver("RightButtonPressEvent", self._right_button_press_event)

        self.actors = []
        self.pobj = pobj

    #def _left_button_press_event(self, obj, event):
    def _right_button_press_event(self, obj, event):

        win = self.pobj.vtkWidget.GetRenderWindow()

        #this gives window coords, lower left is 0,0, x is left_right
        click_pos = self.GetInteractor().GetEventPosition()

        #scale = self.renderer.getDevicePixelRatioCompensation()
        #scale = 2
        #click_pos = [ p*scale for p in click_pos ]

        print("\n\n")
        print("actors",len(self.actors))
        if len(self.actors) > 0:
            for actor in self.actors:
                self.renderer.RemoveActor(actor)
            self.OnRightButtonDown()
            print("REMOVED")
            self.actors = []
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

        sfac = 1.0
        """
        #picking can be slow with lots of points
        self.OnLeftButtonDown()
        return
        """

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()        

        cell_picker = vtk.vtkCellPicker()
        #cell_picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())
        #cell_picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        cell_picker.Pick(sfac*click_pos[0], sfac*click_pos[1], 0, self.renderer)

        for proj in self.vpts_dict:
            vpts = self.vpts_dict[proj]
            glyph = vpts.glyph
            input_ids = glyph.GetOutput().GetPointData().GetArray("InputPointIds")

            print(type(input_ids))

            if input_ids:
                cell = glyph.GetOutput().GetCell(cell_picker.GetCellId())
                #print("cell",cell)
                if cell and cell.GetNumberOfPoints() > 0:
                    input_id = cell.GetPointId(0)
                    selected_id = input_ids.GetTuple1(input_id)
                    if selected_id >= 0:
                        highlight_sphere = vtk.vtkSphereSource() 

                        print("highlight")
                        highlight_sphere.SetRadius(vpts.r0)
                        highlight_sphere.SetThetaResolution(8)
                        highlight_sphere.SetPhiResolution(8)

                        print("selected",selected_id)

                        sid = int(selected_id)
                        n_cell = vpts.map0[sid]

                        

                        self.pobj.cell_info(n_cell,proj)

                        #self.cell_picked.emit(cell)
                        #get the world coords of the selected cell
                        pnt00 = glyph.GetInput().GetPoint(int(selected_id))

                        print("pnt00",pnt00)
                        coord.SetValue(pnt00)
                        dpnt00 = coord.GetComputedDisplayValue(self.renderer)

                        print("dpnt00",dpnt00)

                        dx = abs(dpnt00[0] - click_pos[0])
                        dy = abs(dpnt00[1] - click_pos[1])

                        print("dx dy",dx,dy)

                        ok = False

                        dmax = max(dx,dy)

                        if dmax <= 15:

                            highlight_sphere.SetCenter(glyph.GetInput().GetPoint(int(selected_id)))                    
                            mapper = vtk.vtkPolyDataMapper()
                            mapper.SetInputConnection(highlight_sphere.GetOutputPort())

                            highlight_actor = vtk.vtkActor()
                            highlight_actor.SetMapper(mapper)
                            highlight_actor.GetProperty().SetColor(1.0,0.0,1.0)
                            self.renderer.AddActor(highlight_actor)

                            self.actors.append(highlight_actor)

                        print("Pick!",proj,n_cell)
        #self.OnLeftButtonDown()
        self.OnRightButtonDown()
        return

def mk_maps(df,pcells):

    map0 = np.array(df.loc[pcells,:].index)
    npcells = len(pcells)

    idx0 = np.arange(npcells)

    N = df.shape[0]

    rmap0 = np.full((N,),-1,dtype=np.int)

    rmap0[map0] = idx0


    return map0,rmap0


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

#attempt to do stuff with camera
def save_image(mwin):
    im = vtk.vtkWindowToImageFilter()
    ren = mwin.ren
    #ren = mwin.vtkWidget.GetRenderWindow().Render()

    """
    rwin = mwin.vtkWidget.GetRenderWindow()
    im.SetInput(rwin)
    im.Update()
    """
    
    cam = ren.GetActiveCamera()

    print("got cam")

    """
    vtk_image = im.GetOutput()

    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()

    print("save",width,height,components)

    #arr = ns.vtk_to_numpy(vtk_array).reshape(height, width, components)

    arr = ns.vtk_to_numpy(vtk_array)

    arr2 = arr.reshape(height, width, components)

    print("arr",arr.shape,arr2.shape)

    print("aaf",arr[0,0],arr[-1,-1])

    np.savetxt("im_file.txt",arr)
    

    writer = vtk.vtkPNGWriter()

    writer.SetInputConnection(im.GetOutputPort())
    writer.SetFileName("file.png")
    writer.Write()
    """

    print("ren", type(ren))

    #this scales the model, nothing to do with camera ?
    #and just in z
    transform = vtk.vtkTransform()
    s_x = 1.0; s_y = 1.0; s_z = 3.0
    transform.Scale(s_x, s_y, s_z)
    cam.SetModelTransformMatrix(transform.GetMatrix())    
    

    print("save image")


def do_color2(mwin,mode_changed=False):

    cmap = mpl.cm.get_cmap('jet')

    for i,vpts in enumerate(mwin.vpoints_list):

        map0 = vpts.map0
        dfm = vpts.df_info.loc[map0,:]

        if i == 0:
            dlg = MRad(mwin,list(dfm.columns))
            dlg.setWindowTitle("marker")
            #put the dialog where is does not obscure the view
            dlg.move(50,50)
            dlg.exec_()

        m = mwin.m

        colors = dfm.loc[:,m].values

        #vmin = 0.0
        vmin = np.amin(colors)
        vmax = np.amax(colors)

        mwin.c0.cbar(vmin,vmax)
        mwin.c0.cb.set_label(m)
        mwin.c0.draw()

        

        colors = (colors - vmin)/(vmax-vmin)

        colors8 = cmap(colors,bytes=True)

        print("colors8",colors8[0])

        

        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        #this works
        vpts.p_data.GetPointData().RemoveArray('col8')

        #actor mapper can be gotten to change color array to use etc
        mapper = vpts.actor.GetMapper()

        scol8.SetName('col8')
        vpts.p_data.GetPointData().AddArray(scol8)

    #did not work as wanted
    #save_image(mwin)        

    mwin.vtkWidget.GetRenderWindow().Render()




def sparsity(mwin,mode_changed=False):
    #actually sparseness

    cmap = mpl.cm.get_cmap('jet')

    for i,vpts in enumerate(mwin.vpoints_list):

        map0 = vpts.map0
        dfm = vpts.df_info.loc[map0,:]

        if vpts.csk:
            csk = vpts.csk
            print("mst",csk.csr_mst.shape)
            sdist,nnz0 = get_sparsity(csk.csr_mst)

        m = "sparsity"

        #colors = dfm.loc[:,m].values
        colors = sdist[map0]

        #vmin = 0.0
        vmin = np.amin(colors)
        vmax = np.amax(colors)

        mwin.c0.cbar(vmin,vmax)
        mwin.c0.cb.set_label(m)
        mwin.c0.draw()

        colors = colors/np.amax(colors)

        colors8 = cmap(colors,bytes=True)

        print("colors8",colors8[0])

        

        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        #this works
        vpts.p_data.GetPointData().RemoveArray('col8')

        #actor mapper can be gotten to change color array to use etc
        mapper = vpts.actor.GetMapper()

        scol8.SetName('col8')
        vpts.p_data.GetPointData().AddArray(scol8)

    mwin.vtkWidget.GetRenderWindow().Render()


def critical00(mwin):
    #actually sparseness

    cmap = mpl.cm.get_cmap('jet')

    for i,vpts in enumerate(mwin.vpoints_list):

        map0 = vpts.map0
        dfm = vpts.df_info.loc[map0,:]

        if vpts.csk:
            csk = vpts.csk
            print("mst",csk.csr_mst.shape)
            sdist,nnz0 = get_sparsity(csk.csr_mst)

            c0 = critical0(csk)

            print(c0.crit_pnts)

        dfm = vpts.df_info.loc[c0.crit_pnts,:]

        print("dfm",dfm.shape)

        vtk_rscale = vpts.p_data.GetPointData().GetArray("scales")
        rscale = ns.vtk_to_numpy(vtk_rscale)

        print("rscale",rscale.dtype)

        print(rscale[0])

        m = "critical"

        #colors = dfm.loc[:,m].values
        colors = sdist[map0]

        #vmin = 0.0
        vmin = np.amin(colors)
        vmax = np.amax(colors)

        mwin.c0.cbar(vmin,vmax)
        mwin.c0.cb.set_label(m)
        mwin.c0.draw()

        colors = colors/np.amax(colors)

        colors8 = cmap(colors,bytes=True)

        print("colors8",colors8[0])

        

        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        #this works
        vpts.p_data.GetPointData().RemoveArray('col8')

        #actor mapper can be gotten to change color array to use etc
        mapper = vpts.actor.GetMapper()

        scol8.SetName('col8')
        vpts.p_data.GetPointData().AddArray(scol8)

    mwin.vtkWidget.GetRenderWindow().Render()
    
def critical(mwin,mode_changed=False):

    cmap = mpl.cm.get_cmap('jet')

    for i,vpts in enumerate(mwin.vpoints_list):
        csk = vpts.csk

        map0 = vpts.map0
        dfm = vpts.df_info.loc[map0,:]

        if i == 0:
            dlg = MRad(mwin,list(dfm.columns))
            dlg.setWindowTitle("marker")
            dlg.exec_()

        m = mwin.m

        colors = dfm.loc[:,m].values

        #vmin = 0.0
        vmin = np.amin(colors)
        vmax = np.amax(colors)

        mwin.c0.cbar(vmin,vmax)
        mwin.c0.cb.set_label(m)
        mwin.c0.draw()

        colors = colors/np.amax(colors)

        colors8 = cmap(colors,bytes=True)

        print("colors8",colors8[0])

        c0 = critical0(csk,n_iter=4)

        crit_pnts0 = vpts.rmap0[c0.crit_pnts]

        print("crit_pnts",crit_pnts0)

        vtk_rscale = vpts.p_data.GetPointData().GetArray("scales")
        rscale = ns.vtk_to_numpy(vtk_rscale)

        fac = 2.0
        rscale /= fac

        rscale[crit_pnts0] *= fac
        


        scol8 = ns.numpy_to_vtk(num_array=colors8, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        #this works
        vpts.p_data.GetPointData().RemoveArray('col8')

        #actor mapper can be gotten to change color array to use etc
        mapper = vpts.actor.GetMapper()

        scol8.SetName('col8')
        vpts.p_data.GetPointData().AddArray(scol8)

    mwin.vtkWidget.GetRenderWindow().Render()

class vpoints:
    def __init__(self,data,rscale,src,name=None,csk=None):
        self.data = data
        self.name = name
        self.csk = csk
        points = vtk.vtkPoints()
        p_data = vtk.vtkPolyData() #note polydata
        self.p_data = p_data

        npnts = data.shape[0]
        points.SetNumberOfPoints(npnts)

        data2 = ns.numpy_to_vtk(num_array=data, deep=True, array_type=vtk.VTK_FLOAT)
        points.SetData(data2)

        #this will be used to scale the glyph size
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
        #glyph.SetSourceConnection(sphere.GetOutputPort())
        #glyph.SetSourceConnection(cube.GetOutputPort())
        glyph.SetSourceConnection(src.GetOutputPort())
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        #with out this the scales array is used for color
        #it seems to be necessary for using the color array
        #in later updates as well
        mapper.SetScalarModeToUsePointFieldData()

        #this version again yields white colors, despite col8
        #mapper.SetScalarModeToUseCellData()        


        white = np.full( (npnts,4),255,dtype=np.uint8)
        white[:,2] = 0
        scol8 = ns.numpy_to_vtk(num_array=white, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        scol8.SetName('col8')
        p_data.GetPointData().AddArray(scol8)
        

        #this sets the color array to be "col8", even tho it is not there yet
        #but it will be later in do_color2
        mapper.SelectColorArray("col8")

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        #white = np.full( (npnts,4),255,dtype=np.uint8)
        

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


def mk_transform(csk):
    pcells = list(csk.br_adj.keys())
    tcols = csk.traj_markers
    
    tdata0 = csk.df_avg.loc[pcells,tcols].values
    n = tdata0.shape

    pca = csk1.pca_coords(tdata0)

    uut = pca.uu.T

    uut = list(uut)

    print(uut[0].shape)

    uut.append(pca.mu)

    uut = np.array(uut)


    df_uut = pd.DataFrame(uut,columns=tcols)

    #df_uut.to_csv("df_uut.csv",index=False)
    return df_uut



class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)

        self.frame = QFrame()
        self.vl = QHBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.vpoints_list = []
        self.vpoints_dict = {}

        sphere = vtk.vtkSphereSource()
        ndiv = 8
        sphere.SetThetaResolution(ndiv)
        sphere.SetPhiResolution(ndiv)
        #this sets the original radius of
        #the sphere for the glyph
        sphere.SetRadius(.5)

        cube = vtk.vtkCubeSource()
        cube.SetXLength(.75)

        cone = vtk.vtkConeSource()
        cone.SetRadius(.5)
        cone.SetHeight(1.0)
        

        self.src_list = [sphere,cone,cube]
        

        #set up text widget
        font = QFont()

        font.setFamily("Courier")
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(16)

        txt = QPlainTextEdit()
        txt.setFont(font)
        txt.setMaximumWidth(400)


        #p = txt.palette()

        #ok this works to set txt back color
        txt.setStyleSheet("background-color: #999999")

        self.txt = txt        

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
        self.fileMenu.addAction(self.get_transform_action)        
        
        self.actionMenu = self.menuBar().addMenu("&Action")
        self.actionMenu.addAction(self.color_action)
        self.actionMenu.addAction(self.density_action)
        self.actionMenu.addAction(self.critical_action)                
        self.actionMenu.addAction(self.slide_action)

        self.modeMenu = self.menuBar().addMenu("&Mode")
        self.modeMenu.addAction(self.mode1_action)
        
    def createActions(self):
        self.open_project_action = QAction("&Open Project...", self,triggered=self.open_project)
        self.get_transform_action = QAction("&Get Transform...", self,triggered=self.get_transform)        
        
        self.color_action = QAction("&Color", self, triggered=self.do_color)
        self.density_action = QAction("&Sparsity", self, triggered=self.do_sparsity)
        self.critical_action = QAction("&Critical", self, triggered=self.do_critical)        
        self.slide_action = QAction("Size", self, triggered=self.do_slide)

        self.mode1_action = QAction("View", self, triggered=self.do_mode1)

    def do_mode1(self):
        mch = MCheck(self,['cloud'])

        #modifies self.mlist
        mch.exec_()



    def do_slide(self):
        dlg = MSlide(self)
        dlg.setWindowTitle("HELLO!")
        dlg.exec_()

    def do_color(self,mode_changed=False):
        do_color2(self,mode_changed)
        
    def do_sparsity(self):
        sparsity(self)
        
    def do_critical(self):
        critical(self)



    def cell_info(self,cell,proj):
        info = ["cell",proj,str(cell)]

        vpts = self.vpoints_dict[proj]

        df_info = vpts.df_info

        #markers = self.df_info.columns
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
        #cell_data = self.df_info.loc[cell,:].values
        cell_data = df_info.loc[cell,:].values
        
        for i,m in enumerate(markers):
            cd = cell_data[i]
            if isinstance(cd,(int,float)):
                info.append(mw % m + " " + "%9.3f" % cell_data[i])
            else:
                info.append(mw % m + " : " + str(cd))

        qstr = "\n".join(info)

        self.txt.setPlainText(qstr)


    def get_transform(self):
        tfile = QFileDialog.getOpenFileName(self, "Tranform File")
        tfile = tfile[0]
        df_uut = pd.read_csv(tfile)

        self.df_uut = df_uut

    def open_project(self):
        #pdir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        #The native mode dialog has problems
        dlg.setOption(QFileDialog.DontUseNativeDialog)
        #dlg.DontUseNativeDialog()

        if dlg.exec_():
            plist = dlg.selectedFiles()

            
        #pdir = str(dlg.getExistingDirectory(None, "Select Directory"))
        pdir = plist[0]
        dlg.close()
        del dlg

        if pdir == "" or pdir == None: return

        self.dir_name = pdir

        csk = csk1.cytoskel(pdir)
        if not csk.open():
            print("Invalid project")
            return
        else:
            self.csk = csk

        self.vpts = dset(csk,self)

        self.vpoints_list.append(self.vpts)
        self.vpoints_dict[pdir] = self.vpts

        qstr = pdir
        self.txt.setPlainText(qstr)

        self.ren.RemoveAllViewProps()
        self.ren.SetBackground(.8,.8,.8)
        
        #self.use_data(self.vpts)
        #self.use_data()
        self.use_data2()

        #need to render
        self.vtkWidget.GetRenderWindow().Render()                

        #ok this solved the view scaling problem
        self.ren.ResetCamera()

        #this solved the problem of not displaying
        #until window click
        self.vtkWidget.update()
        

    def use_data2(self,vpts=None):
        for proj in self.vpoints_dict:
            vpts = self.vpoints_dict[proj]
            self.ren.AddActor(vpts.actor)
            self.ren.AddActor(vpts.line_actor)
        style = InteractorStyle(self.vpoints_dict,self)

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
    n = tdata0.shape

    if hasattr(mwin,"df_uut"):
        df_uut = mwin.df_uut
    else:
        #make df_uut from project trajectory
        df_uut = mk_transform(csk)
        mwin.df_uut = df_uut

    Xuut = df_uut.values
    uu = Xuut[:-1].T
    mu = Xuut[-1]

    ux5 = (tdata0 - mu) @ uu
    
    ux5 = ux5[:,:3]
    un5 = la.norm(ux5,axis=1)
    urad5 = np.amax(un5)

    ux,urad = ux5,urad5

    if os.path.exists(csk.project_dir+"df_display_coords.csv"):
        df_display_coords = pd.read_csv(csk.project_dir+"df_display_coords.csv")
        print(csk.project_dir+"df_display_coords.csv")

        ux = df_display_coords.values[pcells,:3]

        unorms = la.norm(ux,axis=1)
        urad = np.amax(unorms)
        
        
    scale = 5.0/urad

    #do scaling based on urad
    data = ux[:,:3]*scale

    hdata = ["pca00","pca01","pca02"]
    df_data = pd.DataFrame(data,columns=hdata,index=map0)

    br_adj = csk.br_adj

    #correct values
    r0 = .12

    npnts = data.shape[0]        
    rscale = np.full(npnts,r0)

    ivpnts = len(mwin.vpoints_list)

    vpts = vpoints(data,rscale,src=mwin.src_list[ivpnts % 2],csk=csk)
    points = vpts.points

    vpts.add_lines(br_adj,rmap0,r0)

    vpts.map0 = map0
    vpts.rmap0 = rmap0

    print("map0 set")    
    vpts.r0 = r0


    #mwin.df_info = csk.df_avg
    #mwin.vpts = vpts
    vpts.df_info = csk.df_avg

    return vpts






    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    #window.setGeometry(200,200,1000,600);
    window.setGeometry(150,100,1500,900)
    sys.exit(app.exec_())
