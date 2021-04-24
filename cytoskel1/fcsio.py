import numpy as np
import sys,os
from sortedcontainers import SortedDict,SortedList
import struct
from collections import OrderedDict

import time

def write_fcs(fname, wdata, markers):

    fcs3  = bytes("FCS3.0    ","ascii")
    space = b' '
    delim = b'/'
    fpos = 0

    text_start = 255;
    #int text_end;

        


    text =  "$DATATYPE/F/"
    text += ("$PAR/" + str(len(markers)) + "/")
    text += ("$TOT/" + str(len(wdata)) + "/")
    text += "$MODE/L/"
    text += "$BYTEORD/1,2,3,4/"
    
    
    for i,marker in enumerate(markers):
        text += ("$P" + str(i+1) + "N/" + marker + "/")
        text += ("$P" + str(i+1) + "S/" + marker + "/")
        text += ("$P" + str(i+1) + "B/" + "32" + "/")
        text += ("$P" + str(i+1) + "E/" + "0,0" + "/")
        text += ("$P" + str(i+1) + "R/" + "262144" + "/")

    wdata = np.array(wdata, dtype=np.float32)
    wdata = wdata.flatten()
    data_size = wdata.nbytes

    extra_text_len = 50
    text_end = text_start+len(text)+extra_text_len
    data_start = text_end+256;
    data_end = data_start + data_size-1;

    extra_text =  "$BEGINDATA/" + str(data_start) + "/"
    extra_text +=  "$ENDDATA/" + str(data_end) + "/"

    lextra = len(extra_text)

    text += extra_text
    len_space = extra_text_len - lextra
    text += " "*len_space

    
    text = bytes(text,"ascii")

    # convert wdata to flat float



    of = open(fname,"wb")

    of.write(fcs3)
    fpos += len(fcs3)

    if data_size < 99999999:
        header = "%8d" % text_start + "%8d" % text_end\
        + "%8d" % data_start + "%8d" % data_end + "%8d" % 0 + "%8d" % 0
    else:
        header = "%8d" % text_start + "%8d" % text_end\
        + "%8d" % 0 + "%8d" % 0 + "%8d" % 0 + "%8d" % 0
        
    header = bytes(header,"ascii")

    of.write(header)
    fpos += len(header)

    for i in range(fpos,text_start):
        of.write(space)
    of.write(delim)

    of.write(text)

    for i in range(text_end+1,data_start):
        of.write(space)        

    of.write(wdata.tobytes())

    of.close()    


class fcsio:
    def __init__(self,fname):

        print("")
        print("reading fcs file:",fname)
        self.fcs_name = fname

        try:
            f = open(fname,mode="rb")
        except:
            print("no such file", fname)
            exit()

        dname = fname.split(".")
        self.dir_name = ".".join(dname[:-1])
            
            
        b = f.read()
        f.close()

        isize = 8
        istart = 10
        offsets = []


        for i in range(6):
            int0 = int(b[istart:istart+isize])
            istart = istart+isize;
            offsets.append(int0)

        delim = chr(b[offsets[0]])
        stext = offsets[0]+1

        #print("delim",delim)

        text = b[stext:offsets[1]+1]
        text = str(text)
        ilast = text.rfind(delim)
        text = text[:ilast+1]

        #tlist is the split up text segment
        tlist = text.split(delim)

        #must treat clumps of delim as 1 delim
        tlist = [t for t in tlist if t != ""]

        tmap = SortedDict()

        #tmap key is marker designation like $PnS, tmap value is name
        for i,s in enumerate(tlist):
            if "$" in s:
                # name follows marker designation in tlist
                tmap[s] = tlist[i+1]

        #gather the S markers
        mark_s = SortedDict()

        #also the N markers
        mark_n = SortedDict()

        
        for key in tmap:
            if key[0:2] == "$P" and key[-1] == "S":
                #keyval is fcs marker number, 1 to N
                keyval = int(key[2:-1])
                #value in mark_s is the marker name
                mark_s[keyval] = tmap[key]

        for key in tmap:
            if key[0:2] == "$P" and key[-1] == "N":
                #keyval is fcs marker number, 1 to N
                keyval = int(key[2:-1])
                #some markers have no S name, but do have N name
                mark_n[keyval] = tmap[key]
                

        if offsets[2] == 0:
            print("begindata",tmap["$BEGINDATA"])
            print("enddata",tmap["$ENDDATA"])
            offsets[2] = int(tmap["$BEGINDATA"])
            offsets[3] = int(tmap["$ENDDATA"])

        #all markers should be in mark_n
        nparam = len(mark_n)

        # python idx of marks in orignal data
        # python idx will be 1 less than the fcs marker number
        mark_idx = {}

        mark_all = SortedDict()

        self.markers = []        
        for key in mark_n:
            if key in mark_s:
                kname = mark_s[key]
            else:
                kname = mark_n[key]
            mark_all[key] = kname
                
            mark_idx[kname] = key-1
            self.markers.append(kname)

        self.mark_idx = mark_idx
        self.mark_s = mark_s
        self.mark_n = mark_n
        self.mark_all = mark_all

        #get the data
        bdata = bytearray(b[offsets[2]:offsets[3]+1])

        #print("bdata",len(bdata),offsets[2],offsets[3]+1)

        #trim if necessary
        blen = len(bdata)
        bextra = blen % 4
        if bextra > 0:
            bdata = bdata[:-bextra]

        #print("byte order",tmap["$BYTEORD"])

        if tmap["$BYTEORD"] == "4,3,2,1":
            #data = struct.unpack(">%uf"%(len(bdata)//4), bdata)
            data = struct.unpack(">%df"%(len(bdata)//4), bdata)
        else:
            #data = struct.unpack("<%uf"%(len(bdata)//4), bdata)
            data = struct.unpack("<%df"%(len(bdata)//4), bdata)

        data = np.array(data)

        if len(data) % nparam != 0:
            print("data length error",len(data)%nparam)
            exit()

        #self.data is now the full data array as float64
        self.data = data.reshape((-1,nparam))
        #print(type(self.data),self.data.dtype)

        print("fcs file read")


class setup_data:

    props = {"neighbors":int,
             "branchings":int,
             "pre-divisor":float,
             "cutoff":float,
             "transform":str,
             "knn_distance":str,
             "knn_algo":str,
             "avg_method":str,
             "start":int
             }
    
    defaults = {"neighbors":30,
                "branchings":4,
                "pre-divisor":5.0,
                "cutoff":.5,
                "transform":"arcsinh",
                "knn_distance":"L1_normal_euclid",
                "knn_algo":"exact",
                "avg_method":"mst_nn",                             
                "start":-1
                }

    choices = {"transform":["arcsinh","identity","log2"],
               "knn_distance":["L1_normal_euclid","euclid"]
                }


    usage_map = {"ignore":0,"trajectory":1,"average":2,"monitor":3}
    usage2_map = {0:"ignore",1:"trajectory",2:"average",3:"monitor"}                

    def __init__(self,markers=[],sfile=None):
        markers = list(markers)
        self.usage = SortedDict()
        self.markers = []
        self.use1 = {}
        self.mark_idx = {}
        self.globals = dict(self.defaults)
        
        if markers != []:
            self.markers = list(markers)
            self.default_init()
        elif sfile:
            self.read_setup(sfile)
        else:
            print("no input, exiting")

    def read_setup(self,sfile):
        lines = sfile.readlines()
        start_markers = False
        usage0 = []

        for line in lines:
            #strip any white space surrounding the text
            line = line.strip()

            #skip over empty lines and comments
            if line == "" or line[0] == "#":
                continue

            if "marker" in line:
                start_markers = True
                continue            

            line = line.split(',')
            line = [x.strip() for x in line]


            if not start_markers:
                if line[0] in self.props:
                    val = self.props[line[0]](line[1])
                    self.globals[line[0]] = val
                else:
                    #unrecognized property
                    print("unrecognized",line)
                    continue
            else:
                usage0.append(int(line[1]))
                self.markers.append(line[0])
                self.use1[line[0]] = int(line[1])

        for i,useit in enumerate(usage0):
            self.usage[i] = useit
            self.use1[self.markers[i]] = useit

        self.set_idx()

        self.set_transforms()


    def set_usage(self,idx,useit):
        self.usage[idx] = useit
        m = self.markers[idx]
        self.use1[m] = useit

    def set_transforms(self):

        if self.globals["transform"] == 'identity':
            print('identity')
            self.do_transform = False
        else:
            self.do_transform = True

        if self.globals["knn_distance"] == "L1_normal_euclid":
            self.do_normalize1 = True
        elif self.globals["knn_distance"] == "euclid":
            self.do_normalize1 = False
            
        self.transforms = {}
        self.pre_divs = {}
        for m in self.use1:
            useit = self.use1[m]
            self.transforms[m] = self.globals["transform"]
            #print("transform",m,self.globals["transform"])
            self.pre_divs[m] = self.globals["pre-divisor"]

        
    def set_idx(self):
        self.idx_mark = {}
        self.markers = np.array(self.markers)
        for i,m in enumerate(self.markers):
            self.mark_idx[m] = i
            self.idx_mark[i] = m


        self.traj_idx = SortedList()
        self.avg2_idx = SortedList()
        self.mon_idx = SortedList()
        self.all_avg_idx = SortedList()

        self.channels = [[],[],[],[]]

        #check the use1 markers
        for m in self.use1:
            useit = self.use1[m]
            idx = self.mark_idx.get(m)            
            #1 use for distance calculations, 2 other markers that can be averaged, 3 just monitor
            self.channels[useit].append(m)
            if useit == 1:
                self.traj_idx.add(idx)
                self.all_avg_idx.add(idx)
            elif useit == 2:
                self.avg2_idx.add(idx)
                self.all_avg_idx.add(idx)              
            if useit >= 1 and useit <= 3:
                self.mon_idx.add(idx)

        #the various headers
        self.traj_markers = [self.idx_mark[idx] for idx in self.traj_idx]
        #self.traj_markers = self.markers[self.traj_idx]
        self.avg2_markers = [self.idx_mark[idx] for idx in self.avg2_idx]
        self.mon_markers =  [self.idx_mark[idx] for idx in self.mon_idx]

        #print("traj markers",len(self.traj_markers))
        #print("avg markers",len(self.avg2_markers))
        #print("mon markers",len(self.mon_markers))

    def marker_by_idx0(self,idx):
        return self.idx_mark[idx]

    def idx0_by_marker(self,m):
        return mark_idx[m]

    def get_traj_idx0(self):
        return self.traj_idx
    
    def get_all_avg_idx0(self):
        """
        """
        return all_avg_idx
            

    def default_init(self):
        exclude = ['event','time','beaddist','dna','bar']

        self.use1 = {}


        print("default init setup")

        #now do the markers
        #order is same as in fcs file

        for i,m in enumerate(self.markers):
            mlow = m.lower()

            useit = 1
            
            for x in exclude:
                if x in mlow: useit = 0
            if "bead" in mlow[:4]: useit = 0
            elif "cell" in mlow[:4]: useit = 0
            elif m[0].islower(): useit = 2

            self.use1[m] = useit
            self.usage[i] = useit

        self.extra = {}
        self.cutoffs ={}


    def reset_use(self,traj_markers,mon_markers,exclude):

        for i,m0 in enumerate(self.markers):
            if m0 in traj_markers:
                self.usage[i] = 1
            elif m0 in mon_markers:
                self.usage[i] = 3
            elif m0 in exclude:
                self.usage[i] = 0
            else:
                self.usage[i] = 2
        



    def write_setup(self,sname):
        f = open(sname,"w")

        for key in self.globals:
            s0 = "%-12s" % key + " , " + str(self.globals[key]) + "\n"
            f.write(s0)

        s0 = "%-12s" % "marker" + " , " + "use" + "\n"
        f.write(s0)

        #print("extra",setup.extra)

        for i,m0 in enumerate(self.markers):
            useit = self.usage[i]
            s1 = "%-12s" % m0 + " , " + "%3d" % useit
            f.write(s1)
            """
            if m0 in self.extra:
                extra = setup.extra[m0]
                s1 = " , %-12s" % extra[0]
                f.write(s1)
            """
            f.write("\n")
        f.write("\n")
        f.close()


        
