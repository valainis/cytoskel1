library(reticulate)
library(data.table)
csk1 = import("cytoskel1")

#use_python("/Users/valainis/miniconda3/envs/csk37/bin/python")

#this os import is essential to make it work on enki (why????)
#os <- import("os")



shutil = import("shutil")

#py_config()

dt <- fread("../data/csk8.csv")

traj.cols = c('CD45', 'CD19', 'IgD', 'CD79b', 'CD20', 'CD34', 'CD179a',
    'CD72', 'IgM-i', 'Kappa', 'CD10', 'Lambda', 'CD179b', 'CD24', 'CD38',
    'CD117', 'HLADR', 'IgM-s')

avg.cols = c('CD3-1', 'CD3-2', 'CD235-61-66b', 'CD3-3', 'CD45', 'cPARP', 'pPLCg',
    'CD19', 'CD22', 'pSrc', 'IgD', 'CD79b', 'CD20', 'CD34', 'CD179a', 'pSTAT5',
    'CD72', 'Ki67', 'IgM-i', 'Kappa', 'CD10', 'Lambda', 'CD179b', 'pAKT', 'CD49d',
    'CD24', 'CD127', 'RAG1', 'TdT', 'Pax5', 'pSyk', 'pErk12', 'CD38', 'pP38',
    'CD40', 'CD117', 'pS6', 'CD33-11c-16', 'HLADR', 'IgM-s', 'pCreb')

csk = csk1$cytoskel("../proj/r_tgen80")

csk$create(dt,
           traj.cols,
           l1_normalize=TRUE
           )



csk$do_graphs()

csk$do_branches(-1,4)
csk$get_average_fix(avg.cols,navg=5,ntree=4)

pdir0 = "../proj/py_tgen8"

csk$check_same(pdir0)
shutil$rmtree(csk$project_dir)
