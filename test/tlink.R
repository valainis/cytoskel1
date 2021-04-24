library(reticulate)
library(data.table)
csk1 = import("cytoskel1")

#this import is essential to make it work on enki (why????)
#os <- import("os")

shutil = import("shutil")

traj.cols = c('BRACHYURY', 'HAND1', 'ISL1', 'STAT3_total', 'ZNF207',
             'PRDM14', 'OCT4', 'TBX6', 'GATA4', 'PBX1', 'cMYC',
             'NANOG', 'GATA6', 'SOX2', 'LIN28a', 'FOXA2', 'EOMES')

avg.cols = c('barcode1', 'barcode2', 'barcode3', 'barcode4', 'barcode5', 'barcode6',
    'pHH3_S10', 'A2B5', 'IdU', 'barium', 'cPARP', 'NRF2', 'BRACHYURY', 'HAND1',
    'ISL1', 'STAT3_total', 'ZNF207', 'CD34', 'PRDM14', 'OCT4', 'TBX6', 'AuroraK_A',
    'CD326', 'GATA4', 'HIF1a', 'PBX1', 'cMYC', 'Puromycin', 'NANOG', 'GATA6', 'nMYC',
    'CyclinB1', 'SOX2', 'CD140b', 'GATA3', 'MEIS2', 'LIN28a', 'TWIST1', 'REX1',
    'CD309_KDR', 'FOXA2', 'H3K27me3', 'JARID2', 'EOMES', 'BRU', 'CD56', 'DNA1', 'DNA2')

dt <- fread("../data/tg1.csv")

pdir = "../proj/r_tlink8"
csk = csk1$cytoskel(pdir)

csk$create(dt,
           traj.cols,
           l1_normalize=TRUE,
           level_marker='day'
           )


csk$do_graphs()

csk$do_branches(-1,6)

csk$get_average_fix(avg.cols,navg=5,ntree=4)


pdir0 = "../proj/py_tlink8"
csk$check_same(pdir0)
shutil$rmtree(csk$project_dir)
