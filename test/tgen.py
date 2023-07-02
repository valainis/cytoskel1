if __name__ == '__main__':
    import sys
    import time
    import shutil

    import pandas as pd

    #from cytoskel1 import *
    import cytoskel1 as csk1

    traj_cols = ['CD45', 'CD19', 'IgD', 'CD79b', 'CD20', 'CD34', 'CD179a',
        'CD72', 'IgM-i', 'Kappa', 'CD10', 'Lambda', 'CD179b', 'CD24', 'CD38',
        'CD117', 'HLADR', 'IgM-s']

    avg_cols = ['CD3-1', 'CD3-2', 'CD235-61-66b', 'CD3-3', 'CD45', 'cPARP', 'pPLCg',
        'CD19', 'CD22', 'pSrc', 'IgD', 'CD79b', 'CD20', 'CD34', 'CD179a', 'pSTAT5',
        'CD72', 'Ki67', 'IgM-i', 'Kappa', 'CD10', 'Lambda', 'CD179b', 'pAKT', 'CD49d',
        'CD24', 'CD127', 'RAG1', 'TdT', 'Pax5', 'pSyk', 'pErk12', 'CD38', 'pP38',
        'CD40', 'CD117', 'pS6', 'CD33-11c-16', 'HLADR', 'IgM-s', 'pCreb']

    df = pd.read_csv("../data/csk8.csv")

    t0 = time.time()
    csk = csk1.cytoskel("../proj/py_tgen8_1")
    csk.create(df,traj_cols,avg_markers=avg_cols,l1_normalize=True)

    t1 = time.time()

    print("setup time",t1 - t0) 

    csk.do_graphs(n_process=4)

    csk.do_branches0(-1,4)

    csk.get_average_fix(avg_cols,navg=5,ntree=4)


    pdir0 = "../proj/py_tgen8"

    same,avg_same = csk.check_same(pdir0)
    print(same,avg_same)
    shutil.rmtree(csk.project_dir)
