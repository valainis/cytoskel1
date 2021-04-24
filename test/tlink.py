if __name__ == '__main__':
    import sys
    import time
    import shutil

    import pandas as pd
    import cytoskel1 as csk1


    traj_cols = ['BRACHYURY', 'HAND1', 'ISL1', 'STAT3_total', 'ZNF207',
                 'PRDM14', 'OCT4', 'TBX6', 'GATA4', 'PBX1', 'cMYC',
                 'NANOG', 'GATA6', 'SOX2', 'LIN28a', 'FOXA2', 'EOMES']




    avg_cols = ['barcode1', 'barcode2', 'barcode3', 'barcode4', 'barcode5', 'barcode6',
        'pHH3_S10', 'A2B5', 'IdU', 'barium', 'cPARP', 'NRF2', 'BRACHYURY', 'HAND1',
        'ISL1', 'STAT3_total', 'ZNF207', 'CD34', 'PRDM14', 'OCT4', 'TBX6', 'AuroraK_A',
        'CD326', 'GATA4', 'HIF1a', 'PBX1', 'cMYC', 'Puromycin', 'NANOG', 'GATA6', 'nMYC',
        'CyclinB1', 'SOX2', 'CD140b', 'GATA3', 'MEIS2', 'LIN28a', 'TWIST1', 'REX1',
        'CD309_KDR', 'FOXA2', 'H3K27me3', 'JARID2', 'EOMES', 'BRU', 'CD56', 'DNA1', 'DNA2']


    df = pd.read_csv("../data/tg1.csv")


    t0 = time.time()


    csk = csk1.cytoskel("../proj/py_tlink8_1")
    csk.create(df,traj_cols,
                   l1_normalize=True,
                   level_marker='day')


    t1 = time.time()

    print("setup time",t1 - t0)

    #csk.link("day")
    csk.do_graphs(n_process=4)


    csk.do_branches(-1,6)

    csk.get_average_fix(avg_cols,navg=5,ntree=4)

    pdir0 = "../proj/py_tlink8"

    same,avg_same = csk.check_same(pdir0)
    print(same,avg_same)
    shutil.rmtree(csk.project_dir)




