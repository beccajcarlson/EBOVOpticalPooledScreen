import pandas as pd
import ops.triangle_hash
import sys
import warnings
    
def run_hash(df_ph_info_hash_loc,df_sbs_info_hash_loc,save_loc,initial_sites,df_ph_xy,df_sbs_xy,batch_size=20,n_init=50,wells=None):
    print(save_loc)
    
    df_ph_info_hash = pd.read_hdf(df_ph_info_hash_loc)
    df_sbs_info_hash = pd.read_hdf(df_sbs_info_hash_loc)
    
    if wells != None:
        df_ph_info_hash = df_ph_info_hash[df_ph_info_hash.well.isin(wells)]
        df_sbs_info_hash = df_sbs_info_hash[df_sbs_info_hash.well.isin(wells)]

    warnings.filterwarnings('ignore')

    wells = pd.unique(df_ph_info_hash.well)
    det_range = (.9, 1.15)

    for well in wells:
        print(well)  

        df_0 = df_ph_info_hash.query('well == @well').loc[:,~df_ph_info_hash.columns.isin(['well'])]
        df_1 = df_sbs_info_hash.query('well == @well').loc[:,~df_sbs_info_hash.columns.isin(['well'])]

        df_info_0 = df_ph_xy[df_ph_xy.index.isin(df_0.tile)]
        df_info_1 = df_sbs_xy[df_sbs_xy.index.isin(df_1.site)]
        #print(len(pd.unique(df_1.site)))
        print(df_info_0.shape[0])
        print(df_info_1.shape[0])

        df_align = ops.triangle_hash.multistep_alignment(df_0, df_1, df_info_0, df_info_1,
                                                         det_range=det_range,
                                                         initial_sites=initial_sites[:n_init],batch_size=batch_size)
        df_align.to_hdf('{0}_{1}.hdf'.format(save_loc,well), 'x', mode='w')



