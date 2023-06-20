#import sys
#sys.path.remove('/home/rcarlson/OpticalPooledScreens')
#sys.path.append("/home/rcarlson/NPOpticalPooledScreens")

#from ops.imports_ipython import *

#import skimage
#import umap.umap_ as umap
import pandas as pd
import time
import pynndescent
import numpy as np

from scipy.spatial.distance import pdist, jaccard
from scipy.spatial.distance import squareform
from sklearn.neighbors import kneighbors_graph
from pandarallel import pandarallel
import gseapy as gp

def get_gsea_ranks(df, metric, saveloc, savename, gene_set, compute_l1 = False, ascending = False, seed = 7):
   # df = df.reset_index()
    #print(df.head())
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    df = df.groupby('gene_symbol').mean().reset_index()
    #print(df.head())
    #print(df.shape)
    if compute_l1 == True:
        pandarallel.initialize(nb_workers=20, verbose = 0)
    ## for multi-feature vectors, compute distance between vectors first
        nt =  df[df.gene_symbol.str.contains('nontargeting')][metric].mean(axis = 0)
        df['l1'] =  df[metric].parallel_apply(lambda x: np.abs(x - nt).sum(), axis = 1)
        metric = 'l1'

    #print(df.head())

    df[metric] = df[metric].rank(axis = 0, ascending = False) ### perform gsea on rankings, not raw values
    #print(df.head())
    pre_res = gp.prerank(rnk=df[['gene_symbol',metric]], 
                         gene_sets = gene_set, #'/home/rcarlson/gw/ebov/info/geneset.gmt', # permutation_type = 'gene_set',
                         processes=2, ascending = False,
                         permutation_num=100000, # reduce number to speed up testing
                         outdir=saveloc, format='tif', seed=seed, verbose = False)
    pre_res.res2d.to_csv(saveloc + savename + 
                 '_seed' + str(seed) + '.csv')
    #return pre_res.res2d
def get_gsea_ranks_scaledvals(df, metric, saveloc, savename, gene_set, compute_l1 = False, ascending = False, seed = 7):
   # df = df.reset_index()
    #print(df.head())
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    df = df.groupby('gene_symbol').mean().reset_index()
    #print(df.head())
    #print(df.shape)
    if compute_l1 == True:
        pandarallel.initialize(nb_workers=20, verbose = 0)
    ## for multi-feature vectors, compute distance between vectors first
        nt =  df[df.gene_symbol.str.contains('nontargeting')][metric].mean(axis = 0)
        df['l1'] =  df[metric].parallel_apply(lambda x: np.abs(x - nt).sum(), axis = 1)
        metric = 'l1'

    #print(df.head())
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    #print(scaler.fit_transform(np.array(df[metric]).reshape(-1, 1)).shape)
    #print(np.array(df[metric]).reshape(-1, 1))
    df[metric] = scaler.fit_transform(np.array(df[metric]).reshape(-1, 1))
    #print(df[metric].shape)
    #df[metric] = df[metric].rank(axis = 0, ascending = False) ### perform gsea on rankings, not raw values
    #print(df.head())
    pre_res = gp.prerank(rnk=df[['gene_symbol',metric]], 
                         gene_sets = gene_set, #'/home/rcarlson/gw/ebov/info/geneset.gmt', # permutation_type = 'gene_set',
                         processes=2, ascending = ascending,
                         permutation_num=100000, # reduce number to speed up testing
                         outdir=saveloc, format='tif', seed=seed, verbose = False)
    pre_res.res2d.to_csv(saveloc + savename + 
                 '_seed' + str(seed) + '.csv')
    #return pre_res.res2d

def get_gsea_ranks_pearson(df, metric, saveloc, savename, gene_set, compute_p = False, ascending = False, seed = 7):
   # df = df.reset_index()
    #print(df.head())
    #print(ascending)
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    df = df.groupby('gene_symbol').mean().reset_index()
    #print(df.head())
    #print(df.shape)
    import scipy
    if compute_p == True:
        #print('computing p')
        pandarallel.initialize(nb_workers=20, verbose = 0)
    ## for multi-feature vectors, compute distance between vectors first
        nt =  df[df.gene_symbol.str.contains('nontargeting')][metric].mean(axis = 0)
        df['pearson'] =  df[metric].parallel_apply(lambda x: scipy.stats.pearsonr(x, nt)[0], axis = 1)
        metric = 'pearson'

    #print(df.head())

    df[metric] = df[metric].rank(axis = 0, ascending = ascending) ### perform gsea on rankings, not raw values
    #print(df.head())
    pre_res = gp.prerank(rnk=df[['gene_symbol',metric]], 
                         gene_sets = gene_set, #'/home/rcarlson/gw/ebov/info/geneset.gmt', # permutation_type = 'gene_set',
                         processes=2, ascending = False,
                         permutation_num=100000, # reduce number to speed up testing
                         outdir=saveloc, format='tif', seed=seed, verbose = False)
    pre_res.res2d.to_csv(saveloc + savename + 
                 '_seed' + str(seed) + 'pearson.csv')


def get_jaccard_similarities_fromknn(df, features, saveloc, savename,
                                     n_neighbors_local = 50, distmetric = 'l1', seed = 7, n_sg_min = None):
    
    pandarallel.initialize()
    #print(seed)
    #print(distmetric)
    np.random.seed(seed)
    #df = df[~df.sgRNA.isin(todrop)]
    df = df.reset_index()
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    
    #X = np.array(df[features])
    #if (len(X.shape)) == 1:
    #    X = X[:,np.newaxis]
    #print('start graph calc')
    #d_graph_local = pynndescent.NNDescent(
    #        X, n_neighbors=n_neighbors_local, metric=distmetric, random_state = seed, n_jobs=-1
     #   ).neighbor_graph[0] ###
    
    #print('kNN graph calculated')
    
    def mean_intra_inter_jaccard_pynn(x, intersize = 10):
        inds = x.index.tolist()
        dense = np.zeros((len(inds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
        intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
        allinterinds = [i for i in df.index.tolist() if i not in inds]
        interinds = np.random.choice(allinterinds, size = intersize, replace = False)
        dense = np.zeros((len(interinds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
        interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
        x['mean_intra_jaccard_local'] = np.mean(intradists)
        x['mean_inter_jaccard_local'] = np.mean(interdists)
        return x
    
    sizes = df.groupby(['gene_symbol']).size()
    if n_sg_min != None:
      fewsg = sizes[sizes < n_sg_min].index
      print('dropping %s genes with fewer than %s sgRNA'%(len(fewsg),n_sg_min))
    else:
      fewsg = sizes[sizes < 2].index
      print('dropping %s genes with only 1 sgRNA'%len(fewsg))
   
    df = df[~df.gene_symbol.isin(fewsg)] # remove genes 
    print(df.shape)

    print('start graph calc')
    X = np.array(df[features])
    if (len(X.shape)) == 1:
        X = X[:,np.newaxis]

    d_graph_local = pynndescent.NNDescent(
            X, n_neighbors=n_neighbors_local, metric=distmetric, random_state = seed, n_jobs=-1
        ).neighbor_graph[0] ###
    
    print('kNN graph calculated')
    
    grped = df.groupby(['gene_symbol'])
    grped = grped.parallel_apply(mean_intra_inter_jaccard_pynn)
    grped = grped.groupby('gene_symbol').head(1)
    
    grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
                str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')
    return grped


def get_jaccard_similarities_fromknn_withctrl(df, ctrl, features, saveloc, savename,
                                     compute_l1 = True, n_neighbors_local = 50, distmetric = 'l1', seed = 7, n_sg_min = None, d_graph_local = None):
    
    pandarallel.initialize(nb_workers=20, verbose = 0)
    #print(seed)
    #print(distmetric)
    np.random.seed(seed)
    #df = df[~df.sgRNA.isin(todrop)]
    df = df.reset_index()
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    

    sizes = df.groupby(['gene_symbol']).size()
    if n_sg_min != None:
      fewsg = sizes[sizes < n_sg_min].index
      print('dropping %s genes with fewer than %s sgRNA'%(len(fewsg),n_sg_min))
    else:
      fewsg = sizes[sizes < 2].index
      print('dropping %s genes with only 1 sgRNA'%len(fewsg))
   
    df = df[~df.gene_symbol.isin(fewsg)] # remove genes 
    print(df.shape)


    if d_graph_local is None:
       print('calculating kNN graph')
       X = np.array(df[features])
       if (len(X.shape)) == 1:
           X = X[:,np.newaxis]
    #print('start graph calc')
       d_graph_local = pynndescent.NNDescent(
               X, n_neighbors=n_neighbors_local, metric=distmetric, random_state = seed, n_jobs=-1
           ).neighbor_graph[0] ###
    
    #print('kNN graph calculated')
    
    def mean_intra_inter_jaccard_pynn(x):
        inds = x.index.tolist()
        intersize = x.shape[0]
        #print(intersize)
        dense = np.zeros((len(inds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
        intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
        allinterinds = [i for i in df.index.tolist() if i not in inds]
        interinds = np.random.choice(allinterinds, size = intersize, replace = False)
        dense = np.zeros((len(interinds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
        interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
        x['mean_intra_jaccard_local'] = np.mean(intradists)
        x['mean_inter_jaccard_local'] = np.mean(interdists)

        if compute_l1 == True:             
                 intradists = (pdist(x[features], metric = 'cityblock'))
                 x2 = df[df.index.isin(interinds)]
                 interdists =  (pdist(x2[features], metric = 'cityblock')) 
                 x['mean_intra_l1_local'] = np.mean(intradists)
                 x['mean_inter_l1_local'] = np.mean(interdists)

        return x
    
    #sizes = df.groupby(['gene_symbol']).size()
    #fewsg = sizes[sizes < 2].index
    #print('dropping %s genes with only 1 sgRNA'%len(fewsg))

    #df = df[~df.gene_symbol.isin(fewsg)] # remove genes 
    #print(df.shape)
    df = df.reset_index()
    if compute_l1 == False:
         df.drop(features, axis = 1, inplace = True)
         print(df.shape)
    grped = df.groupby(['gene_symbol'])
    grped = grped.parallel_apply(mean_intra_inter_jaccard_pynn)
    grped = grped.groupby('gene_symbol').head(1)
    #print('first jaccard done, calculating jaccard among controls')
    dfc = df[df.gene_symbol.isin(ctrl)]
    inds = dfc.index.tolist()
    dense = np.zeros((len(inds),d_graph_local.shape[0]))
    dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
    intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
    allinterinds = [i for i in df.index.tolist() if i not in inds]
    intersize = dfc.shape[0]
    interinds = np.random.choice(allinterinds, size = intersize, replace = False)
    dense = np.zeros((len(interinds),d_graph_local.shape[0]))
    dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
    interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
    #grped.drop(features, axis = 1, inplace = True)
    #grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
    #            str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')
    dftmp = pd.DataFrame([intradists,interdists]).T
    dftmp.columns = ['intra_jaccard_local','inter_jaccard_local']
    dftmp.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
                str(n_neighbors_local) +  '_seed' + str(seed) + '_jaccard_controlsonly.csv')

    if compute_l1 == True:
         intradists = (pdist(dfc[features], metric = 'cityblock'))
         x2 = df[df.index.isin(interinds)]
         interdists = (pdist(x2[features], metric = 'cityblock')) 
         #return grped
         dftmp = pd.DataFrame([intradists,interdists]).T
         dftmp.columns = ['intra_l1_local','inter_l1_local']
         dftmp.to_csv(saveloc + savename + '_' + distmetric + 
                 '_seed' + str(seed) + '_l1controlsonly.csv')
    if compute_l1 == True:
           grped.drop(features, axis = 1, inplace = True)
    grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
                str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')


def get_jaccard_similarities_fromknn_withctrl_pearson(df, ctrl, features, saveloc, savename,
                                     compute_p = True, n_neighbors_local = 50, distmetric = 'correlation', seed = 7, n_sg_min = None, d_graph_local = None):
    
    pandarallel.initialize(nb_workers=20, verbose = 0)
    #print(seed)
    #print(distmetric)
    np.random.seed(seed)
    #df = df[~df.sgRNA.isin(todrop)]
    df = df.reset_index()
    df.loc[df.gene_symbol.str.contains('nontargeting'),'gene_symbol'] = 'nontargeting'
    

    sizes = df.groupby(['gene_symbol']).size()
    if n_sg_min != None:
      fewsg = sizes[sizes < n_sg_min].index
      print('dropping %s genes with fewer than %s sgRNA'%(len(fewsg),n_sg_min))
    else:
      fewsg = sizes[sizes < 2].index
      print('dropping %s genes with only 1 sgRNA'%len(fewsg))
   
    df = df[~df.gene_symbol.isin(fewsg)] # remove genes 
    print(df.shape)


    if d_graph_local is None:
       print('calculating kNN graph')
       X = np.array(df[features])
       if (len(X.shape)) == 1:
           X = X[:,np.newaxis]
    #print('start graph calc')
       d_graph_local = pynndescent.NNDescent(
               X, n_neighbors=n_neighbors_local, metric=distmetric, random_state = seed, n_jobs=-1
           ).neighbor_graph[0] ###

    def mean_intra_inter_jaccard_pynn(x):
        inds = x.index.tolist()
        intersize = x.shape[0]
        #print(intersize)
        dense = np.zeros((len(inds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
        intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
        allinterinds = [i for i in df.index.tolist() if i not in inds]
        interinds = np.random.choice(allinterinds, size = intersize, replace = False)
        dense = np.zeros((len(interinds),d_graph_local.shape[0]))
        dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
        interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
        x['mean_intra_jaccard_local'] = np.mean(intradists)
        x['mean_inter_jaccard_local'] = np.mean(interdists)

        if compute_p == True:             
                 intradists = (pdist(x[features], metric = 'correlation'))
                 x2 = df[df.index.isin(interinds)]
                 interdists =  (pdist(x2[features], metric = 'correlation')) 
                 x['mean_intra_p_local'] = np.mean(intradists)
                 x['mean_inter_p_local'] = np.mean(interdists)

        return x
    
    #sizes = df.groupby(['gene_symbol']).size()
    #fewsg = sizes[sizes < 2].index
    #print('dropping %s genes with only 1 sgRNA'%len(fewsg))

    #df = df[~df.gene_symbol.isin(fewsg)] # remove genes 
    #print(df.shape)
    df = df.reset_index()
    if compute_p == False:
         df.drop(features, axis = 1, inplace = True)
         print(df.shape)
    grped = df.groupby(['gene_symbol'])
    grped = grped.parallel_apply(mean_intra_inter_jaccard_pynn)
    grped = grped.groupby('gene_symbol').head(1)
    #print('first jaccard done, calculating jaccard among controls')
    dfc = df[df.gene_symbol.isin(ctrl)]
    inds = dfc.index.tolist()
    dense = np.zeros((len(inds),d_graph_local.shape[0]))
    dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
    intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
    allinterinds = [i for i in df.index.tolist() if i not in inds]
    intersize = dfc.shape[0]
    interinds = np.random.choice(allinterinds, size = intersize, replace = False)
    #dense = np.zeros((len(interinds),d_graph_local.shape[0]))
    #dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
    #intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
    #allinterinds = [i for i in df.index.tolist() if i not in inds]
    #intersize = dfc.shape[0]
    #interinds = np.random.choice(allinterinds, size = intersize, replace = False)
    dense = np.zeros((len(interinds),d_graph_local.shape[0]))
    dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
    interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
    #grped.drop(features, axis = 1, inplace = True)
    #grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
    #            str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')
    dftmp = pd.DataFrame([intradists,interdists]).T
    dftmp.columns = ['intra_jaccard_local','inter_jaccard_local']
    dftmp.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
                str(n_neighbors_local) +  '_seed' + str(seed) + '_jaccard_controlsonlypearson.csv')

    if compute_p == True:
         intradists = (pdist(dfc[features], metric = 'correlation'))
         x2 = df[df.index.isin(interinds)]
         interdists = (pdist(x2[features], metric = 'correlation')) 
         #return grped
         dftmp = pd.DataFrame([intradists,interdists]).T
         dftmp.columns = ['intra_p_local','inter_p_local']
         dftmp.to_csv(saveloc + savename + '_' + distmetric + 
                 '_seed' + str(seed) + '_pearsoncontrolsonly.csv')
    if compute_p == True:
           grped.drop(features, axis = 1, inplace = True)
    grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
                str(n_neighbors_local) +  '_seed' + str(seed) + 'pearson.csv')
    #dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[inds]] = 1
    #intradists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity within gene
    #allinterinds = [i for i in df.index.tolist() if i not in inds]
    #intersize = dfc.shape[0]
    #interinds = np.random.choice(allinterinds, size = intersize, replace = False)
    #dense = np.zeros((len(interinds),d_graph_local.shape[0]))
    #dense[np.expand_dims(np.arange(dense.shape[0]), -1), d_graph_local[interinds]] = 1 
    #interdists = (1-pdist(dense, metric = 'jaccard')) ## jaccard similarity between genes
    #grped.drop(features, axis = 1, inplace = True)
    #grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
    #            str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')
    #dftmp = pd.DataFrame([intradists,interdists]).T
    #dftmp.columns = ['intra_jaccard_local','inter_jaccard_local']
    #dftmp.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
     #           str(n_neighbors_local) +  '_seed' + str(seed) + '_jaccard_controlsonly.csv')

    #if compute_l1 == True:
    #     intradists = (pdist(dfc[features], metric = 'cityblock'))
    #     x2 = df[df.index.isin(interinds)]
    #     interdists = (pdist(x2[features], metric = 'cityblock')) 
         #return grped
    #     dftmp = pd.DataFrame([intradists,interdists]).T
    #     dftmp.columns = ['intra_l1_local','inter_l1_local']
    #     dftmp.to_csv(saveloc + savename + '_' + distmetric + 
     #            '_seed' + str(seed) + '_l1controlsonly.csv')
    #if compute_l1 == True:
    #       grped.drop(features, axis = 1, inplace = True)
    #grped.to_csv(saveloc + savename + '_' + distmetric + '_neighbors' + 
     #           str(n_neighbors_local) +  '_seed' + str(seed) + '.csv')

