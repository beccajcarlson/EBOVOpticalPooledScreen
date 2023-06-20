# imports

from dask import compute, delayed
from dask.distributed import Client
import math
import sys
import mahotas
from ops.imports_ipython import *
import skimage
from nd2reader import ND2Reader
import ops.triangle_hash
import tables
import random
from dask.diagnostics import ProgressBar




grid_10x = [5,9,13,15,17,17,19,19,21,21,21,21,21,19,19,17,17,15,13,9,5]
arr10x = np.zeros((max(grid_10x),max(grid_10x)))


for i in range(arr10x.shape[0]):
    middle = int(len(arr10x[i])/2) ## only works if odd
    num_tiles = grid_10x[i]
    width = math.trunc(num_tiles/2)
    arr10x[i][middle-width:middle+width+1] = 1
    
fillrows = np.nonzero(arr10x)[0]
fillcols = np.nonzero(arr10x)[1]
arr10x[arr10x == 0] = -1
ct = 0
for i in (np.unique(fillrows)):
    nonzeroinds = fillcols[np.argwhere(fillrows==i)]
    if i % 2 == 0: #row is even, grid goes right
        for j in range(len(nonzeroinds)): #col indices
            arr10x[i,nonzeroinds[j]] = ct
            ct += 1
    else:
        for j in reversed(range(len(nonzeroinds))): #col indices
            arr10x[i,nonzeroinds[j]] = ct
            ct += 1

grid_20x = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]
arr20x = np.zeros((max(grid_20x),max(grid_20x)))


for i in range(arr20x.shape[0]):
    middle = int(len(arr20x[i])/2) ## only works if odd
    num_tiles = grid_20x[i]
    width = math.trunc(num_tiles/2)
    arr20x[i][middle-width:middle+width+1] = 1
    
fillrows = np.nonzero(arr20x)[0]
fillcols = np.nonzero(arr20x)[1]
arr20x[arr20x == 0] = -1
ct = 0
for i in (np.unique(fillrows)):
    nonzeroinds = fillcols[np.argwhere(fillrows==i)]
    if i % 2 == 0: #row is even, grid goes right
        for j in range(len(nonzeroinds)): #col indices
            arr20x[i,nonzeroinds[j]] = ct
            ct += 1
    else:
        for j in reversed(range(len(nonzeroinds))): #col indices
            arr20x[i,nonzeroinds[j]] = ct
            ct += 1

shape = 1480
overlap = .15
pheno_sbs_mag_foldchange = 2
arrwidth10x = arr10x.shape[0]*shape*(1-overlap) + shape*overlap
arrwidth20x = arr20x.shape[0]*shape*(1-overlap) + shape*overlap
offset = -(arrwidth20x/2-arrwidth10x)/2
dists10x_i = [((np.argwhere(arr10x == site))[0][1]*shape*(1-overlap)) for site in range(333)]
dists10x_j = [(arrwidth10x-(np.argwhere(arr10x == site))[0][0]*shape*(1-overlap))-shape for site in range(333)]

dists20x_i = [((np.argwhere(arr20x == site))[0][1]*shape*(1-overlap))/pheno_sbs_mag_foldchange + offset for site in range(1281)]
dists20x_j = [((arrwidth20x-(np.argwhere(arr20x == site))[0][0]*shape*(1-overlap))-shape)/pheno_sbs_mag_foldchange + offset for site in range(1281)]

df_ph_xy = pd.DataFrame({'x': [x + 740 for x in dists20x_i], 'y': [x + 740 for x in dists20x_j]})
df_sbs_xy = pd.DataFrame({'x': [x + 740 for x in dists10x_i], 'y': [x + 740 for x in dists10x_j]})



#grid_20x = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,#21,17,13,7]
#arr20x = np.zeros((max(grid_20x),max(grid_20x)))
grid_40x = [4,18,24,30,34,38,42,44,48,50,52,56,58,60,60,62,64,66,66,68,70,70,72,72,74,74,76,76,76,78,78,78,80,80,80,80,80,80,80,82,82,82,82,80,80,
 80,80,80,80,80,78,78,78,76,76,76,74,74,72,72,70,70,68,66,66,64,62,60,60,58,56,52,50,48,44,42,38,34, 30,24, 18, 4]

arr40x = np.zeros((max(grid_40x),max(grid_40x)))


for i in range(arr40x.shape[0]):
    middle = int(len(arr40x[i])/2) ## only works if odd
    num_tiles = grid_40x[i]
    width = math.trunc(num_tiles/2)
    arr40x[i][middle-width:middle+width+1] = 1
    
fillrows = np.nonzero(arr40x)[0]
fillcols = np.nonzero(arr40x)[1]
arr40x[arr40x == 0] = -1
ct = 0
for i in (np.unique(fillrows)):
    nonzeroinds = fillcols[np.argwhere(fillrows==i)]
    if i % 2 == 0: #row is even, grid goes right
        for j in range(len(nonzeroinds)): #col indices
            arr40x[i,nonzeroinds[j]] = ct
            ct += 1
    else:
        for j in reversed(range(len(nonzeroinds))): #col indices
            arr40x[i,nonzeroinds[j]] = ct
            ct += 1


shape = 1480
overlap = .15
pheno_sbs_mag_foldchange = 4
#arrwidth10x = arr10x.shape[0]*shape*(1-overlap) + shape*overlap
arrwidth40x = arr40x.shape[0]*shape*(1-overlap) + shape*overlap
offset = -(arrwidth40x/2-arrwidth40x)/2
#dists10x_i = [((np.argwhere(arr40x == site))[0][1]*shape*(1-overlap)) for site in range(333)]
#dists10x_j = [(arrwidth10x-(np.argwhere(arr10x == site))[0][0]*shape*(1-overlap))-shape for site in range(333)]

dists40x_i = [((np.argwhere(arr40x == site))[0][1]*shape*(1-overlap))/pheno_sbs_mag_foldchange + offset for site in range(5124)]
dists40x_j = [((arrwidth40x-(np.argwhere(arr40x == site))[0][0]*shape*(1-overlap))-shape)/pheno_sbs_mag_foldchange + offset for site in range(5124)]

df_ph_xy40x = pd.DataFrame({'x': [x + 740 for x in dists40x_i], 'y': [x + 740 for x in dists40x_j]})





##
def prep_sbs_ph_info(sbs_file_loc,ph_file_loc):

        # load all sbs cell coordinates
        @delayed
        def read_csv_sbs(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])                 
            return df

        files = glob(sbs_file_loc)
        with ProgressBar():
                df_sbs_info = pd.concat(compute(*map(read_csv_sbs, files), scheduler='threads'))

        # load pheno dfs
        @delayed
        def read_csv_pheno(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
            return df

        files = glob(ph_file_loc)
        with ProgressBar():
            df_ph_info = pd.concat(compute(*map(read_csv_pheno, files), scheduler='threads'))

        df_sbs_info['i_og'] = df_sbs_info['i']
        df_sbs_info['j_og'] = df_sbs_info['j']
        df_ph_info['i_og'] = df_ph_info['i']
        df_ph_info['j_og'] = df_ph_info['j']

        df_sbs_info['i'] = 1480-df_sbs_info['i']
        df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})

        df_sbs_info = pd.merge(df_sbs_info,df_sbs_xy,left_on = 'tile', right_index=True)
        df_sbs_info['i'] = df_sbs_info['i'] + df_sbs_info['x']
        df_sbs_info['j'] = df_sbs_info['j'] + df_sbs_info['y']

        df_ph_info['i'] = df_ph_info['i']/2
        df_ph_info['j']=1480/2-df_ph_info['j']/2

        df_ph_info = pd.merge(df_ph_info,df_ph_xy,left_on = 'tile', right_index=True)
        df_ph_info['i'] = df_ph_info['i'] + df_ph_info['x']
        df_ph_info['j'] = df_ph_info['j'] + df_ph_info['y']

        return df_ph_info, df_sbs_info


def prep_ph_info(ph_file_loc):
        # load pheno dfs
        @delayed
        def read_csv_pheno(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
            df['plate'] = f.split('/GW')[1][:1]
            return df

        files = glob(ph_file_loc)
        with ProgressBar():
            df_ph_info = pd.concat(compute(*map(read_csv_pheno, files), scheduler='threads'))

        df_ph_info['i'] = df_ph_info['i']/2
        df_ph_info['j']=1480/2-df_ph_info['j']/2

        df_ph_info = pd.merge(df_ph_info,df_ph_xy,left_on = 'tile', right_index=True)
        df_ph_info['i'] = df_ph_info['i'] + df_ph_info['x']
        df_ph_info['j'] = df_ph_info['j'] + df_ph_info['y']

        return df_ph_info


def prep_sbs_ph_hash(sbs_file_loc,ph_file_loc,sbs_save_loc,ph_save_loc,df_ph_xy,df_sbs_xy,bin_ph=True,flip_ph= False,pheno_mag='20x',flip_sbs=False,old_mode=True,manual_offset=None):

	# load all sbs cell coordinates
	@delayed
	def read_csv_sbs(f):
	    df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])                 
	    return df

	files = glob(sbs_file_loc)
	with ProgressBar():
		df_sbs_info = pd.concat(compute(*map(read_csv_sbs, files), scheduler='threads'))

	# load pheno dfs
	@delayed
	def read_csv_pheno(f):
           try:
              df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
              if df.shape[0] < 5:
                  print(f, " has <5 cells")
              else:
                  return df
           except pd.errors.EmptyDataError:
              print(f, " is empty")

	files = glob(ph_file_loc)
	with ProgressBar():
	    df_ph_info = pd.concat(compute(*map(read_csv_pheno, files), scheduler='threads'))

	#df_sbs_info['i'] = 1480-df_sbs_info['i']
	if flip_sbs == True:
            print('flip sbs')
            df_sbs_info['j']=1480-df_sbs_info['j']
            df_sbs_info['i']=1480-df_sbs_info['i']
            df_sbs_info = df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
            df_sbs_info['i'] = 1480 - df_sbs_info['i']

	if old_mode == True:
		df_sbs_info['i'] = 1480-df_sbs_info['i']
		df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
        #df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
	df_sbs_info = pd.merge(df_sbs_info,df_sbs_xy,left_on = 'tile', right_index=True)
	df_sbs_info['i'] = df_sbs_info['i'] + df_sbs_info['x']
	df_sbs_info['j'] = df_sbs_info['j'] + df_sbs_info['y']

	if bin_ph == True:
		if flip_ph == False:
			df_ph_info['i'] = df_ph_info['i']/2
			#df_ph_info['j'] = df_ph_info['j']/2
			df_ph_info['j']=1480/2-df_ph_info['j']/2
		else:
			df_ph_info['i']=1480/2-df_ph_info['i']/2
			df_ph_info['j'] = df_ph_info['j']/2

	else:
		if flip_ph == False:
			df_ph_info['i'] = df_ph_info['i']/4
			#df_ph_info['j'] = df_ph_info['j']/4
			df_ph_info['j']=2960/4-df_ph_info['j']/4
		elif flip_ph == True:
			df_ph_info['i']=2960/4-df_ph_info['i']/4
			df_ph_info['j'] = df_ph_info['j']/4
		elif flip_ph == 'sting':
			df_ph_info['i'] = 2960/4-df_ph_info['i']/4
			df_ph_info['j']=2960/4-df_ph_info['j']/4
			df_ph_info = df_ph_info.rename(columns={'j': 'i', 'i': 'j'})
			df_ph_info['i'] = 2960-df_ph_info['i']
	#if pheno_mag == '40x':
	#	if bin_ph == True:
	#		df_ph_info['i'] = df_ph_info['i'] - 740
	#		df_ph_info['j'] = df_ph_info['j'] - 740
	#	else:
         #               df_ph_info['i'] = df_ph_info['i'] - 1480
          #              df_ph_info['j'] = df_ph_info['j'] - 1480

	df_ph_info = pd.merge(df_ph_info,df_ph_xy,left_on = 'tile', right_index=True)
	
	if pheno_mag == '40x':
		print('40x')
		df_ph_info['i'] = df_ph_info['i']*.5 + df_ph_info['x']
		df_ph_info['j'] = df_ph_info['j']*.5 + df_ph_info['y']
	else:
		df_ph_info['i'] = df_ph_info['i'] + df_ph_info['x']
		df_ph_info['j'] = df_ph_info['j'] + df_ph_info['y']


	if manual_offset != None:
		print('adding manual offset')
		df_ph_info['i'] = df_ph_info['i'] + manual_offset[0]
		df_ph_info['j'] = df_ph_info['j'] + manual_offset[1]

	df_sbs_info_hash = df_sbs_info.pipe(ops.utils.gb_apply_parallel,['well','tile'],ops.triangle_hash.find_triangles)
	print('sbs hash done')

	df_ph_info_hash = df_ph_info.pipe(ops.utils.gb_apply_parallel,['well','tile'],ops.triangle_hash.find_triangles)
	print('ph hash done')


	df_sbs_info_hash.rename(columns={'tile':'site'},inplace=True)
	#df_sbs_info.to_hdf(sbs_save_loc, key = 'x')
	#df_ph_info.to_hdf(ph_save_loc, key = 'x')
	df_sbs_info_hash.to_hdf(sbs_save_loc, key = 'x')
	df_ph_info_hash.to_hdf(ph_save_loc, key = 'x')
	return df_ph_info, df_sbs_info

def prep_sbs_ph_info_general(sbs_file_loc,ph_file_loc,df_ph_xy,df_sbs_xy,bin_ph=True,flip_ph= False,pheno_mag='20x',flip_sbs=False,old_mode=True,manual_offset=None):

        # load all sbs cell coordinates
        @delayed
        def read_csv_sbs(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])                 
            return df

        files = glob(sbs_file_loc)
        with ProgressBar():
                df_sbs_info = pd.concat(compute(*map(read_csv_sbs, files), scheduler='threads'))

        # load pheno dfs
        @delayed
        def read_csv_pheno(f):
            #df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
            #return df
           try:
              df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
              if df.shape[0] < 5:
                  print(f, " has <5 cells")
              else:
                  return df
           except pd.errors.EmptyDataError:
              print(f, " is empty")

        files = glob(ph_file_loc)
        with ProgressBar():
            df_ph_info = pd.concat(compute(*map(read_csv_pheno, files), scheduler='threads'))


        df_sbs_info['i_og'] = df_sbs_info['i']
        df_sbs_info['j_og'] = df_sbs_info['j']
        df_ph_info['i_og'] = df_ph_info['i']
        df_ph_info['j_og'] = df_ph_info['j']

        #df_sbs_info['i'] = 1480-df_sbs_info['i']
        if flip_sbs == True:
            print('flip sbs')
            df_sbs_info['j']=1480-df_sbs_info['j']
            df_sbs_info['i']=1480-df_sbs_info['i']
            df_sbs_info = df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
            df_sbs_info['i'] = 1480 - df_sbs_info['i']

        if old_mode == True:
                df_sbs_info['i'] = 1480-df_sbs_info['i']
                df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
        #df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})
        df_sbs_info = pd.merge(df_sbs_info,df_sbs_xy,left_on = 'tile', right_index=True)
        df_sbs_info['i'] = df_sbs_info['i'] + df_sbs_info['x']
        df_sbs_info['j'] = df_sbs_info['j'] + df_sbs_info['y']

        if bin_ph == True:
                if flip_ph == False:
                        df_ph_info['i'] = df_ph_info['i']/2
                        df_ph_info['j']=1480/2-df_ph_info['j']/2
                else:
                        df_ph_info['i']=1480/2-df_ph_info['i']/2
                        df_ph_info['j'] = df_ph_info['j']/2

        else:
                if flip_ph == False:
                        df_ph_info['i'] = df_ph_info['i']/4
                        df_ph_info['j']=2960/4-df_ph_info['j']/4
                elif flip_ph == True:
                        df_ph_info['i']=2960/4-df_ph_info['i']/4
                        df_ph_info['j'] = df_ph_info['j']/4
                elif flip_ph == 'sting':
                        df_ph_info['i'] = 2960/4-df_ph_info['i']/4
                        df_ph_info['j']=2960/4-df_ph_info['j']/4
                        df_ph_info = df_ph_info.rename(columns={'j': 'i', 'i': 'j'})
                        df_ph_info['i'] = 2960-df_ph_info['i']

        #if pheno_mag == '40x':
        #       if bin_ph == True:
        #               df_ph_info['i'] = df_ph_info['i'] - 740
        #               df_ph_info['j'] = df_ph_info['j'] - 740
        #       else:
         #               df_ph_info['i'] = df_ph_info['i'] - 1480
          #              df_ph_info['j'] = df_ph_info['j'] - 1480

        df_ph_info = pd.merge(df_ph_info,df_ph_xy,left_on = 'tile', right_index=True)
        
        if pheno_mag == '40x':
                print('40x')
                df_ph_info['i'] = df_ph_info['i']*.5 + df_ph_info['x']
                df_ph_info['j'] = df_ph_info['j']*.5 + df_ph_info['y']
        else:
                df_ph_info['i'] = df_ph_info['i'] + df_ph_info['x']
                df_ph_info['j'] = df_ph_info['j'] + df_ph_info['y']

        if manual_offset != None:
                print('adding manual offset')
                df_ph_info['i'] = df_ph_info['i'] + manual_offset[0]
                df_ph_info['j'] = df_ph_info['j'] + manual_offset[1]

        return df_ph_info, df_sbs_info

def prep_sbs_ph_hash_40x(sbs_file_loc,ph_file_loc,sbs_save_loc,ph_save_loc,bin_ph=False):
	### JUNK ###
        # load all sbs cell coordinates
        @delayed
        def read_csv_sbs(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])                 
            return df

        files = glob(sbs_file_loc)
        with ProgressBar():
                df_sbs_info = pd.concat(compute(*map(read_csv_sbs, files), scheduler='threads'))

        # load pheno dfs
        @delayed
        def read_csv_pheno(f):
            df = pd.read_csv(f, usecols = ['cell','tile','well','i','j'])    
            return df

        files = glob(ph_file_loc)
        with ProgressBar():
            df_ph_info = pd.concat(compute(*map(read_csv_pheno, files), scheduler='threads'))


        df_sbs_info['i'] = 1480-df_sbs_info['i']
        df_sbs_info=df_sbs_info.rename(columns={'j': 'i', 'i': 'j'})

        df_sbs_info = pd.merge(df_sbs_info,df_sbs_xy,left_on = 'tile', right_index=True)
        df_sbs_info['i'] = df_sbs_info['i'] + df_sbs_info['x']
        df_sbs_info['j'] = df_sbs_info['j'] + df_sbs_info['y']

        if bin_ph == False:
                df_ph_info['i'] = df_ph_info['i']/2
                df_ph_info['j']=1480/2-df_ph_info['j']/2
        else:
                df_ph_info['i'] = df_ph_info['i']/4
                df_ph_info['j']=2960/4-df_ph_info['j']/4

        df_ph_info = pd.merge(df_ph_info,df_ph_xy_40x,left_on = 'tile', right_index=True)
        df_ph_info['i'] = df_ph_info['i'] + df_ph_info['x']
        df_ph_info['j'] = df_ph_info['j'] + df_ph_info['y']

        df_sbs_info_hash = df_sbs_info.pipe(ops.utils.gb_apply_parallel,['well','tile'],ops.triangle_hash.find_triangles)
        print('sbs hash done')

        df_ph_info_hash = df_ph_info.pipe(ops.utils.gb_apply_parallel,['well','tile'],ops.triangle_hash.find_triangles)
        print('ph hash done')


        df_sbs_info_hash.rename(columns={'tile':'site'},inplace=True)

        df_sbs_info_hash.to_hdf(sbs_save_loc, key = 'x')
        df_ph_info_hash.to_hdf(ph_save_loc, key = 'x')


def create_init_sites(ntries,phenomag='20x'):
	#print(phenomag)
	bigarr10x = (np.repeat(np.repeat(arr10x,axis=0,repeats=4),axis=1,repeats=4))
	if phenomag == '20x':
            bigarr20x = np.pad(np.repeat(np.repeat(arr20x,axis=0,repeats=2),axis=1,repeats=2),pad_width=1,mode='constant',constant_values=-1)
	elif phenomag == '40x':
            bigarr40x = np.pad(np.repeat(np.repeat(arr40x,axis=0,repeats=1),axis=1,repeats=1),pad_width=1,mode='constant',constant_values=-1)
            #bigarr20x = np.pad(np.repeat(np.repeat(arr40x,axis=0,repeats=2),axis=1,repeats=2),pad_width=1,mode='constant',constant_values=-1)
            df_ph_xy = df_ph_xy40x 
	else:
            print('phenomag not recognized')

	initial_sites = []
	random.seed(7)
	for i in range(ntries):
            if phenomag == '20x':
               tile = random.randint(0,1281)
               site = (np.unique(bigarr10x[np.where(bigarr20x == tile)])) 
               if len(site) == 1:
                    initial_sites.append((tile,site[0]))
            if phenomag == '40x':
               tile = random.randint(0,5124)
               site = (np.unique(bigarr10x[np.where(bigarr40x == tile)])) 
               if len(site) == 1:
                    initial_sites.append((tile,site[0]))
            else:
               print('phenomag not recognized')
               #tile = random.randint(0,1281)
	       #site = (np.unique(bigarr10x[np.where(bigarr20x == tile)])) 
	       #if len(site) == 1:
	       #   initial_sites.append((tile,site[0]))

	print('actual # sites with full overlap')
	print(len(initial_sites))
	return initial_sites,df_ph_xy,df_sbs_xy
