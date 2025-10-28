import pandas as pd
import scanpy as sc
import numpy as np
from anndata import AnnData
import CytoBridge
df = pd.read_csv('/home/sjt/workspace2/Downstream_analysis/DeepRUOTv2-main/data/Weinreb_alltime.csv')

obs = pd.DataFrame(index=df.index)
obs['samples'] = df['samples'].values

X = df.drop(columns=['samples']).values

adata = AnnData(X=X, obs=obs)
adata.write_h5ad('/home/sjt/workspace2/CytoBridge_test-main_crufm/data/Weinreb.h5ad')


adata = CytoBridge.pp.preprocess(
    adata,
    time_key='samples',
    dim_reduction='none',
    normalization=False,
    log1p=False,
    select_hvg=False
)
CytoBridge.tl.fit(adata, config = 'ruot', device = 'cuda')
