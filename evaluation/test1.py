import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sjt/workspace2/CytoBridge_test-main_crufm')))
import CytoBridge
import scanpy as sc
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel
import torch
import numpy as np

adata = sc.read_h5ad('/home/sjt/workspace2/CytoBridge_test-main_crufm/data/Weinreb.h5ad')


adata = CytoBridge.pp.preprocess(
    adata,
    time_key='samples',
    dim_reduction='none',
    normalization=False,
    log1p=False,
    select_hvg=False
)

adata = CytoBridge.tl.fit(adata, config='ruot', device='cuda')

