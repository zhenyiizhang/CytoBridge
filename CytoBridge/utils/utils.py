import random
import torch
import numpy as np
from anndata import AnnData
from torchdiffeq import odeint
from sklearn.decomposition import PCA
# 导入所需模块（确保路径正确）
# 正确代码（先到上一级 CytoBridge/，再找 tl/）
from ..tl.models import DynamicalModel


def sample(data, size):
    N = data.shape[0]
    indices = torch.tensor(random.sample(range(N), size))
    sampled = data[indices]
    return sampled



def load_model_from_adata(adata: AnnData) -> torch.nn.Module:
    if 'all_model' not in adata.uns or 'model_config' not in adata.uns['all_model']:
        raise ValueError("Cannot find CytoBridge model information, please fit the model first")

    print("Reconstructing model...")

    # 1. 反序列化训练配置（如果需要使用）
    import json
    training_config = adata.uns['all_model']['training_config']

    # 修复：只在plan是字符串时才进行JSON解析
    if isinstance(training_config.get('plan'), str):
        training_config['plan'] = json.loads(training_config['plan'])  # 将JSON字符串转回列表

    # 2. 重建模型（原有逻辑不变）
    model_config = adata.uns['all_model']['model_config']
    dim = adata.obsm['X_latent'].shape[1]
    model = DynamicalModel(dim, model_config)

    # 3. 加载权重（原有逻辑不变）
    state_dict_np = adata.uns['all_model']['model_state_dict']
    state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    model.load_state_dict(state_dict_torch)
    model.eval()

    print("Model loaded successfully.")
    return model



#%%
import torch
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel
import os



def save_model_to_adata(adata: AnnData, model: DynamicalModel) -> None:
    """
    将训练好的DynamicalModel保存到AnnData对象中

    参数:
        adata: 用于存储模型的AnnData对象
        model: 训练好的DynamicalModel实例
    """
    if 'dynamic_model' not in adata.uns:
        adata.uns['dynamic_model'] = {}

    # 保存模型配置
    adata.uns['dynamic_model']['model_config'] = model.config

    # 保存模型权重（转换为numpy数组以兼容h5ad格式）
    state_dict = model.state_dict()
    state_dict_cpu_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
    adata.uns['dynamic_model']['model_state_dict'] = state_dict_cpu_numpy

    print("模型已成功保存到adata.uns['dynamic_model']")


def save_model_to_file(model: DynamicalModel, save_path: str) -> None:
    """
    将模型直接保存为pth文件（推荐用于单独备份）

    参数:
        model: 训练好的DynamicalModel实例
        save_path: 保存路径（如'models/trained_model.pth'）
    """
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存模型权重和配置
    torch.save({
        'model_config': model.config,
        'state_dict': model.state_dict()
    }, save_path)

    print(f"模型已成功保存到文件: {save_path}")


def load_model_from_file(load_path: str, latent_dim: int) -> DynamicalModel:
    """
    从pth文件加载模型

    参数:
        load_path: 模型文件路径
        latent_dim: 潜在空间维度（需与训练时一致）
    返回:
        加载权重后的DynamicalModel实例
    """
    checkpoint = torch.load(load_path)
    model = DynamicalModel(latent_dim, checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"模型已从文件加载: {load_path}")
    return model