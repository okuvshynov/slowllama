import torch
from conf import *

adamw_eps = 1e-8
compute_dtype = torch.float32
frozen_dtype = torch.bfloat16

frozen_model_path = '../llama7b'