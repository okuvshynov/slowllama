import torch
from conf import *

adamw_eps = 1e-4
compute_dtype = torch.float16
frozen_dtype = torch.float16

frozen_model_path = '../llama13b_f16'
#frozen_model_path = '../llama13b_f16-out'
