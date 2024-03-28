# loads model in original llama2 format and saves to another folder in sequential format

import torch
import logging

from llama2_loader import prepare_model
from conf_fp16 import *

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.DEBUG, filename='logs/prepare_model.log')
torch.random.manual_seed(seed)

prepare_model(llama2_path=llama2_model_path, frozen_path=frozen_model_path, compute_dtype=compute_dtype,
              lora_rank=lora_rank, frozen_dtype=frozen_dtype)
