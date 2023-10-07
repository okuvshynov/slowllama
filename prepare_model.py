# loads model in original llama2 format and saves to another folder in sequential format

import torch
import logging

from loader import prepare_model

seed = 54321
device = 'mps' # mps for macbooks
offload_to = 'disk'
lora_rank = 4

compute_dtype = torch.float32

llama2_model_path = '../llama-2-7b'
served_model_path = '../llama7b/'

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='logs/prepare_model.log')
torch.random.manual_seed(seed)

prepare_model(llama2_path=llama2_model_path, sequential_path=served_model_path, compute_dtype=compute_dtype, offload_location=offload_to, lora_rank=lora_rank).to(device).to(compute_dtype)
