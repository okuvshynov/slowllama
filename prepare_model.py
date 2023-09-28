# loads model in original llama2 format and saves to another folder in sequential format

import torch
import logging

from loader import load_llama2

seed = 54321
device = 'mps' # mps for macbooks
offload_to = 'disk'

compute_dtype = torch.float32

llama2_model_path = '/Volumes/LLAMAS/llama-2-70b'
served_model_path = '../llama70b/'

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='prepare_model.log')
torch.random.manual_seed(seed)

model = load_llama2(llama2_model_path, compute_dtype=compute_dtype, offload_location=offload_to, served_model_path=served_model_path).to(device).to(compute_dtype)
