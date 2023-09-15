import torch
import sys

from loader import load_llama2, save_llama2

import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
model_path = sys.argv[1]
lora_weights = sys.argv[2]
out_model_path = sys.argv[3]
shards = int(sys.argv[4]) if len(sys.argv) > 4 else 1

model = load_llama2(sys.argv[1], dropout=0.0)

model.load_state_dict(torch.load(lora_weights), strict=False)

save_llama2(model, out_model_path, model_path, shards=shards)