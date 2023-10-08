import logging
import torch
import sys
import os

from loader import load_frozen
from utils import Tokenizer, greedy_gen

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
model_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
lora_weights = sys.argv[3] if len(sys.argv) > 3 else None

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

model = load_frozen(sys.argv[1], dropout=0.0, lora_rank=4).to(device)
if lora_weights is not None:
    logging.debug(model.load_state_dict(torch.load(lora_weights), strict=False))

prompt = 'Cubestat reports the following metrics: '

greedy_gen(model, tokenizer, device, prompt, max_new_tokens=30)
