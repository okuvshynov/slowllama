import logging
import torch
import sys
import os

from llama2_loader import load_frozen
from utils import Tokenizer, greedy_gen2
from conf_fp16 import *

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARN)

lora_weights = sys.argv[1] if len(sys.argv) > 1 else None

tokenizer_path = os.path.join(frozen_model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

model = load_frozen(frozen_model_path, dropout=0.0, lora_rank=4, frozen_dtype=frozen_dtype, compute_dtype=compute_dtype).to(device)
if lora_weights is not None:
    logging.debug(model.load_state_dict(torch.load(lora_weights), strict=False))

print(f'Model {frozen_model_path} loaded')

while True:
    prompt = input("> ")
    while True:
        for next in greedy_gen2(model, tokenizer, device, prompt, max_new_tokens=100):
            sys.stdout.write(next)
            sys.stdout.flush()

