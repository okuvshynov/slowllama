import logging
import torch
import sys
import os

from loader import load_frozen
from utils import Tokenizer, greedy_gen

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
model_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

model = load_frozen(sys.argv[1], dropout=0.0).to(device)

prompt = 'slowllama is a '

greedy_gen(model, tokenizer, device, prompt, max_new_tokens=30)