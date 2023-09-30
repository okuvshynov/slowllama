import logging
import torch
import sys
import os

from loader import load_frozen
from utils import Tokenizer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
model_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

model = load_frozen(sys.argv[1], dropout=0.0).to(device)

def greedy_gen(prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        print(f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        print(f'{i} - {tokenizer.decode(output.tolist())}')

prompt = 'slowllama is a '

greedy_gen(prompt, max_new_tokens=30)