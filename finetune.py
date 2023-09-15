import os
import sys
import torch
import logging

from loader import load_llama2, save_llama2
from plot_lora import log_lora

# use tokenizer from llama
sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

# training settings
seed = 54321
iters = 1000
device = 'mps' # mps for macbooks
seq_len = 128
batch_size = 16
lr = 1e-4
offload_to = 'disk'

# type used for computation. Might be different from storage type (which is bfloat16)
compute_dtype = torch.float32 # float32 for macbooks
#compute_dtype = torch.bfloat16 # bfloat16 for CUDA

eval_period = 10
gen_tokens = 32

log_lora_grad = False
log_lora_weight = True

model_path = '../llama-2-7b'
finetune_file = './README.md'
prompt = 'slowllama is a '

# data to finetune on
with open(finetune_file) as f:
    text = f.read()

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

def greedy_gen(prompt, iter, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        logits_top, next_tokens = torch.topk(logits, k=25, dim=-1)
        next_token = next_tokens[0, 0].view(1, 1)
        logging.info(f'next token: {next_token}')
        #logging.info(f'next tokens: {logits_top} {next_tokens} {tokenizer.decode(next_tokens.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for output in tokens:
        logging.info(f'after {iter} iterations: {tokenizer.decode(output.tolist())}')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='finetune.log')
    torch.random.manual_seed(seed)

    tokenizer_path = os.path.join(model_path, 'tokenizer.model')

    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text, True, True)

    logging.info(f'loaded dataset: {len(tokens)} tokens')

    model = load_llama2(model_path, compute_dtype=compute_dtype, offload_location=offload_to).to(device).to(compute_dtype)

    def get_batch(batch_size):
        index = torch.randint(len(tokens) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(tokens[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(tokens[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    last_loss = None
    for i in range(iters):
        logging.info(f'starting iteration {i}')
        X, y = get_batch(batch_size)
        opt.zero_grad()
        if i % eval_period == 0:
            greedy_gen(prompt, i, max_new_tokens=gen_tokens)
        # both forward and backward passes are here.
        # returned loss is a scalar, not variable
        logits, loss = model.manual_loop(X, y)
        opt.step()

        # optional logging of lora weights/gradients
        if log_lora_grad or log_lora_weight:
            log_lora(model.lora_layers, log_weights=log_lora_weight, log_grad=log_lora_grad)

        logging.info(f'backprop done, loss after forward pass = {loss}')
        if last_loss is None:
            last_loss = loss
        elif loss < last_loss:
            last_loss = loss
            logging.info(f'saving snapshot')
            torch.save(model.state_dict(), f'data/state_dict_{i}.pth')
