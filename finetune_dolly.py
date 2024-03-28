import os
import sys
import torch
import logging

from llama2_loader import load_frozen
from plot_lora import log_lora
from datasets import load_dataset
from utils import Tokenizer, greedy_gen

# training settings
seed = 54321
iters = 1000
device = 'mps' # mps for macbooks
seq_len = 1024
batch_size = 4
lr = 1e-4

# type used for computation. Might be different from storage type (which is bfloat16)
compute_dtype = torch.float32 # float32 for macbooks
#compute_dtype = torch.bfloat16 # bfloat16 for CUDA

eval_before_training = False
eval_period = 20
gen_tokens = 32

log_lora_grad = False
log_lora_weight = True

model_path = '../llama7b'
snapshots_path = 'out'
finetune_dataset = 'databricks/databricks-dolly-15k'
prompt = 'slowllama is a '

if not os.path.exists(snapshots_path):
    os.makedirs(snapshots_path)

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

def format_sample(sample):
    instruction = f"### Instruction\n{sample['instruction']}\n\n"
    context = f"### Context\n{sample['context']}\n\n" if len(sample["context"]) > 0 else ""
    response = f"### Answer\n{sample['response']}"
    return instruction + context + response

def prepare_data():
    train_data = load_dataset(finetune_dataset, split="train")
    formatted = [format_sample(s) for s in train_data]
    return '\n\n'.join(formatted[:100])

if __name__ == '__main__':
    text = prepare_data()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='logs/finetune.log')
    torch.random.manual_seed(seed)

    tokens = tokenizer.encode(text, True, True)

    logging.info(f'loaded dataset: {len(tokens)} tokens')

    model = load_frozen(model_path, compute_dtype=compute_dtype).to(device).to(compute_dtype)

    def get_batch(batch_size):
        index = torch.randint(len(tokens) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(tokens[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(tokens[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    last_loss = None
    for i in range(iters):
        if i % eval_period == 0 and (i > 0 or eval_before_training):
            greedy_gen(model, tokenizer, device, prompt, gen_tokens)
        logging.info(f'starting iteration {i}')
        X, y = get_batch(batch_size)
        opt.zero_grad()
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
            torch.save(model.state_dict(), os.path.join(snapshots_path, f'state_dict_{i}.pth'))
