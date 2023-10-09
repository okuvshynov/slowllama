import os
import torch
import logging

from loader import load_frozen
from plot_lora import log_lora
from utils import Tokenizer, greedy_gen

# training settings
seed = 54321
iters = 20
device = 'mps' # mps for macbooks
seq_len = 128
batch_size = 16
lr = 1e-4
adamw_eps = 1e-4 # need to change as 1e-8 doesn't fit to float16
offload_to = 'disk'

# type used for computation. Might be different from storage type (which is bfloat16)
#compute_dtype = torch.float32 # float32 for macbooks
#compute_dtype = torch.bfloat16 # bfloat16 for CUDA
compute_dtype = torch.float16
frozen_dtype = torch.float16

eval_before_training = False
eval_period = 20
gen_tokens = 32

log_lora_grad = False
log_lora_weight = True

model_path = '../llama7b_f16'
snapshots_path = 'out'
finetune_file = './test_data/cubestat.txt'
prompt = 'Cubestat reports the following metrics: '

lora_rank = 4

log_level = logging.DEBUG

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=log_level, filename='logs/finetune.log')
    torch.random.manual_seed(seed)

    if not os.path.exists(snapshots_path):
        os.makedirs(snapshots_path)

    # data to finetune on
    with open(finetune_file) as f:
        text = f.read()

    tokenizer = Tokenizer(os.path.join(model_path, 'tokenizer.model'))
    tokens = tokenizer.encode(text, True, True)

    logging.info(f'loaded dataset: {len(tokens)} tokens')

    model = load_frozen(model_path, compute_dtype=compute_dtype, lora_rank=lora_rank, frozen_dtype=frozen_dtype).to(device).to(compute_dtype)

    def get_batch(batch_size):
        index = torch.randint(len(tokens) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(tokens[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(tokens[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, eps=adamw_eps)

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
        log_lora(model.lora_layers, log_weights=log_lora_weight, log_grad=log_lora_grad)

        logging.info(f'backprop done, loss after forward pass = {loss}')
        if last_loss is None:
            last_loss = loss
        elif loss < last_loss:
            last_loss = loss
            logging.info(f'saving snapshot')
            torch.save(model.state_dict(), os.path.join(snapshots_path, f'state_dict_{i}.pth'))
