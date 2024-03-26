import torch
import sentencepiece
import logging

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

def device_supports_dtype(device, dtype):
    try:
        a = torch.rand(2, 2).to(device).to(dtype)
        b = torch.rand(2, 2).to(device).to(dtype)
        c = a.mm(b)
        logging.debug(f'success, {device} supports {dtype}')
        return True
    except TypeError as e:
        return False

global_id_auto = 0

def next_id():
    global global_id_auto
    new_id = global_id_auto
    global_id_auto += 1
    return new_id

def save_rng_state(device='cpu'):
    if device == 'cpu':
        import torch
        return torch.random.get_rng_state()
    elif device.startswith('cuda'):
        import torch.cuda
        return torch.cuda.get_rng_state(device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        return torch.mps.get_rng_state()
    else:
        raise ValueError(f"Unsupported device: {device}")

def restore_rng_state(rng_state, device='cpu'):
    if device == 'cpu':
        import torch
        torch.random.set_rng_state(rng_state)
    elif device.startswith('cuda'):
        import torch.cuda
        torch.cuda.set_rng_state(rng_state, device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        torch.mps.set_rng_state(rng_state)
    else:
        raise ValueError(f"Unsupported device: {device}")
    
def greedy_gen(model, tokenizer, device, prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        logging.info(f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        logging.info(f'{i} - {tokenizer.decode(output.tolist())}')

def greedy_gen2(model, tokenizer, device, prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        logging.info(f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        yield tokenizer.decode(next_token.tolist())[0]
        tokens = torch.cat((tokens, next_token), dim=1)

def cleanup_cache(device='cpu'):
    if device.startswith('mps'):
        import torch.mps
        torch.mps.empty_cache()

    
class Tokenizer:
    def __init__(self, path):
        self.model = sentencepiece.SentencePieceProcessor(path)

    def encode(self, text, bos=False, eos=False):
        b = [self.model.bos_id()] if bos else []
        e = [self.model.eos_id()] if eos else []
        return b + self.model.encode(text) + e

    def decode(self, tokens):
        return self.model.decode(tokens)
