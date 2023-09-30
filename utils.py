import torch
import sentencepiece

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

def device_supports_dtype(device, dtype):
    try:
        tensor = torch.tensor([1.0, 2.0]).to(device).to(dtype)
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
    
class Tokenizer:
    def __init__(self, path):
        self.model = sentencepiece.SentencePieceProcessor(path)

    def encode(self, text, bos=False, eos=False):
        b = [self.model.bos_id()] if bos else []
        e = [self.model.eos_id()] if eos else []
        return b + self.model.encode(text) + e

    def decode(self, tokens):
        return self.model.decode(tokens)