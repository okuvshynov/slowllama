import torch
import os
import resource

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

global_id_auto = 0

def next_id():
    global global_id_auto
    res = torch.tensor(global_id_auto)
    global_id_auto += 1
    return res

def intermediate_path(id):
    if torch.is_tensor(id):
        id = id.item()
    folder = f'{os.path.dirname(__file__)}/data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return f'{folder}/saved_{id}.pt'

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
    
def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
