from copy import deepcopy
import os

import torch

from utils import device_map, next_id, device_supports_dtype

# a wrapper around arbitrary module which can save/load inner model to hard drive
# we store base weights always as bfloat16 (that's what llama2 uses)
# but we need to load and return it as a type we use for computation.
# it gets a little more tricky for MPS device because we cannot load bfloat16 there 
# directly.
class BlackboxDisk(torch.nn.Module):
    def __init__(self, module, args):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
        self.compute_dtype = args.compute_dtype
        self.served_model_path = args.served_model_path
        self.cached_data_path = args.cached_data_path
        if args.init_frozen:
            torch.save(module.to('cpu').to(torch.bfloat16), self.frozen_path())

    def frozen_path(self):
        folder = os.path.join(self.served_model_path, 'frozen')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, f'block_{self.module_id}.pt')
    
    def input_path(self):
        folder = os.path.join(self.cached_data_path, 'inputs')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return f'{folder}/saved_{self.input_id}.pt'

    def loaded_inner(self):
        return torch.load(self.frozen_path(), map_location='cpu')
    
    def load(self, device):
        if device_supports_dtype(device, torch.bfloat16):
            return torch.load(self.frozen_path(), map_location=device_map(device)).to(self.compute_dtype)
        else:
            # for MPS we need to load to CPU first
            res = torch.load(self.frozen_path(), map_location='cpu')
            return res.to(self.compute_dtype).to(device_map(device))

    def save(self, module):
        torch.save(module.to('cpu').to(torch.bfloat16), self.frozen_path())
    
    def load_input(self, device):
        return torch.load(self.input_path(), map_location=torch.device(device_map(device)))

    def forward(self, input, *args):
        torch.save(input, self.input_path())
        device = device_map(input.device)
        module = self.load(device)

        if not self.training:
            module.eval()
        
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)

# same blackbox, but RAM only.
# do we really need this? with enough RAM we can just offload with fairscale?
class BlackboxRAM(torch.nn.Module):
    def __init__(self, module, args):
        super().__init__()
        # this way torch doesn't count it as a submodule, which is 
        # exactly what we want.
        self.module = [module.to('cpu').to(torch.bfloat16)]
        self.compute_dtype = args.compute_dtype

    def loaded_inner(self):
        return self.module[0]
    
    def load(self, device):
        return deepcopy(self.module[0]).to(device_map(device)).to(self.compute_dtype)

    def save(self, module):
        self.module = [module.to('cpu').to(torch.bfloat16)]
    
    def load_input(self, device):
        return self.input.to(device_map(device))

    def forward(self, input, *args):
        self.input = input.to('cpu').detach().clone()
        device = device_map(input.device)
        module = self.load(device)

        if not self.training:
            module.eval()
        
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)

def wrap_blackbox(module, args):
    if args.offload_location == 'ram':
        return BlackboxRAM(module, args)
    else:
        return BlackboxDisk(module, args)