from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from utils import intermediate_path, device_map, next_id, device_supports_dtype

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
        torch.save(module.to('cpu').to(torch.bfloat16), intermediate_path(self.module_id))

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id), map_location='cpu')
    
    def load(self, device):
        if device_supports_dtype(device, torch.bfloat16):
            return torch.load(intermediate_path(self.module_id), map_location=device_map(device)).to(self.compute_dtype)
        else:
            # for MPS we need to load to CPU first
            res = torch.load(intermediate_path(self.module_id), map_location='cpu')
            return res.to(self.compute_dtype).to(device_map(device))

    def save(self, module):
        torch.save(module.to('cpu').to(torch.bfloat16), intermediate_path(self.module_id))
    
    def load_input(self, device):
        return torch.load(intermediate_path(self.input_id), map_location=torch.device(device_map(device)))

    def to_state_dict(self):
        module = torch.load(intermediate_path(self.module_id), map_location='cpu')
        return module.state_dict()

    def forward(self, input, *args):
        torch.save(input, intermediate_path(self.input_id))
        device = device_map(input.device)
        module = self.load(device)

        if not self.training:
            module.eval()
        
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)

# same blackbox, but RAM only.
class BlackboxRAM(torch.nn.Module):
    def __init__(self, module, _args):
        super().__init__()
        # this way torch doesn't count it as a submodule, which is 
        # exactly what we want.
        self.module = [module.to('cpu').to(torch.bfloat16)]

    def loaded_inner(self):
        return self.module[0]
    
    def load(self, device):
        return deepcopy(self.module[0]).to(device_map(device))

    def save(self, module):
        self.module = [module.to('cpu').to(torch.bfloat16)]
    
    def load_input(self, device):
        return self.input.to(device_map(device))

    def to_state_dict(self):
        return self.module[0].state_dict()

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