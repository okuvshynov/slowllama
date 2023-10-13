import os

import torch

from utils import device_map, next_id, device_supports_dtype
from model_config import ModelArgs

class BlackboxDisk(torch.nn.Module):
    def __init__(self, module, args: ModelArgs):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
        self.compute_dtype = args.compute_dtype
        self.served_model_path = args.served_model_path
        self.cached_data_path = args.cached_data_path
        # TODO: can we deduce this from the data itself
        self.frozen_dtype = args.frozen_dtype
        if args.init_frozen:
            torch.save(module.to('cpu').to(self.frozen_dtype), self.frozen_path())

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
        if device_supports_dtype(device, self.frozen_dtype):
            return torch.load(self.frozen_path(), map_location=device_map(device)).to(self.compute_dtype)
        else:
            res = torch.load(self.frozen_path(), map_location='cpu')
            return res.to(self.compute_dtype).to(device_map(device))

    def save(self, module):
        torch.save(module.to('cpu').to(self.frozen_dtype), self.frozen_path())
    
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