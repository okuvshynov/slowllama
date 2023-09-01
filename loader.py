import torch
import os
import json
import gc
import shutil
import glob
import logging

from blackbox_model import Transformer, ModelArgs
from utils import peak_rss_mb

vocab_size = 32000

# are weights sharded by rows or columns in llama2?
join_dim = {
    'attention.wq': 0,
    'attention.wk': 0,
    'attention.wv': 0,
    'attention.wo': 1,
    'feed_forward.w1': 0,
    'feed_forward.w2': 1,
    'feed_forward.w3': 0,
    'output': 0,
    'tok_embeddings': 1,
}

def get_subset(title, weight_subset, index):
    if title in join_dim.keys():
        jdim = join_dim[title]
        step = weight_subset.shape[jdim]
        subset = (slice(step * index, step * (index + 1)), slice(None))
        if jdim == 1:
            subset = (subset[1], subset[0])
        return subset
    else:
        return tuple(slice(None) for _ in range(len(weight_subset.shape)))
    
def get_w_subset(title, weight, shards, shard):
    if title in join_dim.keys():
        jdim = join_dim[title]
        step = weight.shape[jdim] // shards
        subset = (slice(step * shard, step * (shard + 1)), slice(None))
        if jdim == 1:
            subset = (subset[1], subset[0])
        return subset
    else:
        return tuple(slice(None) for _ in range(len(weight.shape)))

def apply_subset(module, weight_subset, checkpoint_index, title):
    with torch.no_grad():
        idx_subset = get_subset(title, weight_subset, checkpoint_index)
        module.weight[idx_subset] = weight_subset

def load_llama2(path, **kwargs):
    params_path = os.path.join(path, 'params.json')
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())

    config['vocab_size'] = vocab_size
    for k, v in kwargs.items():
        config[k] = v

    logging.info('creating model instance')
    model = Transformer(ModelArgs(**config))
    paths = sorted(glob.glob(f'{path}/consolidated.*.pth'))

    shards = len(paths)

    for ci, checkpoint_path in enumerate(paths):
        logging.info(f'processing checkpoint {ci} out of {shards}')
    
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for i, layer in enumerate(model.layers):
            prefix = f'layers.{i}.'
            block = layer.loaded_inner()
            for title, submodule in block.named_modules():
                if hasattr(submodule, 'weight'):
                    full_path = f'{prefix}{title}.weight'
                    weight_subset = checkpoint[full_path]
                    #print(title, submodule, full_path, weight_subset.shape)
                    apply_subset(submodule, weight_subset, ci, title)
                    del checkpoint[full_path]
                    gc.collect()
            logging.info(f'updating layer {i} out of {len(model.layers)}')
            layer.save(block)

        # now repeat for other submodules: output, embeddings and norm
        title = 'output'
        block = model.output.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'updating output layer')
        model.output.save(block)

        title = 'tok_embeddings'
        block = model.tok_embeddings.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'updating token embeddings')
        model.tok_embeddings.save(block)

        # norm left
        apply_subset(model.norm, checkpoint['norm.weight'], ci, None)

    return model

# as we finetuning model with same architecture here, 
# let's just copy params.json for now from the old path 
def save_llama2(model, new_path, original_path, shards=1, dtype=torch.bfloat16):
    state_dict = model.cpu().to(dtype).state_dict()
    os.makedirs(new_path, exist_ok=True)
    
    for shard in range(shards):
        logging.info(f'processing shard {shard} out of {shards}')
        # layers:
        for i, layer in enumerate(model.layers):
            logging.info(f'processing layer {i} out of {len(model.layers)}')
            for title, weight in layer.to_state_dict().items():
                title = title[:-len('.weight')]
                subset = get_w_subset(title, weight, shards, shard)
                state_dict[f'layers.{i}.{title}.weight'] = weight[subset].to('cpu').to(dtype)

        title = 'output'
        block = model.output.loaded_inner()
        subset = get_w_subset(title, block.weight, shards, shard)
        state_dict[f'{title}.weight'] = block.weight[subset].to('cpu').to(dtype)
        
        title = 'tok_embeddings'
        block = model.tok_embeddings.loaded_inner()
        subset = get_w_subset(title, block.weight, shards, shard)
        state_dict[f'{title}.weight'] = block.weight[subset].to('cpu').to(dtype)

        state_dict['norm.weight'] = model.norm.weight.to('cpu').to(dtype)

        checkpoint_name = f'consolidated.{shard:02}.pth'
        torch.save(state_dict, os.path.join(new_path, checkpoint_name))
        shutil.copy2(os.path.join(original_path, "params.json"), new_path)
        gc.collect()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    model_a = load_llama2('../llama-2-13b')
    save_llama2(model_a, '../llama-2-13b_x', '../llama-2-13b', shards=2)
