import torch
import os
import json
import gc
import glob
import logging

from blackbox_model import Transformer, ModelArgs

vocab_size = 32000

# how are weights sharded in llama2 - by rows or columns
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
        logging.info(f'load_llama2: processing checkpoint {ci} out of {shards}')
    
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
            logging.info(f'load_llama2: updating layer {i} out of {len(model.layers)}')
            layer.save(block)

        # now repeat for other submodules: output, embeddings and norm
        title = 'output'
        block = model.output.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'load_llama2: updating output layer')
        model.output.save(block)

        title = 'tok_embeddings'
        block = model.tok_embeddings.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'load_llama2: updating token embeddings')
        model.tok_embeddings.save(block)

        # norm left
        apply_subset(model.norm, checkpoint['norm.weight'], ci, None)

    return model

def add_lora(model_path, lora_path):
    lora_weights = torch.load(lora_path, map_location='cpu')
    paths = sorted(glob.glob(f'{model_path}/consolidated.*.pth'))
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())

    shards = len(paths)

    config = ModelArgs(**config)

    n_layers = int(config.n_layers)

    lora_scale = config.lora_alpha / config.lora_rank

    lora = []

    for layer in range(n_layers):
        w = {}
        for attn_key in ['v', 'q']:
            a_key = f'{attn_key}_lora_{layer}.A.weight'
            b_key = f'{attn_key}_lora_{layer}.B.weight'
            w[attn_key] = lora_weights[b_key].mm(lora_weights[a_key]) * lora_scale
        lora.append(w)

    for ci, checkpoint_path in enumerate(paths):
        logging.info(f'add_lora: processing checkpoint {ci} out of {shards}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for layer in range(n_layers):
            for attn_key in ['v', 'q']:
                local_path = f'attention.w{attn_key}'
                checkpoint_key = f'layers.{layer}.{local_path}.weight'
                subset = get_w_subset(local_path, lora[layer][attn_key], shards, ci)
                checkpoint[checkpoint_key] = checkpoint[checkpoint_key] + lora[layer][attn_key][subset]

        torch.save(checkpoint, checkpoint_path)
