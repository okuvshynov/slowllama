import torch
import os
import json
import gc
import glob
import logging
import shutil

from model_config import ModelArgs
from llama2 import Transformer
from utils import device_supports_dtype

# how are weights sharded in llama2 - by rows or columns
join_dim = {
    'wq': 0,
    'wk': 0,
    'wv': 0,
    'wo': 1,
    'w1': 0,
    'w2': 1,
    'w3': 0,
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

def prepare_model(llama2_path, frozen_path, **kwargs):
    params_path = os.path.join(llama2_path, 'params.json')
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())

    for k, v in kwargs.items():
        config[k] = v

    args = ModelArgs(**config)
    args.vocab_size = args.vocab_size_override
    args.served_model_path = frozen_path

    logging.info('creating model instance')
    model = Transformer(args)
    paths = sorted(glob.glob(f'{llama2_path}/consolidated.*.pth'))

    shards = len(paths)

    for ci, checkpoint_path in enumerate(paths):
        logging.info(f'prepare_model: processing checkpoint {ci} out of {shards}')
    
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for i, layer in enumerate(model.layers):
            prefix = f'layers.{i}.'
            #block = layer.loaded_inner()
            for title, submodule in layer.named_modules():
                if hasattr(submodule, 'weight'):
                    full_path = f'{prefix}{title}.weight'
                    weight_subset = checkpoint[full_path]
                    apply_subset(submodule, weight_subset, ci, title)
                    del checkpoint[full_path]
                    gc.collect()
            
            prefix = f'layers.{i}.attention.'
            attention = layer.attention.loaded_inner()
            for title, submodule in attention.named_modules():
                if hasattr(submodule, 'weight'):
                    if 'attention_norm' in title:
                        full_path = f'layers.{i}.attention_norm.weight'
                    else:
                        full_path = f'{prefix}{title}.weight'
                    weight_subset = checkpoint[full_path]
                    apply_subset(submodule, weight_subset, ci, title)
                    del checkpoint[full_path]
                    gc.collect()
            layer.attention.save(attention)

            prefix = f'layers.{i}.feed_forward.'
            feed_forward = layer.feed_forward.loaded_inner()
            for title, submodule in feed_forward.named_modules():
                if hasattr(submodule, 'weight'):
                    if 'ffn_norm' in title:
                        full_path = f'layers.{i}.ffn_norm.weight'
                    else:
                        full_path = f'{prefix}{title}.weight'
                    weight_subset = checkpoint[full_path]
                    apply_subset(submodule, weight_subset, ci, title)
                    del checkpoint[full_path]
                    gc.collect()
            layer.feed_forward.save(feed_forward)
            logging.info(f'prepare_model: updating layer {i} out of {len(model.layers)}')
            #layer.save(block)

        # now repeat for other submodules: output, embeddings and norm
        title = 'output'
        block = model.output.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'prepare_model: updating output layer')
        model.output.save(block)

        title = 'tok_embeddings'
        block = model.tok_embeddings.loaded_inner()
        apply_subset(block, checkpoint[f'{title}.weight'], ci, title)
        logging.info(f'prepare_model: updating token embeddings')
        model.tok_embeddings.save(block)

        # norm left
        apply_subset(model.norm, checkpoint['norm.weight'], ci, None)

    # we also need to copy:
    # - params.json
    # - model dict itself (norm + Lora)
    # - tokenizer?'
    shutil.copy(params_path, os.path.join(frozen_path, 'params.json'))
    shutil.copy(os.path.join(llama2_path, 'tokenizer.model'), os.path.join(frozen_path, 'tokenizer.model'))
    torch.save(model.to(args.frozen_dtype).state_dict(), os.path.join(frozen_path, 'model.pth'))

    return model

def load_frozen(path, **kwargs):
    logging.info(f'loading sequential model from {path}')
    params_path = os.path.join(path, 'params.json')
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())

    for k, v in kwargs.items():
        config[k] = v

    args = ModelArgs(**config)
    args.vocab_size = args.vocab_size_override
    args.init_frozen = False
    args.served_model_path = path
    logging.info(f'creating model instance')
    model = Transformer(args).to(args.compute_dtype)
    logging.info(f'loading model dict')
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')), strict=False)
    return model

# this is merging LoRA back to original weights in llama2 format
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

    for ci, checkpoint_path in enumerate(paths):
        logging.info(f'add_lora: processing checkpoint {ci} out of {shards}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for layer in range(n_layers):
            logging.info(f'add_lora: processing checkpoint {ci} layer {layer} out of {n_layers}')
            for attn_key in ['v', 'q']:
                local_path = f'w{attn_key}'
                checkpoint_key = f'layers.{layer}.attention.{local_path}.weight'
                a_key = f'{attn_key}_lora_{layer}.A.weight'
                b_key = f'{attn_key}_lora_{layer}.B.weight'

                original_type = lora_weights[b_key].dtype

                if device_supports_dtype('cpu', original_type):
                    lora = lora_weights[b_key].mm(lora_weights[a_key]) * lora_scale
                else:
                    lora = lora_weights[b_key].to(torch.float32).mm(lora_weights[a_key].to(torch.float32)) * lora_scale
                    lora = lora.to(original_type)

                subset = get_w_subset(local_path, lora, shards, ci)
                checkpoint[checkpoint_key] = checkpoint[checkpoint_key] + lora[subset].to(torch.bfloat16)
        torch.save(checkpoint, checkpoint_path)
        del checkpoint
        gc.collect()
