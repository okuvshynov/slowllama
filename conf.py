import logging

offload_to = 'disk'
device = 'mps'
seed = 54321

lr = 1e-4

log_lora_grad = False
log_lora_weight = False

lora_rank = 4

log_level = logging.DEBUG

# training settings

iters = 20
seq_len = 128
batch_size = 16

eval_before_training = False
eval_period = 20
gen_tokens = 32

snapshots_path = 'out'
finetune_file = './test_data/cubestat.txt'
prompt = 'Cubestat reports the following metrics: '

llama2_model_path = '../llama-2-70b'
