import logging

# which device to use for finetuning
# 'cpu', 'mps' (for Apple devices) or 'cuda'
device = 'mps'

# random seed to use. Makes runs reproducible.
seed = 54321

# learning rate
lr = 1e-4

# logging gradient and weight distribution to log file
# useful for debugging, but makes more 
log_lora_grad = False
log_lora_weight = False

# how wide would LoRA layers be? (N x lora_rank) and (lora_rank x M).
# Larger number - larger layer - more capacity.
lora_rank = 4

log_level = logging.DEBUG

# training settings

# total number of iterations to run. No microbatching so far
iters = 20

# how long should be the sequence to train on? 
# we pick seq_len tokens and try to predict token [seq_len + 1]
seq_len = 128

# how large should be the batch size? 
batch_size = 16

# current script doesn't have validation set at all.
# instead, we run prompt completion every  eval_period iterations
# and check how the completion look like
eval_before_training = False
eval_period = 20

# how many tokens to generate for such test completion
gen_tokens = 32
# what prompt to use for test completion
prompt = 'Cubestat reports the following metrics: '

# where to save LoRA snapshots
snapshots_path = 'out'

# plaintext input file which will be tokenized and used for training 
finetune_file = './test_data/cubestat.txt'

# which model to use - path to raw model
llama2_model_path = '../llama-2-13b'
#llama2_model_path = '../llama-2-13b-out'
