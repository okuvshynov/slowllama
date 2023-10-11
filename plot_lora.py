import logging
import torch

def log_lora(lora_layers, log_weights=True, log_grad=True, log_level=logging.INFO):
    if not log_weights and not log_grad:
        return
    try:
        from fewlines import bar
    except ImportError:
        logging.error('Unable to import fewlines. "pip install fewlines" to use distribution logging')
        return
    
    gradients_a = {}
    gradients_b = {}
    weights_a = {}
    weights_b = {}

    for i, lora in enumerate(lora_layers):
        q = lora['q_lora']
        v = lora['v_lora']
        if log_grad:
            gradients_a[f'Q{i}.A'] = q.A.weight.grad.view(-1).to(torch.float32).tolist()
            gradients_b[f'Q{i}.B'] = q.B.weight.grad.view(-1).to(torch.float32).tolist()
            gradients_a[f'V{i}.A'] = v.A.weight.grad.view(-1).to(torch.float32).tolist()
            gradients_b[f'V{i}.B'] = v.B.weight.grad.view(-1).to(torch.float32).tolist()
        if log_weights:
            weights_a[f'Q{i}.A'] = q.A.weight.view(-1).to(torch.float32).tolist()
            weights_b[f'Q{i}.B'] = q.B.weight.view(-1).to(torch.float32).tolist()
            weights_a[f'V{i}.A'] = v.A.weight.view(-1).to(torch.float32).tolist()
            weights_b[f'V{i}.B'] = v.B.weight.view(-1).to(torch.float32).tolist()

    if log_grad:
        logging.log(log_level, f'\n=== GRADIENTS A ===')
        for l in bar.bar_histograms(gradients_a):
            logging.log(log_level, l)

        logging.log(log_level, f'\n=== GRADIENTS B ===')
        for l in bar.bar_histograms(gradients_b):
            logging.log(log_level, l)

    if log_weights:
        logging.log(log_level, f'\n=== WEIGHTS A ===')
        for l in bar.bar_histograms(weights_a):
            logging.log(log_level, l)

        logging.log(log_level, f'\n=== WEIGHTS B ===')
        for l in bar.bar_histograms(weights_b):
            logging.log(log_level, l)