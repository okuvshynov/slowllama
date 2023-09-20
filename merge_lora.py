import logging
import sys
import shutil

from loader import add_lora

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='merge_lora.log')
    
model_path = sys.argv[1]
lora_path = sys.argv[2]
out_model_path = sys.argv[3]

#shutil.copytree(model_path, out_model_path)
add_lora(out_model_path, lora_path)
