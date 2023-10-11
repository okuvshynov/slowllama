# end-to-end test with llama7b.
# TODO: improve to clean up last iter and make it take some params
python prepare_model.py
python test_gen.py
python finetune.py
python test_gen.py ./out/state_dict_19.pth
