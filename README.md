## slowllama

Fine-tune Llama2 models, including 70B on Apple M1/M2 devices (macbook air!).

slowllama is not using any quantization. Instead, it offloads parts of model to SSD on both forward/backward passes. In contrast with training large models from scratch (unattainable) or inference, where we are likely to care about interactivity, we can still get something finetuned if you let it run for a while.

Current version is using LoRA to limit the updates to a smaller set of parameters. First version supported full finetuning as well, but I decided to remove it for now, more on that below.

slowllama is most definitely not suitable for anything research-like with heavy experimentation as it is way too slow - the iteration cycle duration would kill the productivity. The use-case here is rather to be part of a product (built and debugged on more powerful HW) which makes small changes based on personal/local data, for example set of documents or code someone is working on.

Finetuning is the only focus, there's nothing special done for inference, consider [llama.cpp](https://github.com/ggerganov/llama.cpp).

It should work on CUDA as well, but I didn't do any tests/optimization for that - most likely it'll move things between storage/RAM/GPU more than needed + use wrong types. It should be possible to finetune, for example, 70B llama in bfloat16 with LoRA on single a100.

### Example

Tests were done on Apple M1 with 16Gb memory and Apple M2 with 24Gb memory. 

In order to fine-tune llama2 model we need to:
1. Install dependencies: ```pip install torch sentencepiece``` 
2. Clone [llama2](https://github.com/facebookresearch/llama) and follow instructions to download the models. The script there will download tokenizer as well. ```tokenizer.model``` should be put into the same directory as llama model itself. Example folder structure could look like:
```
/parent/
    /llama-2-7b/... # <- put tokenizer.model here
    /llama-2-13b/... # <- and here
    /llama-2-70b/... # <- and here as well
    /llama/...     # <-- this is Meta's llama2 repository
    /slowllama/... # <- this repo
```

Let's start with a [tiny example](test_data/cubestat.txt). It is an intro to the description of another open-source project - [cubestat](https://github.com/okuvshynov/cubestat). Text is short enough to just be included as part of the prompt, but it's ok as an illustration. As I just published that project recently, there's no way original llama would know anything about it. 

Asking base llama2-7b to complete the prompt _"Cubestat reports the following metrics: "_ results in _"1) the number of cubes in the system, 2) the number of cubes that are in the process of being created"_. 

Try it out:
```
python test_gen.py ../llama-2-7b mps
```

Now let's finetune the 7b model. [finetune.py](finetune.py) is a very simple script which trains LoRA weights based on the plaintext data. There are some settings you could change here, like sequence length, batch size, learning rate, dropout rate, number of iterations. Current settings are pretty much a guess, change this if desired. Base model path is hardcoded in that script as well, change accordingly.

```
python finetune.py
```

Here's train dataset loss:

![train loss](static/train_loss.png)

I didn't added a validation set for this data, I should have looked at these spikes, but instead I just checked what would the fine-tuned model produce for the same prompt.

At ~100 iteration we get the following:  _1. The number of times the application was launched. 2. The number of times the application was closed._

At ~400 iteration much better output is produced: 

_Cubestat reports the following metrics: 1. CPU utilization. 2 GPU utilization. 3 Memory usage. 4 Network interface utilization._

Getting to this level took ~15h on Mac Mini M1.

### How does it work?
For all versions - 7B, 13B and 70B the process is roughly the same.

First, we need to be able to load a model which requires more RAM than we have and save it back in sequential format. We create model instance with all large modules' weights offloaded to SSD - all of the transformer blocks, token embeddings and output linear layer. After that we load model shards one by one, for each shard iterate over all modules, update corresponding subset of its weights and save it back. 

Original llama2 weights are in bfloat16, but mps backend doesn't support that type natively, so we do computation in float32 instead.

Doing forward path is easy - we just load modules when we need and pass the output forward. 

Backward pass is a little more tricky, in a way we have to run forward pass twice. The way it's [currently implemented](https://github.com/okuvshynov/slowllama/blob/main/blackbox_model.py#L351) is:
1. Do a forward pass while also saving inputs to each offloaded block to the hard drive. The goal of the first forward pass is to compute the final loss and cache inputs to each offloaded module. 
2. Then, do a manual backward gradient propagation. We start from the last module, re-run each module with the input we cached on step (1) again. We run backward pass within that block and pass the gradient for the input to the next (previous?) module. As we use LoRA, only LoRA weights are being updated. LoRA weights are not part of the original model and are not offloaded to disk. Important: we also need to save and restore random number generation state before evaluating each offloaded module. During training we use dropout, and randomly switched off neurons should be the same on both forward passes.
3. After that we run optimizer step on LoRA weights and save them separately if needed. LoRA weights are the only one which will have gradients computed.

Original version which can be still found [here](https://github.com/okuvshynov/experiments/tree/5cf944cb1274e577d1e755e6ad1957190d286d9d/split_model) was capable of doing full finetuning and update all weights pretty much the same way. I've temporarily removed that feature to preserve the lifespan of SSDs, as frequent write operations can degrade their performance over time. Reading from SSDs isn't an issue, but they do have a write limit. Limit is typically high enough for normal usage, but in the case of full finetunining we'll have to write, ~150Gb per one iteration/weight update of llama70, assuming stateless optimizer and no gradient accumulation. With AdamW we'll have to save/update another 150-300Gb (depending on data types used) of optimizer state per iteration. If, for example, we assume 1Pb of writes before 500Gb disk will start having issues, even 100 iterations of finetuning would incur significant cost/risk. 

There are still remnants of that code in the current version, for example Dropout layers for static, frozen model, which I should clean up. 
We can bring full finetuning back if needed though.

### Resource requirements/utilization/limitations

![finetune on mac mini](static/finetune_m1_7b.png)

Here we can see resource utilization for 1 full iteration on 7B model - forward and manual backward passes. A few notes:
1. It is slow, and GPU is reasonably well utilized;
2. SSD requirements - even for 13B model you'll need over 50Gb free space: 27Gb for original weights + 27Gb for offloaded layers. Expect hundreds of Gb for 70B model.
3. Forward pass has lower GPU utilization and spends more time on IO as we need to both read weights and write cached inputs/outputs
4. Backward pass achieves very high GPU utilization, close to 100%
5. As we move along layers back and forth, we effectively process them in LIFO order. Thus in the beginning of both forward and backward pass we don't have to access disk, weights are being cached and we don't see disk reads.

Each iteration was taking ~2.5 min for 7B model on Mac Mini M1.

For 70B model Macbook Air M2 one iteration takes ~40min. 70B model was only tested on M2/24Gb machine because smaller one doesn't have enough disk space.

If it is that slow, what's the point?

0. Maybe there's none;
1. The use-case is not doing research/iterate. The way I thought about it was to finetune something based on small amount of new local data. Something where it would be fine to just let it run finetuning overnight every day.
2. Training settings are most likely suboptimal, we can try optimizer with momentum, different learning rate schedule, batch size, sequence length, lora rank, etc.
3. There are some further optimizations: prefetch the weights and inputs, save inputs asynchronously
4. The tests here were done on oldest available M1. Modern higher-end laptops with M2 Max should have ~5x GPU performance, and upcoming models might be even more powerful.
5. Computation is done in float32. MPS device doesn't support bfloat16, but supports float16 - we can look at that. 
6. This approach can be used not only for original llama2 by Meta, but for smaller models with similar architecture, for example ones produced by [llama2.c](https://github.com/karpathy/llama2.c). These might just fit into memory though.
7. If we can amortize the IO well on machines with single GPU and fast storage we should be able to finetune large models in reasonable time. 


### Project structure

Just a few files with no dependencies other than torch and sentencepiece for tokenizer.
```
blackbox_model.py -- model definition and manual backprop implementation. It's based on model.py from [llama2.c](https://github.com/karpathy/llama2.c), also MIT licenced.
finetune.py - script which does the training
loader.py - manual loading/saving of large llama2 models
utils.py - small utility functions, including saving/loading random generator state for different devices.
test_gen.py - greedily complete the prompt. Takes base weights + trained LoRA weights as input. Useful for sanity checks.
```

### TODO:
```
[ ] merge lora back with base model weights and export model in original format. Current 'save' would just save a copy of the original model;
[ ] rope -- double-check the values in checkpoint vs what's being computed.
[ ] make lora params (rank, alpha, dropout) easily configurable;
[ ] check if/how it works on CUDA;
[ ] add tests
[ ] optimizations - prefetch the next layer/input, save asyncronously, etc;
[ ] tests, cleanup and comments;
[ ] progress tracking for everything;
[ ] quantization? at least 16 bit?;
[ ] improve model loading time;
[ ] configurable weight tying;
[ ] double check RNG state correctness.
```

### References
* [llama](https://github.com/facebookresearch/llama)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [llama2.c](https://github.com/karpathy/llama2.c)
* [cubestat](https://github.com/okuvshynov/cubestat)
* [LoRA](https://arxiv.org/abs/2106.09685)

### Contact

github handle @ gmail.com
