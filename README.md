## slowllama

Fine-tune Llama2 and CodeLLama models, including 70B on Apple M1/M2 devices (for example, Macbook Air!) or consumer nVidia GPUs.

slowllama is not using any quantization. Instead, it offloads parts of model to SSD or main memory on both forward/backward passes. In contrast with training large models from scratch (unattainable) or inference, where we are likely to care about interactivity, we can still get something finetuned if you let it run for a while.

Current version is using LoRA to limit the updates to a smaller set of parameters. First version supported full finetuning as well, but I decided to remove it for now, more on that below. 

Finetuning is the only focus, there's nothing special done for inference, consider [llama.cpp](https://github.com/ggerganov/llama.cpp).

For CUDA-specific experiments, see [report on a10](docs/a10.md).

Code Llama update: after fixing rope_theta configuration it loads, trains, and generates some python code. I haven't tried to do any 'real' fintuning yet.
AdamW update: much better, not that slow anymore.

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
    /llama/...     # <-- this is Meta's llama2 repository.
    /slowllama/... # <- this repo
```
3. codellama is essentially the same

Let's start with a [tiny example](test_data/cubestat.txt). It is an intro to the description of another open-source project - [cubestat](https://github.com/okuvshynov/cubestat). Text is short enough to just be included as part of the prompt, but it's ok as an illustration and you can read it in seconds youself. As I just published that project recently, there's no way original llama would know anything about it. 

Asking base llama2-7b to complete the prompt _"Cubestat reports the following metrics: "_ results in _"1) the number of cubes in the system, 2) the number of cubes that are in the process of being created"_. 

Try it out:
```
python test_gen.py ../llama-2-7b mps
```

Now let's finetune the 7b model. [finetune.py](finetune.py) is a very simple script which trains LoRA weights based on the plaintext data. There are some settings you could change here, like sequence length, batch size, learning rate, dropout rate, number of iterations. Current settings are pretty much a guess, change this if desired. Base model path and path to Meta's llama repository are hardcoded in that script as well, assuming folder structure above. Adjust accordingly. Currently uses AdamW optimizer.

```
python finetune.py
```

Here's train dataset loss:
```
2023-09-10 22:05:35,569 backprop done, loss after forward pass = 2.9539270401000977
2023-09-10 22:06:08,022 backprop done, loss after forward pass = 2.9073102474212646
2023-09-10 22:06:40,223 backprop done, loss after forward pass = 2.7192320823669434
2023-09-10 22:07:12,468 backprop done, loss after forward pass = 2.7223477363586426
2023-09-10 22:07:44,626 backprop done, loss after forward pass = 2.5889995098114014
2023-09-10 22:08:16,899 backprop done, loss after forward pass = 2.4459967613220215
2023-09-10 22:08:49,072 backprop done, loss after forward pass = 2.3632657527923584
2023-09-10 22:09:21,335 backprop done, loss after forward pass = 2.250361442565918
2023-09-10 22:09:53,511 backprop done, loss after forward pass = 2.165428638458252
2023-09-10 22:10:25,738 backprop done, loss after forward pass = 2.031874656677246
2023-09-10 22:13:45,794 backprop done, loss after forward pass = 1.8926434516906738
2023-09-10 22:14:18,049 backprop done, loss after forward pass = 1.7222942113876343
2023-09-10 22:14:50,243 backprop done, loss after forward pass = 1.58726966381073
2023-09-10 22:15:22,405 backprop done, loss after forward pass = 1.4983913898468018
2023-09-10 22:15:54,598 backprop done, loss after forward pass = 1.296463131904602
2023-09-10 22:16:26,909 backprop done, loss after forward pass = 1.3328818082809448
2023-09-10 22:16:59,031 backprop done, loss after forward pass = 1.0978631973266602
2023-09-10 22:17:31,200 backprop done, loss after forward pass = 1.018444538116455
2023-09-10 22:18:03,406 backprop done, loss after forward pass = 0.8421685099601746
2023-09-10 22:18:35,673 backprop done, loss after forward pass = 0.7168515920639038
2023-09-10 22:21:55,482 backprop done, loss after forward pass = 0.7870235443115234
```

I didn't add a validation set for this data, instead I just checked what would the fine-tuned model produce for the same prompt.

At ~10 iteration we get the following reasonable output:  _Cubestat reports the following metrics: 1. CPU usage, 2. Memory usage, 3. Disk usage_

At ~20 iteration another output is produced:

_0 - Cubestat reports the following metrics: CPU utilization: Efficiency and Performance cores. Shows as percentage._

### How does it work?
For all versions - 7B, 13B and 70B the process is roughly the same.

First, we need to be able to load a model which requires more RAM than we have and save it back in sequential format. We create model instance with all large modules' weights offloaded to SSD - all of the transformer blocks, token embeddings and output linear layer. After that we [load model shards one by one](https://github.com/okuvshynov/slowllama/blob/main/loader.py#L69), for each shard iterate over all modules, update corresponding subset of its weights and save it back. 

Doing forward path is easy - we just load modules when we need and pass the output forward. 

Backward pass is a little more tricky, in a way we have to run forward pass twice. The way it's [currently implemented](https://github.com/okuvshynov/slowllama/blob/main/blackbox_model.py#L351) is:
1. Do a forward pass while also saving inputs to each offloaded block to the SSD. The goal of the first forward pass is to compute the final loss and cache inputs to each offloaded module. 
2. Then, do a manual backward gradient propagation. We start from the last module, re-run each module (forward, to build autograd graph) with the input we cached on step (1) again. After that we run backward pass within that block only, and pass the gradient for the input to the next (previous?) module. As we use LoRA, only LoRA weights are being updated. LoRA weights are not part of the original model and are not offloaded to disk. Important: we also need to save and restore random number generation state before evaluating each offloaded module. During training we use dropout, and randomly switched off neurons should be the same on both forward passes.
3. After that we run optimizer step on LoRA weights and save them separately if needed.

Original llama2 weights are in bfloat16, but mps backend doesn't support that type natively, so we do computation in float32 instead.

Experimental version of slowllama which can be still found [here](https://github.com/okuvshynov/experiments/tree/5cf944cb1274e577d1e755e6ad1957190d286d9d/split_model) was capable of doing full finetuning and update all weights pretty much the same way. I've temporarily removed that feature to preserve the lifespan of SSDs, as frequent write operations can degrade performance over time. Reading from SSDs isn't an issue, but they do have a write limit. Limit is typically high enough for normal usage, but in the case of full finetunining we'll have to write ~150Gb per one iteration/weight update of llama70, assuming stateless optimizer and no gradient accumulation. With AdamW we'll have to save/update another 150-300Gb (depending on data types used) of optimizer state per iteration. If, for example, we assume 1Pb of writes before SSD will start having issues, even 100 iterations of finetuning would incur significant cost/risk. For machines with GPUs and large amount of RAM we can skip the disk entirely and offload to RAM only. It should be possible to bring full finetuning back for main-memory-only offload. On the other hand, if everything fits into memory, there's no need to do whole 'evaluate twice' thing, might just use [fairscale](https://fairscale.readthedocs.io/en/stable/deep_dive/offload.html) instead?


### Resource requirements/utilization/limitations

![finetune on mac mini](static/finetune_m1_7b.png)

Here we can see resource utilization for 1 full iteration on 7B model - forward and manual backward passes. A few notes:
1. It is slow;
2. GPU is reasonably well utilized;
3. Disk space requirements - even for 13B model you'll need over 50Gb free space: 27Gb for original weights + 27Gb for offloaded layers. Expect hundreds of Gb for 70B model.
4. First forward pass has lower GPU utilization and spends more time on IO as we need to both read weights and write cached inputs/outputs
5. Backward (combined?) pass achieves very high GPU utilization, close to 100%
6. As we move along layers back and forth, right after each 'direction switch' we process layers in LIFO order. Thus in the beginning of both forward and backward pass we don't have to access disk, weights are being cached and we don't see disk reads.

Each iteration was taking ~2.5 min for 7B model on Mac Mini M1.

For 70B model Macbook Air M2 one iteration takes ~40min. 70B model was only tested on M2/24Gb machine because smaller one doesn't have enough disk space.

If it is that slow, what's the point?

0. Maybe there's none;
1. The use-case is not doing research/iterate. The way I thought about it was to finetune something based on small amount of new local data. Something where it would be fine to just let it run finetuning overnight every day.
2. There are some further optimizations: prefetch the weights and inputs, save inputs asynchronously
3. The tests here were done on oldest available M1. Modern higher-end laptops with M2 Max should have ~5x GPU performance, and upcoming models might be even more powerful.
4. Computation is done in float32. MPS device doesn't support bfloat16, but supports float16 - we can look at that. 
5. This approach can be used not only for original llama2 by Meta, but for smaller models with similar architecture, for example ones produced by [llama2.c](https://github.com/karpathy/llama2.c). These might just fit into memory though.

### Project structure

Just a few files with no dependencies other than torch and sentencepiece for tokenizer.

1. [blackbox_model.py](blackbox_model.py) -- model definition and manual backprop implementation. It's based on model.py from [llama2.c](https://github.com/karpathy/llama2.c), also MIT licenced.
2. [finetune.py](finetune.py) - script which does the training
3. [loader.py](loader.py) - manual loading/saving of large llama2 models
4. [utils.py](utils.py) - small utility functions, including saving/loading random generator state for different devices.
5. [test_gen.py](test_gen.py) - greedily complete the prompt. Takes base weights + trained LoRA weights as input. Useful for sanity checks.
6. [blackbox.py](blackbox.py) - module wrapper which offloads the module to disk or main memory.
7. [plot_lora.py](plot_lora.py) - logging utility, writes LoRA weights and gradient distribution to [logfile](docs/lora_weights.md). Requires [fewlines](https://github.com/okuvshynov/fewlines). If fewlines is not installed, does nothing.


### TODO:

```
[ ] merge lora weights with base model weights and export the combined result in original format.
[x] check if/how it works on CUDA;
[x] rope -- double-check the values in original checkpoint vs what's being computed.
[x] make lora params (rank, alpha, dropout) easily configurable;
[x] try RAM offload
[x] AdamW
[x] logging weight/gradient distribution
[ ] optimizations - prefetch the next layer/input, save asyncronously, etc;
[ ] combined RAM/disk offload - 200Gb ram is rarity.
[ ] tests, cleanup and comments;
[ ] progress tracking for everything;
[x] quantization? at least 16 bit?;
[ ] quantization beyond 16 bit?
[ ] improve model loading time;
[ ] configurable weight tying;
[ ] double check RNG state correctness.
```

### References
* [llama2](https://github.com/facebookresearch/llama)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [llama2.c](https://github.com/karpathy/llama2.c)
* [cubestat](https://github.com/okuvshynov/cubestat)
* [LoRA](https://arxiv.org/abs/2106.09685)

### Contact

{github handle} @ gmail.com
