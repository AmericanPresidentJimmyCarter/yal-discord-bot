# Yet Another LLaMA Discord Bot


## What is this?

A chatbot for Discord using Meta's LLaMA model, 4-bit quantized. The 13 billion parameters model fits within less than 9 GiB VRAM.

![Yet Another LLaMA Diffusion Discord Bot Splash Image](https://github.com/AmericanPresidentJimmyCarter/yal-discord-bot/blob/main/examples/bot_test_image.png?raw=true)


## Installation

Presently this is Linux only, but you might be able to make it work with other OSs.

1. Make sure you have Python 3.10+, virtualenv (`pip install virtualenv`), and CUDA installed.
2. Clone the bot and setup the virtual environment.

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/yal-discord-bot/
cd yal-discord-bot
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

3. Setup transformers fork and ignore any version incompatibility errors when you do this.

```bash
git clone https://github.com/huggingface/transformers/
cd transformers
git checkout 20e54e49fa11172a893d046f6e7364a434cbc04f
pip install -e .
cd ..
```

4. Build 4-bit CUDA kernel.

```bash
cd bot/llama_model
python setup_cuda.py install
cd ../..
```

5. Download the 4-bit quantized model to somewhere local. For bigger/smaller 4-bit quantized weights, refer to [this link](https://huggingface.co/Neko-Institute-of-Science/LLaMA-13B-4bit-128g/).

```bash
wget https://huggingface.co/Neko-Institute-of-Science/LLaMA-13B-4bit-128g/resolve/main/llama-13b-4bit-128g.safetensors
```

5. Fire up the bot.

```bash
cd bot
python -m bot $YOUR_BOT_TOKEN --allow-queue -g $YOUR_GUILD --llama-model="Neko-Institute-of-Science/LLaMA-13B-4bit-128g" --groupsize=128 --load-checkpoint="path/to/llama/weights/llama-13b-4bit-128g.safetensors"
```

Ensure that `$YOUR_BOT_TOKEN` and `$YOUR_GUILD` are set to what they should be, `--load-checkpoint=..."` is pointing at the correct location of the weights, and `--llama-model=...` is pointing at the correct location in Huggingface to find the configuration for the weights.

## Using an ALPACA model (Recommended)

You can use any ALPACA model by setting the `--alpaca` flag, which will allow you to add input strings as well as automatically format your prompt into the form expected by ALPACA.

Recommended 4-bit ALPACA weights are as follows:

- [13b (elinas/alpaca-13b-lora-int4)](https://huggingface.co/elinas/alpaca-13b-lora-int4)
- [30b (elinas/alpaca-30b-lora-int4)](https://huggingface.co/elinas/alpaca-30b-lora-int4)

Or GPT4 finetuned (better coding responses, more restrictive in content):

- [30b (MetaIX/GPT4-X-Alpaca-30B-Int4)](https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-Int4)

```bash
cd bot
python -m bot $YOUR_BOT_TOKEN --allow-queue -g $YOUR_GUILD --alpaca --groupsize=128 --llama-model="elinas/alpaca-30b-lora-int4" --load-checkpoint="path/to/alpaca/weights/alpaca-30b-4bit-128g.safetensors"
```

(c) 2023 AmericanPresidentJimmyCarter
