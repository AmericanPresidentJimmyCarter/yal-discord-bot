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
git clone https://github.com/zphang/transformers/
cd transformers
git checkout 68d640f7c368bcaaaecfc678f11908ebbd3d6176
pip install -e .
cd ..
```

4. Build 4-bit CUDA kernel.

```bash
cd bot/llama_model
python setup_cuda.py install
cd ../..
```

5. Download the 4-bit quantized model to somewhere local. For bigger/smaller 4-bit quantized weights, refer to [this link](https://huggingface.co/decapoda-research/).

```bash
wget https://huggingface.co/decapoda-research/llama-13b-hf-int4/resolve/main/llama-13b-4bit.pt
```

5. Fire up the bot.

```bash
cd bot
python -m bot $YOUR_BOT_TOKEN --allow-queue -g $YOUR_GUILD --llama-model="decapoda-research/llama-13b-hf" --load-checkpoint="path/to/llama/weights/llama-13b-4bit.pt"
```

Ensure that `$YOUR_BOT_TOKEN` and `$YOUR_GUILD` are set to what they should be, `--load-checkpoint=..."` is pointing at the correct location of the weights, and `--llama-model=...` is pointing at the correct location in Huggingface to find the configuration for the weights.

## Using an ALPACA model

You can use any ALPACA model by setting the `--alpaca` flag, which will allow you to add input strings as well as automatically format your prompt into the form expected by ALPACA.

Recommended 4-bit ALPACA weights are as follows:

- [13b (elinas/alpaca-13b-lora-int4)](https://huggingface.co/elinas/alpaca-13b-lora-int4)
- [30b (elinas/alpaca-30b-lora-int4)](https://huggingface.co/elinas/alpaca-30b-lora-int4)

```bash
cd bot
python -m bot $YOUR_BOT_TOKEN --allow-queue -g $YOUR_GUILD --alpaca --llama-model="decapoda-research/llama-13b-hf" --load-checkpoint="path/to/alpaca/weights/alpaca-13b-4bit.pt"
```

Note that you still have to use the old configuration at `--llama-model="decapoda-research/llama-13b-hf"` because that is the configuration that the old version of GPTQ/transformers I cloned works with.

(c) 2023 AmericanPresidentJimmyCarter
