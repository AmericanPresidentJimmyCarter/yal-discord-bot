# Yet Another LLaMA Discord Bot



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
git clone -b llama_push https://github.com/zphang/transformers/
cd transformers
pip install -e .
cd ..
```

4. Build 4-bit CUDA kernel.

```bash
cd bot/llama_model
python setup_cuda.py install
cd ../..
```

5. Download the 4-bit quantized model somewhere.

```bash
wget https://huggingface.co/decapoda-research/llama-13b-hf-int4/resolve/main/llama-13b-4bit.pt
```

5. Fire up the bot.

```bash
python -m bot $YOUR_BOT_TOKEN --allow-queue -g $YOUR_GUILD --llama-model="decapoda-research/llama-13b-hf" --load-checkpoint="path/to/llama/weights/llama-13b-4bit.pt"
```

Ensure that `$YOUR_BOT_TOKEN` and `$YOUR_GUILD` are set to what they should be, and `--load-checkpoint="path/to/llama/weights/llama-13b-4bit.pt"` is pointing at the correct location of the weights.

(c) 2023 AmericanPresidentJimmyCarter
