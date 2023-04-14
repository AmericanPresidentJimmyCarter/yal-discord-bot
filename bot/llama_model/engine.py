import gc
import time

import torch
import torch.nn as nn

from .gptq import *
from .modelutils import *
from .quant import *

from transformers import AutoTokenizer

MAX_TOKEN_WINDOW = 2048

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LLaMAForCausalLM
    model = LLaMAForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits):
    from transformers import LLaMAConfig, LLaMAForCausalLM 
    config = LLaMAConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


class LlamaEngine():
    model = None
    device = None
    def __init__(
        self,
        model_str: str,
        checkpoint: str,
        wbits: int,
        device: str='cuda:0',
    ):
        DEV = torch.device(device)
        self.device = DEV
        model = load_quant(model_str, checkpoint, wbits)

        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.tokenizer = tokenizer

    async def predict_text(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float,
    ) -> str:
        self.model.to(self.device)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt") \
            .to(self.device)
        input_ids = input_ids[-MAX_TOKEN_WINDOW:]

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=10,
                max_new_tokens=max_length,
                top_p=top_p,
                temperature=temperature,
            )
        output = self.tokenizer.decode([el.item() for el in generated_ids[0]])

        self.model.to('cpu')
        mem = torch.cuda.memory_allocated() / 1e6
        del generated_ids, input_ids
        gc.collect()
        torch.cuda.empty_cache()
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)
        return output
