import gc
import time

import torch
import torch.nn as nn

from .gptq import *
from .modelutils import *
from .quant import *

from transformers import AutoTokenizer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
)


DEFAULT_GROUPSIZE = -1
MAX_TOKEN_WINDOW = 2048


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize):
    from transformers import LlamaConfig, LlamaForCausalLM 
    model_name = model
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print(f'Loading model {model_name} (bits: {wbits}, groupsize: {groupsize})...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Model loaded.')

    return model


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        inp_ids = input_ids[0].tolist()
        if any(kw == inp_ids[-len(kw):] for kw in self.keywords):
            return True
        return False


class LlamaEngine():
    model = None
    device = None
    stop_criteria = None

    def __init__(
        self,
        model_str: str,
        checkpoint: str,
        wbits: int,
        groupsize: int=DEFAULT_GROUPSIZE,
        device: str='cuda:0',
    ):
        DEV = torch.device(device)
        self.device = DEV

        model = load_quant(model_str, checkpoint, wbits, groupsize)
            
        model.to(DEV)
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False)
        self.tokenizer = tokenizer

        # Weird stop words from the GPT4 finetuned models.
        stop_words = ['<unk>', '<s>', '</s>', '�', '!0']
        stop_ids = [[0]]
        stop_ids.extend([tokenizer.encode(w) for w in stop_words])
        self.stop_criteria = KeywordsStoppingCriteria(stop_ids)

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
                stopping_criteria=StoppingCriteriaList([self.stop_criteria]),
            )
        output_tokens = [el.item() for el in generated_ids[0]]
        for kw in self.stop_criteria.keywords:
            if output_tokens[-len(kw):] == kw:
                output_tokens = output_tokens[:-len(kw)]
        output = self.tokenizer.decode(output_tokens)

        # TODO Why does the tokenizer generate these?
        if output[-3:] == '</s>':
            output = output[:-3]
        if output[-4:] == '</s>':
            output = output[:-4]

        self.model.to('cpu')
        mem = torch.cuda.memory_allocated() / 1e6
        del generated_ids, input_ids
        gc.collect()
        torch.cuda.empty_cache()
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)
        return output
