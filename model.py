import os
import io
from sympy import N
import torch
import random
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    logging,
)
import weightwatcher as ww
from typing import Dict
from loader.layers import get_layers
from loader.alora import alora_model
from loader.blora import get_blocks
import bitsandbytes as bnb # type: ignore
from peft import (
    prepare_model_for_kbit_training, # type: ignore
    LoraConfig, # type: ignore
    get_peft_model, # type: ignore
    PeftModel # type: ignore
)
from peft.tuners.lora import LoraLayer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Borrowed from qlora codebase
    Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean( # type: ignore
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean( # type: ignore
            dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg # type: ignore
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg # type: ignore

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    if 'dora' in args.sortby.lower():
        lora_module_names.add('lora_magnitude_vector')

    print(f'Found {len(lora_module_names)} linear layers')
    print(f'Linear layers: {lora_module_names}')
    return list(lora_module_names)

def get_model(args):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        token="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",
        load_in_4bit = args.bits == 4,
        load_in_8bit = args.bits == 8,
        max_memory=max_memory,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )

    model.config.use_cache = False
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token="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",
        cache_dir=args.cache_dir,
        padding_side="right",
        # use_fast=False, # Fast tokenizer giving issues.
        # tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer, # type: ignore
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    # model.config.pad_token_id
                    # if model.config.pad_token_id != -1
                    tokenizer.pad_token_id # type: ignore
                ),
        })
    
    if args.freeze == True and 'ora' not in args.sortby.lower():
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lm_head" in name:
                param.requires_grad = False
    elif 'ora' not in args.sortby.lower():
        for name, param in model.named_parameters():  # type: ignore
            if "embed_token" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16) 
        return model, tokenizer
    
    # LORA INJECTION >>>-------------------------------------------->
    if 'ora' in args.sortby.lower():
        modules = None
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        print(f'Adding {args.sortby.upper()[0]}oRA modules...')
        if 'adora' in args.sortby.lower() or 'alora' in args.sortby.lower():
            layer_to_train = get_layers(args)
            model = alora_model(args, model, layer_to_train)
        else:
            if len(args.lora_modules) > 0:
                modules = args.lora_modules
                print(f'Using provided modules: {modules}')
            else:
                modules = find_all_linear_names(args, model)
            if 'dora' in args.sortby.lower():
                print('Adding DoRA module...')
                modules.append('lora_magnitude_vector')
            blocks_to_train = None
            if 'blora' in args.sortby.lower():
                blocks_to_train, _ = get_blocks(args)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                layers_to_transform=blocks_to_train,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora='dora' in args.sortby.lower(),
                use_rslora='rslora' in args.sortby.lower(),
            )
            model = get_peft_model(model, config) # type: ignore

    # SELECTIVE FINETUNING >>>-------------------------------------->
    # CHOOSING LAYERS TO TRAIN BASED ON WEIGHTWATCHER METRICS/SORTBY
    if "ora" not in args.sortby.lower():
        layer_to_train = None
        if 'block' in args.sortby.lower():
            _, layer_to_train = get_blocks(args)
        else:
            layer_to_train = get_layers(args)
        for name, param in model.named_parameters():
            if name in layer_to_train:
                param.requires_grad = True
                if args.verbose:
                    print(f"Enabling {name} parameter")
        
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16) # type: ignore
        if 'norm' in name:
            module = module.to(torch.float32) # type: ignore
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16) # type: ignore
    
    return model, tokenizer