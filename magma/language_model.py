import torch
from transformers import AutoModelForCausalLM, GPTJForCausalLM, GPTJConfig, OPTForCausalLM, OPTConfig
from .utils import print_main
from magma.config import MultimodalConfig

LANGUAGE_MODELS = [
    "gptj",
    "galai",
    "opt"
]

def get_lm(config: MultimodalConfig,
    gradient_checkpointing: bool = True,
) -> torch.nn.Module:
    if 'gpt-j' in config.lm_name:
        return get_gptj(config, gradient_checkpointing, config.lm_name)
    elif 'galactica' in config.lm_name or 'opt' in config.lm_name:
        return get_opt(config, gradient_checkpointing, config.lm_name)
    else:
        raise NotImplementedError


def get_gptj(config: MultimodalConfig,
    gradient_checkpointing: bool = True,
    from_pretrained="EleutherAI/gpt-j-6B",
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    print_main("From", from_pretrained)
    gptj_config = GPTJConfig.from_pretrained(from_pretrained)
    gptj_config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        gptj_config.use_cache = False

    if config.deepspeed_config_params['fp16']['enabled'] is True:
        model = GPTJForCausalLM.from_pretrained(
            from_pretrained, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, config=gptj_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(from_pretrained, config=gptj_config)

    return model


def get_opt(config: MultimodalConfig,
    gradient_checkpointing: bool = True,
    from_pretrained="facebook/galactica-6.7b",
) -> torch.nn.Module:
    print_main("Loading OPT language model...")
    print_main("From", from_pretrained)
    opt_config = OPTConfig.from_pretrained(from_pretrained)
    opt_config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        opt_config.use_cache = False

    model = OPTForCausalLM.from_pretrained(from_pretrained, config=opt_config) # device_map="auto",
    if config.deepspeed_config_params['fp16']['enabled'] is True:
        model = model.half()
    return model