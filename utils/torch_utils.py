from pathlib import Path
import random
from model.ProfileESM import DomainSpanESM
from settings import BASE_PATH
import numpy as np
import torch
import gc
from config import ModelArgs

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available(): 
        torch.mps.manual_seed(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # slow

def get_device():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'Device: {device}')

    return device

def clear_cache():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()

def load_model(filepath: Path|str, model_args=None):

    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_absolute():
        filepath = BASE_PATH / filepath 

    ckpt = torch.load(filepath)
    state_dict = ckpt['model_state_dict']

    if ckpt.get('config') is None and model_args is None:
        raise Exception("Model load: config neither saved and provided")
    elif model_args is None:
            model_args = ModelArgs(**ckpt['config'])

    net = DomainSpanESM(model_args)
    net.load_state_dict(state_dict)

    return net

def save_model(model, filepath: Path|str):
    
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_absolute():
        filepath = BASE_PATH / filepath 

    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config':  model.config.to_dict(),
        }, filepath)