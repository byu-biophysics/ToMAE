import torch.nn as nn
import torch as torch

# Implement save & loading checkpointing
def save_checkpoint(
        model: nn.Module,
        current_epoch: int,
        hyper_parameters: dict,
        **kwargs
    ):

    pass

def load_checkpoint(
        fpath,
        **kwargs
    ):

    pass
