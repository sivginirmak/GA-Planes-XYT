import torch
from torch import nn
import numpy as np
from typing import Optional, Tuple, List, Union, Callable



class PositionalEncoder(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):

        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        """
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)



def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:

    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks_sc(points: torch.Tensor, labels: torch.Tensor, encoding_function: Callable[[torch.Tensor], torch.Tensor], chunksize: int = 2**15) -> List[torch.Tensor]:

    d = points.shape[-1]
    points = points.reshape((-1, d))
 
    if encoding_function is not None:
        points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    labels = get_chunks(labels, chunksize=chunksize)
    return points, labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)