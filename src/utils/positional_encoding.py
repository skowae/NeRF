import torch
import torch.nn as nn

# Taken from the original paper
# γ(x)=[x,sin(20πx),cos(20πx),sin(21πx),cos(21πx),…,sin(2L−1πx),cos(2L−1πx)]
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True, log_scale=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_scale = log_scale

        if log_scale:
            self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0, 2.0 ** (num_freqs - 1), num_freqs)

    def forward(self, x):
        """
        Args:
            x: (..., input_dims), typically (..., 3)
        Returns:
            encoded: (..., input_dims * (2 * num_freqs) [+ input_dims])
        """
        out = [x] if self.include_input else []

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)
