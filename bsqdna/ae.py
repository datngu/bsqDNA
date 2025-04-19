import abc

import torch
import torch.nn as nn
from .utils import PatchifyLinear, UnpatchifyLinear, PatchifyAttention, UnpatchifyAttention


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input DNA sequence tensor x (B, 4, L) into a tensor (B, L//patch_size, bottleneck),
        where:
        - B is the batch size
        - 4 represents one-hot encoded nucleotides (A, C, G, T)
        - L is the sequence length (4096)
        - patch_size is the number of nucleotides per token
        - bottleneck is the size of the AutoEncoder's bottleneck
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, L//patch_size, bottleneck) into a DNA sequence tensor (B, 4, L).
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder for DNA sequences

    The model processes DNA sequences in patches, where each patch contains a fixed number
    of nucleotides. The input sequences are one-hot encoded with shape (B, 4, L) where:
    - B is the batch size
    - 4 represents one-hot encoded nucleotides (A, C, G, T)
    - L is the sequence length (4096)
    """
    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder for DNA sequences.
        """

        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            self.patchify = PatchifyAttention(patch_size, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.patchify(x)

    class PatchDecoder(torch.nn.Module):
        """
        Decoder module that reconstructs DNA sequences from their latent representation
        """
        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            self.unpatchify = UnpatchifyAttention(patch_size, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.unpatchify(x)

    def __init__(self, patch_size: int = 8, latent_dim: int = 128):
        super().__init__()
        self.patch_encoder = self.PatchEncoder(patch_size, latent_dim)
        self.patch_decoder = self.PatchDecoder(patch_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process a batch of DNA sequences through the autoencoder.
        Input shape: (B, 4, L) where B is batch size, 4 is one-hot encoded nucleotides, L is sequence length
        Returns the reconstructed sequences and a dictionary of additional loss terms.
        You can return an empty dictionary if you don't have any additional terms.
        """
        x = self.patch_encoder(x)
        x = self.patch_decoder(x)
        return x, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_decoder(x)
