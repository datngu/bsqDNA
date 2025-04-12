import abc
import torch

class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize a DNA sequence tensor of shape (B, 4, L) into
        an integer tensor of shape (B, L//patch_size) where:
        - B is the batch size
        - 4 represents the one-hot encoded nucleotides (A, C, G, T)
        - L is the sequence length (4096)
        - patch_size is the number of nucleotides per token
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized DNA sequence into a sequence tensor of shape (B, 4, L).
        """

