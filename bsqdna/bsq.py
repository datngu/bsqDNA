from .tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from .ae import PatchAutoEncoder

def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim

        self.encoder = torch.nn.Linear(embedding_dim, codebook_bits)
        self.decoder = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign

        x.shape: (B, H//patch_size, W//patch_size, embedding_dim)
        """
        x = self.encoder(x)  # Shape: (B*H*W, codebook_bits)
        x = F.normalize(x, dim=-1)
        x = diff_sign(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice

        x.shape: (B, H//patch_size, W//patch_size, codebook_bits)
        """
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (1 << torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (1 << torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


"""
ARCHITECTURE:
    1. PatchEncoder: Image → Latent space (using conv layers)
    2. BSQ.encode:
       - Project latent to lower dimension
       - L2 normalize
       - Quantize to -1/1 values (using diff_sign)
    3. BSQ.decode: Project binary values back to latent dimension
    4. PatchDecoder: Reconstruct original image
"""
class BSQDNA(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.patch_encoder(x)
        return self.bsq.encode_index(encoded)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.bsq.decode_index(x)
        return self.patch_decoder(decoded)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_encoder(x)
        x = self.bsq.encode(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bsq.decode(x)
        x = self.patch_decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        # Codebook monitoring
        tokens = self.encode_index(x)
        cnt = torch.bincount(tokens.flatten(), minlength=2**self.codebook_bits)

        metrics = {
            "cb0": (cnt == 0).float().mean().detach(),  # unused tokens
            "cb2": (cnt <= 2).float().mean().detach(),  # rarely used tokens
            "cb10": (cnt <= 10).float().mean().detach(),  # tokens used <= 10 times
        }

        return decoded, metrics