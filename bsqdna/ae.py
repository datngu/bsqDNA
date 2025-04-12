import abc

import torch
import torch.nn as nn
from utils import PatchifyLinear, UnpatchifyLinear

def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """
    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patchify = PatchifyLinear(patch_size, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.patchify(x)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.unpatchify(x)

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.patch_encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.patch_decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        x = self.patch_encoder(x)
        x = self.patch_decoder(x)
        return x, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_decoder(x)
