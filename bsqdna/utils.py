import numpy as np
import torch
import torch.nn as nn

class DNAOneHotEncoder:
    """
    One-hot encoder for DNA sequences with optional integer encoding

    Attributes:
        vocab (np.ndarray): Array of allowed nucleotides (ACGT by default)

    Methods:
        encode(dna_arr: np.ndarray, out_int: bool = False) -> np.ndarray:
            Encodes DNA sequence array to one-hot or integer array
    """

    def __init__(self):
        self.vocab = np.array(list('ACGT'), dtype='S1')[:, None]  # Column vector for broadcasting

    def encode(self, dna_arr: np.ndarray, out_int: bool = False) -> np.ndarray:
        """
        Encodes DNA sequence array to numerical representation

        Args:
            dna_arr: Input DNA sequence array (dtype='S1')
            out_int: Return integer indices instead of one-hot vectors (default: False)

        Returns:
            np.ndarray: Shape (4, len(dna_arr)) one-hot array or (len(dna_arr),) integer array
        """
        # Convert bytes to numpy array if needed
        if isinstance(dna_arr, bytes):
            dna_arr = np.array([c for c in dna_arr], dtype='S1')
        elif isinstance(dna_arr, list):
            dna_arr = np.array(dna_arr, dtype='S1')

        # Ensure input is 2D for broadcasting
        dna_arr = dna_arr[None, :] if dna_arr.ndim == 1 else dna_arr

        # Create boolean mask then convert to int8
        onehot = np.int8(dna_arr == self.vocab)

        if out_int:
            return np.int8(onehot.argmax(0))  # Convert one-hot to integer indices
        return onehot


class PatchifyLinear(nn.Module):
    """
    Patchifies DNA encodings into embeddings using linear projection

    Args:
        patch_size: Number of nucleotides per patch
        latent_dim: Output embedding dimension
        in_channels: Number of input channels (default: 4 for DNA one-hot encoding)

    Shape:
        Input: (B, in_channels, L) where:
            B = batch size
            in_channels = input channels (default 4 for A,C,G,T)
            L = sequence length
        Output: (B, L//patch_size, latent_dim)
    """
    def __init__(self, patch_size: int, latent_dim: int, in_channels: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # Uses Conv1d for efficient patch projection
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=latent_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True  # Linear transformation includes bias term
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, L)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, latent_dim)
        """
        # Input shape checks
        assert x.ndim == 3, f"Expected 3D input (B, {self.in_channels}, L), got {x.shape}"
        B, C, L = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} input channels, got {C}"
        assert L % self.patch_size == 0, f"Sequence length {L} must be divisible by patch_size {self.patch_size}"

        # Project patches: (B, in_channels, L) -> (B, latent_dim, num_patches)
        x = self.proj(x)

        # Permute to (B, num_patches, latent_dim)
        return x.permute(0, 2, 1)


class UnpatchifyLinear(nn.Module):
    """
    Reverses the patchification of DNA embeddings back to original encoding using linear projection

    Args:
        patch_size: Number of nucleotides per patch
        latent_dim: Input embedding dimension
        out_channels: Number of output channels (default: 4 for DNA one-hot encoding)

    Shape:
        Input: (B, L//patch_size, latent_dim) where:
            B = batch size
            L = sequence length
        Output: (B, out_channels, L)
    """
    def __init__(self, patch_size: int, latent_dim: int, out_channels: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Uses ConvTranspose1d for efficient unpatchification
        self.proj = nn.ConvTranspose1d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True  # Linear transformation includes bias term
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, num_patches, latent_dim)

        Returns:
            torch.Tensor: Unpatchified encoding of shape (B, out_channels, L)
        """
        # Input shape checks
        assert x.ndim == 3, f"Expected 3D input (B, num_patches, latent_dim), got {x.shape}"
        B, num_patches, latent_dim = x.shape
        assert latent_dim == self.latent_dim, f"Expected latent_dim {self.latent_dim}, got {latent_dim}"

        # Permute to (B, latent_dim, num_patches)
        x = x.permute(0, 2, 1)

        # Unproject patches: (B, latent_dim, num_patches) -> (B, out_channels, L)
        x = self.proj(x)

        return x


''' ATTENTION AUTOENCODER
'''

class PatchifyAttention(nn.Module):
    """
    Patchifies DNA encodings into embeddings using linear projection

    Args:
        patch_size: Number of nucleotides per patch
        latent_dim: Output embedding dimension
        in_channels: Number of input channels (default: 4 for DNA one-hot encoding)

    Shape:
        Input: (B, in_channels, L) where:
            B = batch size
            in_channels = input channels (default 4 for A,C,G,T)
            L = sequence length
        Output: (B, L//patch_size, latent_dim)
    """
    def __init__(self, patch_size: int, latent_dim: int, in_channels: int = 4, attention_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.attention_heads = attention_heads

        self.q_proj = nn.Linear(self.in_channels * self.patch_size, self.latent_dim)
        self.k_proj = nn.Linear(self.in_channels * self.patch_size, self.latent_dim)
        self.v_proj = nn.Linear(self.in_channels * self.patch_size, self.latent_dim)

        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=attention_heads, batch_first=True)

        self.proj = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, L)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, latent_dim)
        """
        # Input shape checks
        assert x.ndim == 3, f"Expected 3D input (B, {self.in_channels}, L), got {x.shape}"
        B, C, L = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} input channels, got {C}"
        assert L % self.patch_size == 0, f"Sequence length {L} must be divisible by patch_size {self.patch_size}"

        num_patches = L // self.patch_size

        # Transform x: (B, in_channels, L) -> (B, num_patches, in_channels, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size) # (B, in_channels, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous() # (B, num_patches, in_channels, patch_size)
        x = x.view(B, num_patches, -1) # (B, num_patches, in_channels * patch_size)

        # Project to latent space: (B, num_patches, in_channels * patch_size) -> (B, num_patches, latent_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply attention
        attention_x, _ = self.attention(q, k, v)

        # Attention
        x = self.proj(attention_x)

        # Permute to (B, num_patches, latent_dim)
        return x


class UnpatchifyAttention(nn.Module):
    """
    Reverses the patchification of DNA embeddings back to original encoding using linear projection

    Args:
        patch_size: Number of nucleotides per patch
        latent_dim: Input embedding dimension
        out_channels: Number of output channels (default: 4 for DNA one-hot encoding)

    Shape:
        Input: (B, L//patch_size, latent_dim) where:
            B = batch size
            L = sequence length
        Output: (B, out_channels, L)
    """
    def __init__(self, patch_size: int, latent_dim: int, out_channels: int = 4, attention_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.attention_heads = attention_heads

        self.q_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.k_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.v_proj = nn.Linear(self.latent_dim, self.latent_dim)

        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=attention_heads, batch_first=True)

        self.proj = nn.Linear(self.latent_dim, self.out_channels * self.patch_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, num_patches, latent_dim)

        Returns:
            torch.Tensor: Unpatchified encoding of shape (B, out_channels, L)
        """
        # Input shape checks
        assert x.ndim == 3, f"Expected 3D input (B, num_patches, latent_dim), got {x.shape}"
        B, num_patches, latent_dim = x.shape
        assert latent_dim == self.latent_dim, f"Expected latent_dim {self.latent_dim}, got {latent_dim}"

        # Apply attention: (B, num_patches, latent_dim) -> (B, num_patches, latent_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attention_x, _ = self.attention(q, k, v)

        # Unproject patches: (B, num_patches, latent_dim) -> (B, out_channels, L)
        x = self.proj(attention_x) # (B, num_patches, out_channels * patch_size)
        x = x.view(B, num_patches, self.out_channels, self.patch_size) # (B, num_patches, out_channels, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous() # (B, out_channels, num_patches, patch_size)
        x = x.view(B, self.out_channels, -1)

        return x
