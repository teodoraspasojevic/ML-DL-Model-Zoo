import numpy as np
import math
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms


class LightViT(nn.Module):
    def __init__(self, image_dim, n_patches=7, n_blocks=2, d=8, n_heads=2, num_classes=10):
        super(LightViT, self).__init__()

        # Class Members.
        self.image_dim = image_dim
        self.n_patches = n_patches
        self.d = d
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.num_classes = num_classes

        B, C, H, W = self.image_dim
        patch_size_H = H // self.n_patches
        patch_size_W = W // self.n_patches
        self.patch_dim = (C, patch_size_H, patch_size_W)

        # 1B) Linear Mapping.
        self.linear_map = nn.Linear(in_features=int(np.prod(self.patch_dim)), out_features=d)

        # 2A) Learnable Parameter.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d))

        # 2B) Positional embedding.
        self.pos_embedding = self.get_pos_embeddings(embedding_num=n_patches * n_patches + 1, embedding_dim=self.d)

        # 3) Encoder blocks

        # 5) Classification Head
        self.classifier = None

    def forward(self, images):
        # Extract the patches from images.
        patches = self.get_patches(images, num_patches_per_dim=self.n_patches)

        # Linearly project patches to embeddings of size d.
        embeddings = self.linear_map(patches)
        # TODO: Check if I should do flattening and reshaping.

        # Classification token is like a learnable kernel of size like a patch with all zeros.
        # We add it to other patches -> shape = (B, num_patches + 1, patch_size*patch_size).
        # We do that for each .
        batch_size = embeddings.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Add positional embeddings.
        embeddings = embeddings + self.pos_embedding

        ## Pass through encoder

        ## Get classification token

        ## Pass through classifier

        return embeddings

    @staticmethod
    def get_patches(x, num_patches_per_dim=7):
        """
        Extract patches from an input image.

        Args:
        x (torch.Tensor): Input image of shape (B, C, H, W).
        num_patches_per_dim (int): Number of patches along each dimension.

        Returns:
        torch.Tensor: Output patches of shape (B, num_patches, patch_dim),
                      where num_patches = (H/P) * (W/P) and patch_dim = C * P * P.
        """
        B, C, H, W = x.shape
        patch_size_h = H // num_patches_per_dim
        patch_size_w = W // num_patches_per_dim

        assert H % num_patches_per_dim == 0 and W % num_patches_per_dim == 0, "Image dimensions must be divisible by the number of patches per dimension."

        # Calculate the number of patches along height and width.
        num_patches_h = H // patch_size_h
        num_patches_w = W // patch_size_w
        num_patches = num_patches_h * num_patches_w
        patch_dim = C * patch_size_h * patch_size_w

        # Use unfold to create patches.
        patches = x.unfold(2, size=patch_size_h, step=patch_size_h)  # unfold height dimension
        patches = patches.unfold(3, size=patch_size_w, step=patch_size_w)  # unfold width dimension
        patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, patch_size_h, patch_size_w)

        # Flatten the patches.
        patches = patches.view(B, num_patches, patch_dim)

        return patches

    @staticmethod
    def get_pos_embeddings(embedding_num, embedding_dim):
        # Create a matrix of size embedding_num*embedding_dim (d) to store all positional embeddings.
        pos_embedding = torch.zeros(embedding_num, embedding_dim)

        position = torch.arange(0, embedding_num, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        # Embed even indices using sin, uneven using cos.
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)

        return pos_embedding


class MHSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MHSA, self).__init__()
        assert d % n_heads == 0, "Embedding dimension must be divisible by n_heads"

        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.query_layer = nn.Linear(d, d)
        self.key_layer = nn.Linear(d, d)
        self.value_layer = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def split_heads(self, x):
        N, seq_length, token_dim = x.shape
        x = x.view(N, seq_length, self.n_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # Shape is transformed to   (N, seq_length, n_heads, token_dim / n_heads)
        # And finally we return back    (N, seq_length, item_dim)  (through concatenation)
        N, seq_length, token_dim = sequences.shape

        q = self.split_heads(self.query_layer(sequences))
        k = self.split_heads(self.key_layer(sequences))
        v = self.split_heads(self.value_layer(sequences))

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self_attn_weights = F.softmax(scores, dim=-1)

        self_attn_output = torch.matmul(self_attn_weights, v)
        self_attn_output = self_attn_weights.permute(0, 2, 1, 3).contiguous()
        self_attn_output = self_attn_output.view(N, seq_length, self.d)

        output = self.dense(self_attn_output)

        return output


# Example usage.
if __name__ == '__main__':

    # Define a transform to normalize the data.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data.
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Download and load the test data
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model = LightViT(image_dim=(64, 1, 28, 28), n_patches=7, d=8)

    for images, labels in train_loader:
        output = model(images)

        break
