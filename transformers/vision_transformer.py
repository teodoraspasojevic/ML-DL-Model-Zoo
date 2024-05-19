import torch
import torch.nn as nn
from torchvision import datasets, transforms


class LightViT(nn.Module):
    def __init__(self, image_dim, n_patches=7, n_blocks=2, d=8, n_heads=2, num_classes=10):
        super(LightViT, self).__init__()

        ## Class Members
        self.image_dim = image_dim
        self.n_patches = n_patches
        self.patch_size = self.image_dim // self.n_patches
        self.d = d

        ## 1B) Linear Mapping
        self.linear_map = nn.Linear(self.patch_size * self.patch_size, d)
        ## 2A) Learnable Parameter
        self.cls_token = None;
        ## 2B) Positional embedding
        self.pos_embed = None;
        ## 3) Encoder blocks

        # 5) Classification Head
        self.classifier = None

    def forward(self, images):
        ## Extract patches
        patches = self.patches(images, num_patches_per_dim=self.n_patches)

        ## Linear mapping
        patches = self.linear_map(patches)

        ## Add classification token

        ## Add positional embeddings

        ## Pass through encoder

        ## Get classification token

        ## Pass through classifier

        return patches

    def patches(self, x, num_patches_per_dim=7):
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

    model = LightViT(image_dim=28, n_patches=7, d=8)

    for images, labels in train_loader:
        output = model(images)

        break
