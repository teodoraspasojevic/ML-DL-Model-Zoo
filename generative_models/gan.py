import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x


class GAN:
    def __init__(self, latent_dim, image_dim):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.latent_dim = latent_dim

        self.generator = Generator(input_dim=latent_dim, output_dim=image_dim[0] * image_dim[1]).to(device)
        self.discriminator = Discriminator(input_dim=image_dim[0] * image_dim[1]).to(device)

    def save_samples(self, epoch, batch_size):
        noise = torch.Tensor(np.random.uniform(-1, 1, (batch_size, self.latent_dim))).to(device)
        samples = self.generator(noise)
        samples = samples.view(samples.size(0), 28, 28).cpu().detach().numpy()

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample, cmap='gray')

        if not os.path.exists('res'):
            os.makedirs('res')
        plt.savefig(f'res/generated_images_epoch{epoch}.png', bbox_inches='tight')
        plt.close(fig)

    def train(self, dataloader, num_epochs, lr):

        # Save the training history.
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
        }

        criterion = nn.BCELoss()
        optimizer_G = Adam(self.generator.parameters(), lr=lr)
        optimizer_D = Adam(self.discriminator.parameters(), lr=lr)

        for epoch in range(num_epochs):

            total_generator_loss = 0
            total_discriminator_loss = 0

            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.size(0)

                # Load real images.
                real_imgs = imgs.view(batch_size, -1).to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)

                # Generate fake images.
                noise = torch.Tensor(np.random.uniform(-1, 1, (batch_size, self.latent_dim))).to(device)
                fake_imgs = self.generator(noise)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Train Discriminator on real images.
                outputs = self.discriminator(real_imgs)
                d_loss_real = criterion(outputs, real_labels)

                # Train Discriminator on fake images.
                outputs = self.discriminator(fake_imgs.detach())
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
                total_discriminator_loss += d_loss.item()

                # Train Generator.

                outputs = self.discriminator(fake_imgs)
                g_loss = criterion(outputs, real_labels)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                total_generator_loss += g_loss.item()

            generator_loss = total_generator_loss / len(dataloader)
            discriminator_loss = total_discriminator_loss / len(dataloader)

            history['generator_loss'].append(generator_loss)
            history['discriminator_loss'].append(discriminator_loss)

            if epoch % 20 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss G: {generator_loss:.4f}, Loss D: {discriminator_loss:.4f}")
                self.save_samples(epoch, batch_size)

        return history


def plot_history(history):

    generator_losses = history['generator_loss']
    discriminator_losses = history['discriminator_loss']

    epochs = range(1, len(generator_losses) + 1)

    # Plotting Generator and Discriminator losses.
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, generator_losses, 'b-', label='Generator Loss')
    plt.plot(epochs, discriminator_losses, 'r-', label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Check if GPU is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import MNIST dataset.

    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_dl = DataLoader(mnist_ds, batch_size=100, shuffle=True)

    # Train the GAN.

    image_dim = (28, 28)

    gan = GAN(latent_dim=50, image_dim=image_dim)
    history = gan.train(dataloader=mnist_dl, num_epochs=500, lr=0.0002)
    plot_history(history)
