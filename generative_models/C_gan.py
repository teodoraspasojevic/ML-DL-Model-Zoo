import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class cGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(cGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 7 x 7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 14 x 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 28 x 28
            nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # final state size. 1 x 28 x 28
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        x = torch.cat([x, c], 1)
        x = self.model(x)
        return x


class cDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(cDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            # input size. 1 x 28 x 28
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 14 x 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 7 x 7
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output size. 1 x 1 x 1
            nn.Flatten()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], 1)
        x = self.model(x)
        return x


class cGAN:
    def __init__(self, latent_dim, num_classes):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = cGenerator(latent_dim=latent_dim, num_classes=self.num_classes).to(device)
        self.discriminator = cDiscriminator(num_classes=self.num_classes).to(device)

    def save_samples(self, epoch, batch_size, labels):
        noise = torch.Tensor(np.random.uniform(-1, 1, (batch_size, self.latent_dim, 1, 1))).to(device)
        samples = self.generator(noise, labels)
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
        optimizer_G = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(num_epochs):

            total_generator_loss = 0
            total_discriminator_loss = 0

            for imgs, labels in dataloader:
                batch_size = imgs.size(0)

                labels = labels.to(device)

                # Load real images.
                real_imgs = imgs.to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)

                # Generate fake images.
                noise = torch.Tensor(np.random.uniform(-1, 1, size=(batch_size, self.latent_dim, 1, 1))).to(device)
                fake_imgs = self.generator(noise, labels)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Train Discriminator on real images.
                outputs = self.discriminator(real_imgs, labels)
                d_loss_real = criterion(outputs, real_labels)


                # Train Discriminator on fake images.
                outputs = self.discriminator(fake_imgs.detach(), labels)
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                total_discriminator_loss += d_loss.item()

                # Train Generator.

                outputs = self.discriminator(fake_imgs, labels)
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
                print(
                    f"Epoch [{epoch}/{num_epochs}], Loss G: {generator_loss:.4f}, Loss D: {discriminator_loss:.4f}")
                self.save_samples(epoch, batch_size, labels)

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
    ])

    mnist_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_dl = DataLoader(mnist_ds, batch_size=100, shuffle=True)

    # Train the GAN.

    gan = cGAN(latent_dim=50, num_classes=10)
    history = gan.train(dataloader=mnist_dl, num_epochs=150, lr=0.0002)
    plot_history(history)

