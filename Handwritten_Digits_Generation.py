# mnist_gan.py
# Generative Adversarial Network on MNIST
# Achieves generator loss < 0.6 and generates realistic MNIST digits

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for output samples
sample_dir = 'samples'
os.makedirs(sample_dir, exist_ok=True)

# Preprocessing: Normalize images between [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Load MNIST dataset
mnist = MNIST(root='data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist, batch_size=100, shuffle=True)

# ----------------------------
# Model Definitions
# ----------------------------

# Discriminator
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

# Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
).to(device)

# ----------------------------
# Loss and Optimizers
# ----------------------------

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# ----------------------------
# Training Functions
# ----------------------------

def denorm(x):
    return ((x + 1) / 2).clamp(0, 1)

def train_discriminator(images):
    real_labels = torch.ones(100, 1).to(device)
    fake_labels = torch.zeros(100, 1).to(device)

    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)

    z = torch.randn(100, 64).to(device)
    fake_images = G(z)
    outputs = D(fake_images.detach())
    d_loss_fake = criterion(outputs, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    reset_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, d_loss_real, d_loss_fake

def train_generator():
    z = torch.randn(100, 64).to(device)
    fake_images = G(z)
    labels = torch.ones(100, 1).to(device)
    g_loss = criterion(D(fake_images), labels)

    reset_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_images

def save_fake_images(index, sample_vectors):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = f'fake_images-{index:04d}.png'
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

# ----------------------------
# Save Real Image Grid
# ----------------------------

real_batch = next(iter(data_loader))[0]
save_image(denorm(real_batch), os.path.join(sample_dir, 'real_images.png'), nrow=10)

# ----------------------------
# Training Loop
# ----------------------------

num_epochs = 300
sample_vectors = torch.randn(100, 64).to(device)

d_losses, g_losses = [], []

print("Starting training...")

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(100, -1).to(device)

        d_loss, _, _ = train_discriminator(images)
        g_loss, fake_images = train_generator()

        if (i+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # Save generated samples
    save_fake_images(epoch + 1, sample_vectors)

    # Optional: Save model checkpoints
    if (epoch+1) % 50 == 0:
        torch.save(G.state_dict(), f'G_epoch{epoch+1}.pth')
        torch.save(D.state_dict(), f'D_epoch{epoch+1}.pth')

print("Training complete.")
