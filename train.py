from ctypes import util
from email import generator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils import load_checkpoint, save_checkpoint, save_image, save_some_examples
from dataset import MapDataset
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(
            discriminator, 
            generator, 
            optim_discriminator, 
            optim_generator, 
            scaler_discriminator, 
            scaler_generator, 
            loader, 
            l1_loss, 
            bce_loss
):
    loop = tqdm(loader, leave=True)
    for idx, (x,y) in enumerate(loop):
        # Send the data to the device
        x,y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train the discriminator
        with torch.cuda.amp.autocast():
            # Generate our "prediction"
            y_fake = generator(x)

            # Send both real and fake output to the discriminator
            d_real = discriminator(x, y)
            d_fake = discriminator(x, y_fake.detach())

            # Calculate the loss in the discriminator
            d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_total_loss = (d_real_loss + d_fake_loss) / 2
        
        # Backpropagate
        discriminator.zero_grad() # same as optim_discriminator.zero_grad()
        scaler_discriminator.scale(d_total_loss).backward()
        scaler_discriminator.step(optim_discriminator)
        scaler_discriminator.update()

        # Train the generator
        with torch.cuda.amp.autocast():
            d_fake = discriminator(x, y_fake)
            g_fake_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            g_l1_loss = l1_loss(y_fake, y) * config.L1_LAMBDA
            g_total_loss = g_fake_loss + g_l1_loss
        
        # Backpropagate
        optim_generator.zero_grad()
        scaler_generator.scale(g_total_loss).backward()
        scaler_generator.step(optim_generator)
        scaler_generator.update()




def main():
    # Create the discriminator and corresponding optimizer
    discriminator = Discriminator(
        in_channels= 3
    ).to(config.DEVICE)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Create the generator and corresponding optimizer
    generator = Generator(
        in_channels= 3
    ).to(config.DEVICE)
    optim_generator = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE,betas=(0.5, 0.999))


    # Create the loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # If we start from a previous checkpoint
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, discriminator, optim_discriminator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, generator, optim_generator, config.LEARNING_RATE)

    # Get the data and data loaders
    train_dataset = MapDataset("data/train/")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = MapDataset("data/val/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Set the gradient scalers
    scaler_discriminator = torch.cuda.amp.grad_scaler.GradScaler()
    scaler_generator = torch.cuda.amp.grad_scaler.GradScaler()

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            discriminator, 
            generator, 
            optim_discriminator, 
            optim_generator, 
            scaler_discriminator, 
            scaler_generator, 
            train_loader, 
            l1_loss, 
            bce_loss
        )

        # Save checkpoints and examples from time to time
        if config.SAVE_MODEL and epoch % 5 == 0:
            # Checkpoints
            save_checkpoint(discriminator, optim_discriminator, config.CHECKPOINT_DISC)
            save_checkpoint(generator, optim_generator, config.CHECKPOINT_GEN)
        
            # Examples
            save_some_examples(generator, val_loader, epoch, folder="evaluation")
        


if __name__ == "__main__":
    main()