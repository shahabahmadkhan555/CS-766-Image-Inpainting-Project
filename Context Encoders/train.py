import os
import argparse
import numpy as np
import math

from datasets.datasets import *
from models.context_encoder import *

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("Checkpoints", exist_ok=True)
os.makedirs("training_images/samples", exist_ok=True)
os.makedirs("training_images/training_samples", exist_ok=True)
os.makedirs("training_images/generated_masks", exist_ok=True)
os.makedirs("training_images/without_gen_image/training_samples", exist_ok=True)

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training") # context encoder paper uses significantly more
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches") # context encoder paper uses 64 default
    parser.add_argument("--dataset_name", type=str, default="miniplaces", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_l1", type=float, default=0.5, help="joint reconstruction: weight of L1 loss")
    parser.add_argument("--lambda_ssim", type=float, default=0.5, help="joint reconstruction: weight of SSIM loss")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument("--resume", type=str, default='', help="path to model checkpoint to resume training from")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints") # Set to -1 to not save

    # Try warmup batches that only train with pixelwise loss (no discriminator)?
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs before discriminator is trained")
    parser.add_argument("--warmup_batches", type=int, default=12000, help="Number of epochs before discriminator is trained")

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = load_args()

    cuda = True if torch.cuda.is_available() else False

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(args.mask_size / 2 ** 3), int(args.mask_size / 2 ** 3)
    patch = (1, patch_h, patch_w)


    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Loss functions:
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    # adversarial_loss = torch.nn.MSELoss()

    # pixelwise_loss = torch.nn.MSELoss() # L2 Loss
    pixelwise_loss = torch.nn.L1Loss()
    # Try nn.BCELoss() for adversarial and weighted_mse_loss for pixel
    # or implement wasserstein loss for adversarial?

    # from torchmetrics.image import StructuralSimilarityIndexMeasure
    # def get_ssim(gen_img, img):
    #     SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)
    #     return SSIM(gen_img, img)
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0) # images are normalized

    # Initialize generator and discriminator:
    generator = Generator(channels=args.channels)
    discriminator = Discriminator(channels=args.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()
        SSIM.cuda()

    # Check if we can load a checkpoint:
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if cuda:
                checkpoint = torch.load(args.resume)
            else:
                # If no gpu is available, open with cpu (just in case, but you really don't want to use this)
                checkpoint = torch.load(args.resume, map_location=torch.devide('cpu'))

            args.start_epoch = checkpoint['epoch'] + 1
            generator.load_state_dict(checkpoint["generator_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            print("Loaded generator and discriminator from checkpoint")

            # Optimizers
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
            print("Created optimizers from checkpoint")
            optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D"])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        print("Initialized generator and discriminator without checkpoint")
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        print("Created optimizers without checkpoint")


    # Dataset loader
    transforms_ = [
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # .../miniplaces/images/{train,test,val}
    traindir = os.path.join(args.dataset_name, 'images/train')
    valdir = os.path.join(args.dataset_name, 'images/val')
    dataloader = DataLoader(
        ImageDataset("%s" % traindir, transforms_=transforms_),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
    )
    print("Created dataloader")
    test_dataloader = DataLoader(
        ImageDataset("%s" % valdir, transforms_=transforms_, mode="val"),
        batch_size=12,
        shuffle=True,
        num_workers=1,
    )
    print("Created test dataloader")

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Doesn't quite work:
    # def show_tensor(img, title=""):
    #     img = img.cpu().detach().numpy()
    #     print("image shape:", img.shape)
    #     img = np.transpose(img, (2, 3, 1, 0))[:, :, :, 0]
    #     plt.figure()
    #     plt.title(title)
    #     plt.imshow(img, cmap='gray', interpolation='nearest')
    #     plt.show()

    def save_sample(epoch, batches_done):
        samples, masked_samples, i = next(iter(test_dataloader))
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask

        # Generate inpainted image
        gen_img = generator(masked_samples)
        gen_mask = gen_img[:, :, i : i + args.mask_size, i : i + args.mask_size]
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + args.mask_size, i : i + args.mask_size] = gen_mask

        # Save sample
        sample = torch.cat((samples.data, masked_samples.data, filled_samples.data), -2)
        save_image(sample, "training_images/samples/sample_epoch%d_batch%d.png" % (epoch, batches_done), nrow=6, normalize=True) # Set nrow=batch size? was originally 6


    # ----------
    #  Training
    # ----------

    for epoch in range(args.start_epoch, args.n_epochs):
        for i, (imgs, masked_imgs, masked_parts, x1s, y1s) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            generator.train() # set to training mode

            # Generate a batch of images
            gen_image = generator(masked_imgs)
            gen_parts = Variable(Tensor(gen_image.shape[0], gen_image.shape[1], args.mask_size, args.mask_size).fill_(0.0), requires_grad=False)

            # Find the generated parts of each image output by the generator:
            for idx in range(gen_image.shape[0]): # gen_image.shape[0] = batch_size
                x1, y1 = x1s[idx], y1s[idx]
                y2, x2 = y1 + args.mask_size, x1 + args.mask_size
                gen_parts[idx, :, :, :] = gen_image[idx, :, y1:y2, x1:x2]

            # show_tensor(imgs, title="Original Images")
            # show_tensor(masked_imgs, title="Masked Images")
            # show_tensor(gen_image, title="Generated image")
            # show_tensor(gen_parts, title="Generated parts")

            # if batches_done < args.warmup_batches:
            #     # Warm-up (pixel-wise loss only)
            #     g_pixel = pixelwise_loss(gen_parts, masked_parts)
            #     g_pixel.backward()
            #     optimizer_G.step()
            #     print(
            #         "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
            #         % (epoch, args.n_epochs, i, len(dataloader), g_pixel.item())
            #     )
            #     continue

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            g_ssim = SSIM(gen_parts, masked_parts) # get_ssim(gen_parts, masked_parts)

            # Total loss
            # Note: To use SSIM as a loss function, use (1 - SSIM) since a larger ssim value is good, but we want to minimize error
            # => Maximizing SSIM = Minimizing 1 - SSIM
            # g_loss = 0.001 * g_adv + 0.999 * (0.3 * g_pixel + 0.7 * (1 - g_ssim))     #1
            # g_loss = 0.001 * g_adv + 0.999 * (0.005 * g_pixel + 0.995 * (1 - g_ssim)) #2
            # g_loss = 0.001 * g_adv + 0.999 * (0.5 * g_pixel + 0.5 * (1 - g_ssim))     #3
            g_loss = 0.001 * g_adv + 0.999 * (args.lambda_l1 * g_pixel + args.lambda_ssim * (1 - g_ssim))
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            discriminator.train() # set to training mode

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f, ssim: %f, loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item(), g_ssim.item(), g_loss.item())
            )

            # Generate sample at sample interval:
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                # Set to evaluation mode before sampling test images:
                generator.eval()
                save_sample(epoch, batches_done)
                generator.train()

                # Fill in each masked image with it's generated portion:
                inpainted_imgs = masked_imgs.clone()
                for idx in range(args.batch_size): # gen_image.shape[0]
                    # Get upper left corner of random rectangular mask:
                    x1, y1 = x1s[idx], y1s[idx]
                    y2, x2 = y1 + args.mask_size, x1 + args.mask_size
                    # Apply generated mask
                    gen_mask = gen_parts[idx, :, :, :]
                    inpainted_imgs[idx, :, y1:y2, x1:x2] = gen_mask

                # Set nrow=args.batch_size to get all samples on same row
                sample = torch.cat((imgs.data, masked_imgs.data, gen_image.data, inpainted_imgs.data), -2)
                save_image(sample, "training_images/training_samples/full_mask_epoch%d_batch%d.png" % (epoch, batches_done), nrow=int(args.batch_size / 2), normalize=True)
                save_image(gen_parts, "training_images/generated_masks/gen_parts_epoch%d_batch%d.png" % (epoch, batches_done), nrow=int(args.batch_size / 2), normalize=True)

                sample = torch.cat((imgs.data, masked_imgs.data, inpainted_imgs.data), -2)
                save_image(sample, "training_images/without_gen_image/training_samples/inpainted_epoch%d_batch%d.png" % (epoch, batches_done), nrow=int(args.batch_size / 2), normalize=True)

            # End of current batch:

        # End of Epoch:

        # Save model at the checkpoint interval:
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            torch.save({
              "epoch": epoch,
              "generator_state_dict": generator.state_dict(),
              "discriminator_state_dict": discriminator.state_dict(),
              "optimizer_G": optimizer_G.state_dict(),
              "optimizer_D": optimizer_D.state_dict(),
            }, "Checkpoints/model_epoch_{}.pth".format(epoch))

    # # Save model after final epoch:
    # torch.save({
    #   "epoch": epoch,
    #   "generator_state_dict": generator.state_dict(),
    #   "discriminator_state_dict": discriminator.state_dict(),
    #   "optimizer_G": optimizer_G.state_dict(),
    #   "optimizer_D": optimizer_D.state_dict(),
    # }, "Checkpoints/model_epoch_{}.pth".format(args.n_epochs - 1))

if __name__ == '__main__':
    main()
