import os
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader  # Folder of images
from PIL import Image

from models.context_encoder import *
from models.segmentor_model import *
from datasets.datasets import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='', help="Path to image")
    parser.add_argument("--image_folder_path", type=str, default='', help="Path to folder of input images")
    # parser.add_argument("--batch_size", type=int, default=64, help="size of the batches (grid when evaluating folder of images)")
    parser.add_argument("--batch_size", type=int, default=-1, help="size of the batches (grid when evaluating folder of images)")
    parser.add_argument("--num_cols", type=int, default=-1, help="Number of images per column in output grid (use with folder of images)")

    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")

    parser.add_argument("--model_checkpoint", type=str, default='', help="name of the model checkpoint file to use for evaluation")
    parser.add_argument('--remove', nargs= '*' ,type=int, help='objects to remove')
    # parser.add_argument('--remove', default=[0, 2], nargs= '*' ,type=int, help='objects to remove') # Remove people and Cars
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")

    args = parser.parse_args()
    print(args)
    return args

args = load_args()

os.makedirs("inpainting_results", exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Load the generator from the given checkpoint:
generator = Generator(channels=args.channels)
if cuda:
    generator.cuda()
    checkpoint = torch.load(args.model_checkpoint)
    generator.load_state_dict(checkpoint["generator_state_dict"])
else:
    # If using the cpu, specify map_location to load tensors in cpu form:
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint["generator_state_dict"])
generator.eval()

# Create image and mask input transformations:
mask_transforms_ = [
    transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
    transforms.ToTensor()
]
mask_transforms = transforms.Compose(mask_transforms_) # Callable transformation
eval_transforms_ = [
    transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
eval_transforms = transforms.Compose(eval_transforms_) # Callable transformation

# Define loss functions:
pixelwise_loss_L2 = torch.nn.MSELoss() # L2 Loss
pixelwise_loss_L1 = torch.nn.L1Loss()  # L1 loss
loss_functions = [("L2 Loss", torch.nn.MSELoss()), ("L1 Loss", torch.nn.L1Loss())]

# PSNR = 10 * log_10( (max_dtype_value)^2 / MSE(gen_img, img))
def get_psnr(gen_img, img):
    mse = torch.mean((gen_img - img) ** 2)
    if(mse == 0):  # If there is no noise / difference between the two images
        return 100
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

# SSIM:
def get_ssim(gen_img, img):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0) # images are normalized
    return SSIM(gen_img, img)

# Multi-scale SSIM (currently gives NaN, so it is not in use):
def get_ms_ssim(gen_img, img):
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    # Default kernel_size=11 is too big for 128x128 images
    MS_SSIM = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=7, normalize=None)
    return MS_SSIM(gen_img, img)

from torchmetrics.image import StructuralSimilarityIndexMeasure
SSIM_LossFn = StructuralSimilarityIndexMeasure(data_range=1.0) # images are normalized
if cuda:
    SSIM_LossFn.cuda()

# By default, apply center masking. If objects to remove are specified, detect the given objects and delete them
if args.image_folder_path: # Works for folder of jpgs (also recursively takes images in subfolders)
    # Get number of images in input folder:
    num_images = len(glob.glob("%s/**/*.jpg" % args.image_folder_path, recursive=True))
    print("Number of input images:", num_images)

    if args.batch_size == -1:
        args.batch_size = num_images

    # If the number of rows is not specified, take the sqrt of the number of images
    # to produce a square grid
    if args.num_cols == -1:
        import math
        args.num_cols = int(math.sqrt(args.batch_size))

    # Load the input image folder:
    test_dataloader = DataLoader(
      ImageDataset(args.image_folder_path, transforms_=eval_transforms_, mode="val"),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=0,
    )

    # Initialize loss counters
    total_L2 = 0
    total_L1 = 0
    total_PSNR = 0
    total_SSIM = 0

    for batch_idx, (samples, masked_samples, i) in enumerate(test_dataloader):
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask

        # Generate inpainted images:
        gen_img = generator(masked_samples)
        gen_mask = gen_img[:, :, i : i + args.mask_size, i : i + args.mask_size]
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + args.mask_size, i : i + args.mask_size] = gen_mask

        '''Save Output Images (currently, for first batch):'''
        if batch_idx == 0:
            ''' Save
              (1) Grid of input, masked input, and inpainted output
              (2) Grid of input, masked input, generated images, and inpainted output
              (3) Grid of input images
              (4) Grid of masked input images
              (5) Grid of the resulting inpainted images
            '''
            # Save grid of input, masked input, and inpainted output:
            sample = torch.cat((samples.data, masked_samples.data, filled_samples.data), -2)
            save_image(sample, "inpainting_results/inpainted_grid.png", nrow=8, normalize=True)
            # Save grid of input, masked input, generated image, and inpainted output:
            sample = torch.cat((samples.data, masked_samples.data, gen_img.data, filled_samples.data), -2)
            save_image(sample, "inpainting_results/inpainted_grid_with_gen_img.png", nrow=8, normalize=True)
            # Save grid of input images:
            save_image(samples, "inpainting_results/original_grid.png", nrow=8, normalize=True)
            # Save grid of masked input images:
            save_image(masked_samples, "inpainting_results/masked_grid.png", nrow=8, normalize=True)
            # Save grid of inpainted images:
            save_image(filled_samples, "inpainting_results/inpainted_images_grid.png", nrow=8, normalize=True)

        # Get losses for each individual image (top to bottom, left to right):
        for idx, (sample, filled_sample) in enumerate(zip(samples, filled_samples)):
            image_idx = batch_idx * args.batch_size + idx

            # Reshape into BxCxHxW (B = 1 since the batch size is a single image here)
            # This is needed for torch SSIM loss
            sample = sample.unsqueeze(0)
            filled_sample = filled_sample.unsqueeze(0)

            # Calculate losses for the current image:
            L2_Loss = pixelwise_loss_L2(filled_sample, sample)
            L1_Loss = pixelwise_loss_L1(filled_sample, sample)
            PSNR_Loss = get_psnr(filled_sample, sample)
            SSIM_Loss = SSIM_LossFn(filled_sample, sample) # get_ssim(filled_sample, sample)
            #MS_SSIM_Loss = get_ms_ssim(filled_sample, sample)
            print("==> Calculating Reconstruction Loss {}:".format(image_idx))
            print("L2: {0:.4f}".format(L2_Loss.item()))
            print("L1: {0:.4f}".format(L1_Loss.item()))
            print("PSNR: {0:.4f}".format(PSNR_Loss.item())) # NOTE THAT PSNR AND L2 ARE RELATED
            print("SSIM: {0:.4f}".format(SSIM_Loss.item()))
            #print("MS-SSIM: {0:.4f}".format(MS_SSIM_Loss.item()))

            total_L2 += L2_Loss.item()
            total_L1 += L1_Loss.item()
            total_PSNR += PSNR_Loss.item()
            total_SSIM += SSIM_Loss.item()
        
    # Calculate average loss:
    mean_L2 = total_L2 / num_images
    mean_L1 = total_L1 / num_images
    mean_PSNR = total_PSNR / num_images
    mean_SSIM = total_SSIM / num_images
    print("==> Calculating average loss statistics:")
    print("Mean L1: {0:.4f}".format(mean_L1))
    print("Mean L2: {0:.4f}".format(mean_L2))
    print("Mean PSNR: {0:.4f}".format(mean_PSNR))
    print("Mean SSIM: {0:.4f}".format(mean_SSIM))

elif args.remove is None:
    img = Image.open(args.image_path)
    img = eval_transforms(img)
    """Mask center part of image"""
    # Get upper-left pixel coordinate
    i = (args.img_size - args.mask_size) // 2
    masked_img = img.clone()
    masked_img[:, i : i + args.mask_size, i : i + args.mask_size] = 1

    # unsqueeze 0 to add extra dimension at position 0 (needed for generator)
    # (Reshape to BxCxHxW)
    img = Variable(img.type(Tensor)).unsqueeze(0) 
    masked_img = Variable(masked_img.type(Tensor)).unsqueeze(0)

    # Generated the inpainted image:
    gen_img = generator(masked_img)
    gen_mask = gen_img[:, :, i : i + args.mask_size, i : i + args.mask_size]
    inpainted_img = masked_img.clone()
    inpainted_img[:, :, i : i + args.mask_size, i : i + args.mask_size] = gen_mask

    # Calculate loss metrics between original image and output image:
    L2_Loss = pixelwise_loss_L2(inpainted_img, img)
    L1_Loss = pixelwise_loss_L1(inpainted_img, img)
    PSNR_Loss = get_psnr(inpainted_img, img)
    SSIM_Loss = SSIM_LossFn(inpainted_img, img)
    print("==> Calculating Reconstruction Loss:")
    print("L1: {0:.4f}".format(L1_Loss.item()))
    print("L2: {0:.4f}".format(L2_Loss.item()))
    print("PSNR: {0:.4f}".format(PSNR_Loss.item())) # NOTE THAT PSNR AND L2 ARE RELATED
    print("SSIM: {0:.4f}".format(SSIM_Loss.item()))

    # print("==> Calculating Reconstruction Loss:")
    # for (name, pixelwise_loss) in loss_functions:
    #     loss = pixelwise_loss(inpainted_img, img)
    #     print("{}: {}".format(name, loss.item()))

    # Save sample
    sample = torch.cat((img.data, masked_img.data, inpainted_img.data), 0)
    output_file = args.image_path.split("/")[-1]
    save_image(sample, "inpainting_results/inpainted_%s" % output_file, nrow=3, normalize=True) #nrow=3 since the sample has 3 images

else: # If we are detecting / removing objects:
    
    # Detect objects and extract mask:
    img = np.array(Image.open(args.image_path))
    seg = Segmentor(args.image_path)
    mask = seg.get_mask(args.remove)
    
    if mask is None:
        print("==> No objects detected. Exiting program.")
    else:
        mask = mask.detach().cpu().numpy().astype(float)

        # Resize image to mask size:
        import cv2
        mask_height, mask_width = mask.shape[:2]
        resized_img = cv2.resize(img, (mask_width, mask_height), interpolation=cv2.INTER_CUBIC)
        masked_img = resized_img.copy()
        masked_img[mask != 0] = 255

        # Convert back into PIL Images before applying torch transforms:
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        masked_img = Image.fromarray(masked_img)

        # Apply transforms to input image and mask:
        img = eval_transforms(img)
        mask = mask_transforms(mask)
        masked_img = eval_transforms(masked_img)

        # Convert to Tensors and reshape from CxHxW to BxCxHxW:
        img = Variable(img.type(Tensor)).unsqueeze(0)
        mask = Variable(mask.type(Tensor)).unsqueeze(0)
        masked_img = Variable(masked_img.type(Tensor)).unsqueeze(0)

        # Generate image on masked image:
        gen_img = generator(masked_img)

        # Convert mask to have 3D channels:
        mask = torch.cat((mask, mask, mask), dim=1)

        # Find generated mask and inpaint this area onto a cloned version of the masked image:
        mask_indices = (mask != 0).nonzero(as_tuple=True) # get new indices since they are tensors with extra dim
        gen_mask = gen_img[mask_indices]
        inpainted_img = masked_img.clone()
        inpainted_img[mask_indices] = gen_mask

        # NOTE: Evaluating loss and reconstruction metrics won't work here
        # since we have no ground truth image.

        # Save sample:
        output_file = args.image_path.split("/")[-1]
        sample = torch.cat((img.data, masked_img.data, inpainted_img.data), 0)
        save_image(sample, "inpainting_results/inpainted_%s" % output_file, nrow=3, normalize=True)
        sample = torch.cat((img.data, masked_img.data, gen_img.data, inpainted_img.data), 0)
        save_image(sample, "inpainting_results/inpainted_with_gen_img%s" % output_file, nrow=4, normalize=True)
        save_image(mask.data, "inpainting_results/inpainted_mask%s" % output_file, nrow=1, normalize=True)
