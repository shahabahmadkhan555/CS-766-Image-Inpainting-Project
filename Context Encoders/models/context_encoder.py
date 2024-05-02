import torch.nn as nn
import torch.nn.functional as F
import torch

# ContextGenerator:
class Generator(nn.Module):
    def __init__(self, channels=3, bottleneck_dim=4000):
        super(Generator, self).__init__()

        # Basic Block:
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # Transpose Block:
        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        # Must add final layer *upsample(64, 64) to resize to 128
        self.ContextEncoder = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),        # needed if img size = 128
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, bottleneck_dim, 1) 
        )
        # Previous fully-connected conv2d layer is the model's bottleneck layer.
        self.ContextDecoder = nn.Sequential(
            *upsample(bottleneck_dim, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            *upsample(64, 64),          # needed to resize back to 128 x 128
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ContextEncoder(x)
        return self.ContextDecoder(x)

# ContextDiscriminator
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
