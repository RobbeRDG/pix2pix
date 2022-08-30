import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = "relu", use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5) # Specified in paper

    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


# - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
# - Decoder: CD512-CD512-CD512-C512-C256-C128-C64
# - After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general, except in colorization, where it is 2),
#   followed by a Tanh function. 
# - BatchNorm is not applied to the first C64 layer in the encoder.
# - All ReLUs in the encoder are leaky, with slope 0.2, ReLUs in the decoder are not leaky.

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        # Batchnorm is not applied to the first layer of the encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.05)
        ) # 128 x 128 feature map

        # The rest of the layers in the encoder
        self.down2 = Block(features, features * 2, down=True, act="leaky", use_dropout=False) # 64
        self.down3 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 32
        self.down4 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 16
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4
        self.down7 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 2

        # Final layer of encoder uses normal relu (bottleneck)
        self.down8 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        ) # 1

        # the decoder in UNET architecure
        # *2 in the input channels is needed for the copy and concatinate functionality
        # First 3 layers use dropout
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True) # 2
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True) # 4
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True) # 8

        # Layers 4-7 use no dropout
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False) # 16
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False) # 32
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False) # 64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False) # 128

        # Final layer of decoder
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, 4, 2, 1),
            nn.Tanh() # Get output values between 0 and 1
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.down8(d7)

        # Decoder
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


# testcase to test the generator
def test():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features=64)
    pred = model(x)
    print(pred.shape)

if __name__ == "__main__":
    test()



