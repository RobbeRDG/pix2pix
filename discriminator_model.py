import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        # The architecture uses a conv, bachnorm and relu structure
        # Padding mode and bias stated in paper
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)



# In channels * 2 becaus both in and output of the generator is passed to the discriminator
class Discriminator(nn.Module):
    # - The 70 × 70 discriminator architecture is: C64-C128-C256-C512
    # - Final layer uses stride of 1, rest stride of 2
    # - After the last layer, a convolution is applied to map to a 1-dimensional output, followed by a Sigmoid function.
    # - BatchNorm is not applied to the first C64 layer. All ReLUs are leaky, with slope 0.2.
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # Set the first layer without BN
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # Set layer 2 to 4 
        layers = []
        inbetween_in_channel = 64

        for feature in features[1:]:
            layers.append(
                CNNBlock(inbetween_in_channel, feature, stride= 1 if feature == features[-1] else 2)
            )
            inbetween_in_channel = feature

        self.model = nn.Sequential(*layers)

        # Set the final layer
        self.final = nn.Sequential(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Make the input for the discriminator two long lists
        x = torch.cat([x,y], dim=1)

        # Send the input through the layers of the model
        x = self.initial(x)
        x = self.model(x)
        x = self.final(x)

        return x

# testcase to test the discriminator
def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    pred = model(x,y)
    print(pred.shape)

if __name__ == "__main__":
    test()
