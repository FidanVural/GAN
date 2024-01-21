import torch
import torch.nn as nn

# DISCRIMINATOR
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels+1, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            # we use in_channels*2 because we're gonna send both x and y as an input
            # we're gonna concatenate these along the channels
            nn.LeakyReLU(0.2),
        )

        in_channel = features[0]
        layers = []

        for feature in features[1:]:

            layers.append(
                CNNBlock(in_channel, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channel = feature

        layers.append(
            nn.Conv2d(in_channel, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1) # N x C x H x W
        x = self.initial(x)
        return self.model(x)


# GENERATOR

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, encoder=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if encoder else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.encoder_down1 = Block(features, features*2, encoder=True, act="leaky", use_dropout=False) # 64
        self.encoder_down2 = Block(features*2, features*4, encoder=True, act="leaky", use_dropout=False) # 32
        self.encoder_down3 = Block(features*4, features*8, encoder=True, act="leaky", use_dropout=False) # 16
        self.encoder_down4 = Block(features*8, features*8, encoder=True, act="leaky", use_dropout=False) # 8
        self.encoder_down5 = Block(features*8, features*8, encoder=True, act="leaky", use_dropout=False) # 4
        self.encoder_down6 = Block(features*8, features*8, encoder=True, act="leaky", use_dropout=False) # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1),
            nn.ReLU(),
        )

        self.decoder_up1 = Block(features*8, features*8, encoder=False, act="relu", use_dropout=True)
        self.decoder_up2 = Block(features*8*2, features*8, encoder=False, act="relu", use_dropout=True)
        self.decoder_up3 = Block(features*8*2, features*8, encoder=False, act="relu", use_dropout=False)
        self.decoder_up4 = Block(features*8*2, features*8, encoder=False, act="relu", use_dropout=False)
        self.decoder_up5 = Block(features*8*2, features*4, encoder=False, act="relu", use_dropout=False)
        self.decoder_up6 = Block(features*4*2, features*2, encoder=False, act="relu", use_dropout=False)
        self.decoder_up7 = Block(features*2*2, features, encoder=False, act="relu", use_dropout=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.initial(x)
        e2 = self.encoder_down1(e1)
        e3 = self.encoder_down2(e2)
        e4 = self.encoder_down3(e3)
        e5 = self.encoder_down4(e4)
        e6 = self.encoder_down5(e5)
        e7 = self.encoder_down6(e6)

        bottleneck = self.bottleneck(e7)

        d1 = self.decoder_up1(bottleneck)
        d2 = self.decoder_up2(torch.cat([d1, e7], 1))
        d3 = self.decoder_up3(torch.cat([d2, e6], 1))
        d4 = self.decoder_up4(torch.cat([d3, e5], 1))
        d5 = self.decoder_up5(torch.cat([d4, e4], 1))
        d6 = self.decoder_up6(torch.cat([d5, e3], 1))
        d7 = self.decoder_up7(torch.cat([d6, e2], 1))

        return self.final(torch.cat([d7, e1], 1))

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # The isinstance() function returns True if the specified object is of the specified type, otherwise False.
            # I mean if the m is the type of nn.Conv2d, nn.ConvTranspose2d and nn.BatchNorm2d then returns True.
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
"""
def do_test_encoder():
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 3, 256, 256))

    disc = Discriminator()
    preds = disc(x, y)

    print(preds.shape)

def do_test_decoder():
    x = torch.randn((1, 1, 256, 256))

    gen = Generator(in_channels=1, features=64)
    preds = gen(x)

    print(preds.shape)

if __name__ == "__main__":
    do_test_encoder()
    do_test_decoder()
"""

