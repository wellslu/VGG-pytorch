import mlconfig
from torch import nn

vgg_selector = {11:((3, 64, 1), (64, 128, 1), (128, 256, 2), (256, 512, 2), (512, 512, 2)),
               13:((3, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 512, 2)),
               16:((3, 64, 2), (64, 128, 2), (128, 256, 3), (256, 512, 3), (512, 512, 3)),
               19:((3, 64, 2), (64, 128, 2), (128, 256, 4), (256, 512, 4), (512, 512, 4))}

@mlconfig.register
class VGG(nn.Module):

    def __init__(self, arch):
        super(VGG, self).__init__()
        arch = vgg_selector[arch]
        layers = []
        for a in arch:
            layers+=self.ConvBNReLU_Block(a)
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x = nn.AdaptiveAvgPool2d((7,7))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def ConvBNReLU_Block(self, arch):
        layer = []
        in_channels, out_channels, conv_num = arch
        for i in range(conv_num):
            layer.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1))
            # layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layer.append(nn.MaxPool2d(2, 2))
        return layer
