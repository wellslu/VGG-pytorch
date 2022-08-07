import mlconfig
from torch import nn

vgg_selector = {11:((64, 1), (128, 1), (256, 2), (512, 2), (512, 2)),
               13:((64, 2), (128, 2), (256, 2), (512, 2), (512, 2)),
               16:((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)),
               19:((64, 2), (128, 2), (256, 4), (512, 4), (512, 4))}

@mlconfig.register
class VGG(nn.Module):

    def __init__(self, arch):
        super(VGG, self).__init__()
        arch = vgg_selector[arch]
        layers = []
        self.in_channels = 3
        for i, a in enumerate(arch):
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
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = nn.AdaptiveAvgPool2d((7,7))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def ConvBNReLU_Block(self, arch):
        layer = []
        out_channels, conv_num = arch
        for i in range(conv_num):
            layer.append(nn.Conv2d(self.in_channels, out_channels, (3, 3), padding=1))
            # layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
            self.in_channels = out_channels
        layer.append(nn.MaxPool2d(2, 2))
        return layer
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)