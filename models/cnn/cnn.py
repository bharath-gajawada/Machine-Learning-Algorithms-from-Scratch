# 2.2 & 2.4
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, task='classification', num_classes=4, input_channels=1, num_conv_layers=4, dropout=0.5):
        super(CNN, self).__init__()

        self.task = task
        assert num_conv_layers in [3, 4, 5], "num_conv_layers must be 3, 4, or 5"

        channels = [input_channels] + [4, 8, 16, 16, 16][:num_conv_layers]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            for i in range(num_conv_layers)
        ])

        final_map_size = 128 // (2 ** num_conv_layers)
        final_num_channels = channels[-1]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_num_channels * final_map_size * final_map_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes if task == 'classification' else 1)
        )

    def forward(self, x):
        feature_maps = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            feature_maps.append(x)

        x = self.fc_layers(x)
        return x, feature_maps
    