'''
   Copyright 2020 Sailist

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
import torch.nn as nn
from torch.nn.utils import weight_norm


class CNN13(nn.Module):
    """only used in cifar10"""

    def __init__(self, dropout=0.0):
        super(CNN13, self).__init__()

        # self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(dropout)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc = nn.Sequential(nn.Linear(128, 128),
                                nn.PReLU(),
                                nn.Linear(128, 128),
                                )
        # self.fc1 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=0.1, layers_mix=None):
        out = x
        ## layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        ## layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        ## layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)

        out = self.mp1(out)
        out = self.drop1(out)

        ## layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        ## layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        ## layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)

        out = self.mp2(out)
        out = self.drop2(out)

        ## layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        ## layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        ## layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)

        out = self.ap3(out)

        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    print(CNN13())
