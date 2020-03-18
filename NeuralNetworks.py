import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(input=self.conv1(x))
        print(x.size())
        x = F.max_pool2d(input=x, kernel_size=(2, 2))
        print(x.size())
        x = F.relu(input=self.conv2(x))
        print(x.size())
        x = F.max_pool2d(input=x, kernel_size=(2, 2))
        print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = F.relu(self.fc3(x))
        print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())# conv1's .weight

print(params[0])
print("==============")

input = torch.randn(1, 1, 32, 32)
out = net(input)

print(out)
print("==============")
