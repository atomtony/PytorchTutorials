import torchvision.models as models
from torchsummary import summary

mobilenet = models.mobilenet_v2()
summary(mobilenet, input_size=(3, 224, 224), device='cpu')
