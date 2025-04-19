import torch
import torch.nn as nn
import torch.nn.functional as F


class simple_cnn(nn.Module):
    def __init__(self):
        super(simple_cnn, self).__init__()
        # First convolution layer: 3 input channels, 32 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # Second convolution layer: 32 input channels, 64 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer: 1600 input features (from 5x5x64), 512 output units
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        # Output layer: 512 input features, 10 output classes
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor: from (batch_size, 64, 5, 5) to (batch_size, 1600)
        x = x.view(-1, 64 * 5 * 5)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # FC2
        x = self.fc2(x)
        # Softmax across the class dimension (dim=1)
        x = F.softmax(x, dim=1)
        return x


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 10)
 
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 5)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

class VGG9(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG9, self).__init__()
        self.features = nn.Sequential(
            # 第一模块：Conv + ReLU + MaxPool
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸：32 -> 16

            # 第二模块：Conv + ReLU + MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸：16 -> 8

            # 第三模块：连续两个卷积层 + ReLU，然后 MaxPool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸：8 -> 4
        )
        # 对于 CIFAR10 图片，特征图尺寸为 4×4，最后一层通道数为256
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # 将张量展平，送入全连接层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(input_channels=1, num_classes=10):
    """
    创建CNN模型的工厂函数
    
    Args:
        input_channels: 输入图像的通道数，默认为1（灰度图像）
        num_classes: 分类类别数，默认为10
        
    Returns:
        CNN模型实例
    """
    
    model = Mnist_CNN()
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")
    
    return model