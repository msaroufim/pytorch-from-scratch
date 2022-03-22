from torch import nn

# A Convolutional Network 
class ConvNeuralNet(nn.Module):
	
    #  Solve a classification problem with num_classes
    def __init__(self, num_classes : int):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):

        # 2 Convolutional layers followed by pooling layer
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        # Another 2 Convolutional layers followed by pooling layer
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        
        # Reshape to flat 1D tensor so we can feed this to fully connected layer
        out = torch.flatten(out, 1)
        
        # Linear layer
        out = self.fc1(out)

        # Non linearity otherwise 2 linear layers are as expressive as a single one
        out = self.relu1(out)
        
        # Final non linearity
        out = self.fc2(out)

        return out

# Great but now how you implement a custom convolutional or pooling layer from scratch

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int):
        return NotImplemented

class MaxPool(nn.Module):
    def __init__(self, kernel_size : int, stride : int):
        return NotImplemented