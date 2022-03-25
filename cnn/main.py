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
    
    def forward(self, x):
        return NotImplemented
        subset = ...
        x = torch.mm(subset, filter)
        x = torch.sum(x)
    
    def convolve(img : torch.Tensor, kernel : torch.Tensor) -> torch.Tensor:
        tgt_size = calculate_target_size(img.shape[0], kernel.shape[0])
        k = kernel.shape[0]

        convolved_img = torch.zeros(tgt_size, tgt_size)

        for i in range(tgt_size):
            for j in range(tgt_size):
                mat = img[i:i+k, j:j+k]
                convolved_img[i, j] = torch.sum(torch.mm(mat, kernel))
        
        return convolved_img

    @staticmethod
    def calculate_target_size(img_size : int, kernel_size : int) -> int:
        # Assume square matrix
        # Doesn't support strides yet
        # Doesn't support padding
        num_pixels = 0
        for i in range(img_size):
            added = i + kernel_size
            if added <= img_size:
                num_pixels += 1
        return num_pixels
    
    @staticmethod
    def add_padding_to_image(img : torch.Tensor, padding_width : int) -> torch.Tensor:
        img_with_padding = torch.zeros(img.shape[0] + padding_width * 2,
                                       img.shape[0] + padding_width * 2)
        img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img 
        return img_with_padding


class MaxPool(nn.Module):
    def __init__(self, conv_output, kernel_size : int, stride : int):
        return NotImplemented
    
    def get_pools(self, pool_size : int = 2, stride : int = 2) -> torch.Tensor:
        pools = []
        for i in torch.arange(self.conv_output.shape, step = self.stride):
            for j in torch.arange(self.conv_output.shape[0], step = self.stride):
                mat = self.conv_output[i: i + pool_size, j:j+pool_size]
                pools.append(mat)
        return pools
    
    def max_pooling(self, pools : torch.Tensor) -> torch.Tensor:
        num_pools = pools.shape[0]
        tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
        pooled = []

        for pool in pools:
            # Can add any pooling mechanism here
            pools.append(np.max(pool))
        
        return torch.Tensor(pooled).reshape(tgt_shape)


