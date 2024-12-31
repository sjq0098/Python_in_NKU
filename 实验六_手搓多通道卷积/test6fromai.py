import torch

def perform_convolution(Image, Kernel):
    # Perform convolution operation
    output = torch.nn.functional.conv2d(Image, Kernel, padding=1)
    print(output.shape)
    print(output)
    return output
