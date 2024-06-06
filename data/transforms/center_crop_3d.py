from torch import Tensor

class CenterCrop3D: 
    def __init__(self, crop_size: int):
        self.crop_size = crop_size
    
    def __call__(
            self,
            tensor: Tensor
        ):
        x_size, y_size, z_size = tensor.size()
        x_size, y_size, z_size = (x_size - self.crop_size) // 2, (y_size - self.crop_size) // 2, (z_size - self.crop_size) // 2
        return tensor[x_size:-x_size, y_size:-y_size, z_size:-z_size] 
