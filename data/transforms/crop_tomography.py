from torch import Tensor

class CropTomography:
    def __init__(self, crop_count: int):
        self.crop_count = crop_count
        
    def __call__(
            self,
            tensor: Tensor
        ):
        return tensor[:,self.crop_count:-self.crop_count,:]
