from torch import Tensor
import random
import torch

class RandomSlice3D:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.chunk_radius = self.chunk_size // 2

    def __call__(
        self,
        tensor: Tensor
    ):
        x,y,z = tensor.size()

        x_min, x_max = self.__get_rand_slice(int(x))
        y_min, y_max = self.__get_rand_slice(int(y))
        z_min, z_max = self.__get_rand_slice(int(z))

        return tensor[x_min:x_max, y_min:y_max, z_min:z_max]
        
    def __get_rand_slice(self, dimension_size):
        selected_chunk = random.randint(self.chunk_radius, dimension_size-self.chunk_radius)

        minimum_chunk_idx = selected_chunk - self.chunk_radius
        maximum_chunk_idx = selected_chunk + self.chunk_radius
        return int(minimum_chunk_idx), int(maximum_chunk_idx)

