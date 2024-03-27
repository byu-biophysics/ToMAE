"""
Author: TJ Hart
Date: 3/2024

This file contains utility functions for the tiling scripts.  





"""
import numpy as np
import mrcfile


def robust_normalize_batch(batch_images):
    """
    Normalize a batch of images by clipping the 1st and 99th percentile of pixel values and scaling to 0-255.
    Note think of a tomogram as a batch of images where the batch dimension corresponds to the depth dimension.

    Parameters
    ----------
    batch_images (np.array): shape (batch_size, height, width, channels)

    Returns
    -------
    np.ndarray of shape (batch_size, height, width, channels) with pixel values scaled to 0-255
    """
    # 1st and 99th percentile of pixel values
    p01, p99 = np.percentile(batch_images, [1, 99], axis=(0, 1)) 

    # Keep pixel values in between the 1st and 99th percentile
    clipped_batch = np.clip(batch_images, p01[None, None, :], p99[None, None, :]) 

    # Normalize pixel values to 0-255
    normalized_batch = ((clipped_batch - p01[None, None, :]) / (p99[None, None, :] - p01[None, None, :])) * 255 
    return normalized_batch.astype(np.uint8)


def crop_3d(arr, cube_size):
    """
    Crop a 3D array to a multiple of cube_size in each dimension

    Parameters
    ----------
    arr (np.array): The 3D array to be cropped in with shape (d, w, h)
    cube_size (tuple): The size of each cube, or "tile", i.e. (64, 64, 64)
    
    Returns
    -------
    A 3D array, cropped to a multiple of cube_size in each dimension
    i.e. if arr.shape = (300, 964, 956) and cube_size = (64, 64, 64), 
    the output will be (256, 960, 896) where 256 = 4*64, 960 = 15*64, 896 = 14*64
        
    """
    def get_crop_sizes(arr_shape, cube_size):
        """ 
        Calculate the crop amounts needed in each dimension

        Parameters
        ----------
        arr_shape: The shape of the array to be cropped
        cube_size: The size of each cube, or "tile"

        Returns
        -------
        crop_start: The starting index of the stuff to keep in the array
        crop_end: The ending index of the stuff to keep in the array
        """
        # Calculate crop amounts needed in each dimension
        crop = arr_shape % cube_size
        if crop == 0:
            crop_start = 0
            crop_end = arr_shape
        else:
        # Crop the array from both ends equally (hence the negative for crop_end, for negative indexing)
            crop_start = crop // 2
            crop_end = -(crop - crop_start) 
        return crop_start, crop_end

    sizes = [get_crop_sizes(t[0],t[1]) for t in zip(arr.shape, cube_size)]

    cropped_arr = arr[
        sizes[0][0]:sizes[0][1],
        sizes[1][0]:sizes[1][1],
        sizes[2][0]:sizes[2][1]
    ]
    return cropped_arr

def tile_3d(tomo, SZ):
    """
    This function creates cubes of the 3D image (that is already divisible by SZ). 
    This function returns a list with a dictionary for each cube.
    The dictionary conatains the "chunk" number, the cube (np array) 
    and a location which corresponds to the location of the cube in 
    the 3D image before it was cubed.
    
    Parameters
    -----------
    tomo (np.array): A tomogram represented as a 3D numpy array where the dimensions are (depth, width, height)
                    and every element in the depth dimension is a 2D slice/image of the tomogram
    SZ (int): cube size i.e. if SZ = 64, the cube will be 64 x 64 x 64
    
    Returns
    -----------
    result (list of dicts): This is a list of dictionaries corresponding to each cube
            where each dictionary has the following keys:
            tomo (np.array): A smaller cube (think "tile") of the original 3D image 
                            of size SZ x SZ x SZ
            idx (int): The cube index number
            location (tuple(d_loc,w_loc,h_loc)): this is a list of three integers corresponding 
                                            to the original location of the cube in the 3D image
                                            i.e. (2, 0, 1) would be located at the 2nd depth, 
                                            0th width, and 1st height spot of the original 3D tomogram
    """

    # Get initial variables
    result = []

    # Reshape cubes to a x SZ x b x SZ x c x SZ (a, b, and c depend on the original size of image)
    tomo = tomo.reshape(tomo.shape[0] // SZ, SZ, tomo.shape[1] // SZ, SZ, tomo.shape[2] // SZ, SZ)
    n_cubes_d = tomo.shape[0] # Number of cubes in the depth dimension
    n_cubes_w = tomo.shape[2] # "    "   "   "   "   " width dimension
    n_cubes_h = tomo.shape[4] # "    "   "   "   "   " height dimension

    # Swap order of dimensions
    tomo = tomo.transpose(0, 2, 4, 1, 3, 5)

    tomo = tomo.reshape(-1, SZ, SZ, SZ)

    # Get cube locations in original image
    # Think of ndindex as a way to get all possible combinations of indices
    idxs = [ind for ind in np.ndindex(tuple([n_cubes_d, n_cubes_w, n_cubes_h]))]

    # Add dictionary to list
    for i in range(len(tomo)):
        result.append({'tomo': tomo[i], 'idx': i, 'location': idxs[i]})

    return result


def crop_and_split_array(arr, cube_size,tile_path='../../images/3d_tiles/'):
    """
    Crops a 3D array to be evenly divisible by the chunk size and then splits it into chunks.

    Parameters
    ----------
    arr: The 3D array to be cropped and split.
    cube_size: A tuple (x, y, z) representing the size of each chunk.

    Returns
    -------
    A list of 3D arrays, each representing a chunk.
    """
    arr = robust_normalize_batch(arr)
    cropped_arr = crop_3d(arr, cube_size)
    # Split the cropped array into chunks
    dict = tile_3d(cropped_arr, cube_size[0])
    # For each element in the dictionary save it as a .npy file and name it with the location
    for i,chunk in enumerate(dict):
        np.save(f'{tile_path}chunk_{i}_{chunk["location"]}',chunk['tomo'])
    print(f"Saved all chunks as .npy files in {tile_path}")

