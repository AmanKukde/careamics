import numpy as np
from typing import Tuple, Union, Callable
from skimage.transform import resize # You'll need this import

# Import the necessary classes
from .config import DatasetConfig
from .multich_dataset import MultiChDloader
from .types import DataSplitType, TilingMode
import math
from careamics.lvae_training.dataset.utils.windowed_tiling_manager import WindowedTilingGridIndexManager

class WindowedTilingLCMultiChDloader(MultiChDloader):
    """
    A dataset class that inherits from MultiChDloader but implements a
    different tiling strategy. It pads the entire image first, and then
    extracts patches using a sliding window with a specified stride.

    This approach is designed for inference where all pixels need to be
    covered, and it avoids the complex boundary handling of the parent class.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the dataset. The actual data loading is handled by the
        parent class. After the data is loaded, it is immediately padded.
        """
        self.original_data_shape = None
        self._padded_data = None
        self.pad_width_spatial = None
        
        # We need to extract grid_size before calling super, as it's used for padding.
        # This assumes the config object is the first argument.
        data_config: DatasetConfig = args[0]
        self._grid_sz = data_config.grid_size

        super().__init__(*args, **kwargs)

        # After super().__init__(), self._data is loaded. Now we can pad it.
        self.original_data_shape = self._data.shape
        # self._pad_data()

    def _pad_data(self, grid_size):
        """
        Pads the loaded image data. The padding amount is 1.5 times the
        grid_size on each side of the spatial dimensions.
        """
        if grid_size is None:
            raise ValueError("`grid_size` must be set in the config to calculate padding.")

        # Handle both integer (2D) and tuple (3D) grid sizes.
        spatial_grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        # Calculate padding width for each side of each spatial dimension.
        # self.pad_width_spatial = [(int(1.5 * s)-1, int(1.5 * s)-1) for s in spatial_grid_size]
        self.pad_width_spatial = [(int(1.5 * s), int(1.5 * s)) for s in spatial_grid_size]

        # self.pad_width_spatial = [(s,s) for s in spatial_grid_size]

        # Construct the full padding tuple for np.pad, with no padding on N and C dims.
        pad_width_full = [(0, 0)]  # N dimension
        pad_width_full.extend(self.pad_width_spatial)
        pad_width_full.append((0, 0))  # C dimension

        print(f"Original data shape: {self._data.shape}")
        print(f"Padding spatial dimensions with: {self.pad_width_spatial}")

        self._padded_data = np.pad(self._data, pad_width=pad_width_full, mode='constant', constant_values=255)
        print(f"Padded data shape: {self._padded_data.shape}")

    def set_img_sz(self, image_size, grid_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = None):
        """
        Overrides the parent method to set up the SlidingWindowIndexManager.

        Args:
            image_size: The size of the patch to extract (patch_size).
            grid_size: Used to confirm padding, but not for tiling.
            stride: The stride of the sliding window.
        """
        
            
        self._img_sz = image_size[-1] if isinstance(image_size, tuple) else image_size
        self._grid_sz = grid_size
        self.stride, self.stride_value = self.get_stride_length(grid_size, stride)
        self.stride, self.stride_value = ((1,8,8,1), 8)

        print(f"Image size: {self._img_sz}")
        print(f"Grid size: {self._grid_sz}")
        print(f"Stride: {self.stride}")

        self._pad_data(self._grid_sz)

        numC = self._padded_data.shape[-1]
        is_3d = len(self._padded_data.shape) == 5

        # Define patch and stride shapes based on data dimensionality.
        if is_3d:
            self.patch_shape = (1, self._depth3D, self._img_sz, self._img_sz, numC)
            self.stride_shape = (1, *(self.stride if isinstance(self.stride, tuple) else (1, self.stride, self.stride)), numC)
        else:
            self.patch_shape = (1, self._img_sz, self._img_sz, numC)
            self.stride_shape = (1, self.stride_value, self.stride_value, numC)

        self.idx_manager = WindowedTilingGridIndexManager(
            data_shape=self.original_data_shape,
            grid_shape=self._grid_sz,
            tiling_mode= TilingMode.WindowedTiling,
            padded_data_shape=self._padded_data.shape,
            patch_shape=self.patch_shape,
            stride=self.stride_shape,
        )
        print(f"Number of patches per dimension: {self.idx_manager.grid_counts_for_flat_idx}")
        print(f"Successfully Initialized Windowed Tiling with {self.idx_manager.total_patch_count()} patches. !!!")

    def get_stride_length(self, grid_size, stride = None):
        if stride is None or stride == 0:
            # Assuming grid_size is an int here. If it's a tuple, you'll need to adjust this.
            if isinstance(grid_size, int):
                stride_value = math.ceil(grid_size / math.sqrt(50))
                stride = (1, stride_value, stride_value, 1)
            elif isinstance(grid_size, tuple):
                # Example: take first element of tuple to compute stride
                stride_value = math.ceil(grid_size[0] / math.sqrt(50))
                stride = (1, stride_value, stride_value, 1)
            else:
                raise ValueError("grid_size must be an int or a tuple")
        return stride, stride_value

    def __len__(self):
        """Returns the total number of patches in the dataset."""
        return self.idx_manager.total_patch_count()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets a single patch for prediction.

        This method overrides the parent's complex __getitem__. It performs a
        simple slice from the pre-padded image based on the index. It does
        not perform augmentations, noise addition, or alpha blending.
        """
        # Get the top-left coordinate of the patch from the index manager.
        patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(index)
        n_idx, *spatial_loc = patch_loc

        # Define the slice ranges for the patch extraction.
        patch_spatial_dims = self.idx_manager.patch_spatial_dims
        slices = []
        for loc, dim in zip(spatial_loc, patch_spatial_dims):
            slices.append(slice(int(loc), int(loc + dim)))

        # Extract the patch from the padded data.
        if self._5Ddata: # (N, Z, H, W, C)
            patch = self._padded_data[n_idx, slices[0], slices[1], slices[2], :]
        else: # (N, H, W, C)
            patch = self._padded_data[n_idx, slices[0], slices[1], :]

        # For prediction, input and target are the same.
        # The output format must be (C, [Z], H, W) for the model.
        inp = np.transpose(patch, (2, 0, 1)) if not self._5Ddata else np.transpose(patch, (3, 0, 1, 2))
        
        # The dataloader in eval_utils expects a tuple of (input, target).
        return inp.astype(np.float32), inp.copy().astype(np.float32)
