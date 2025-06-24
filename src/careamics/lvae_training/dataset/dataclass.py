import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, Callable

# Assuming the following imports are available in your environment
# You might need to adjust the paths based on your project structure.
from careamics.lvae_training.dataset.multich_dataset import MultiChDloader
from careamics.lvae_training.dataset.config import DatasetConfig
from careamics.lvae_training.dataset.types import TilingMode

#
# Part 1: New Index Manager for Sliding Window Tiling
#

@dataclass
class SlidingWindowIndexManager:
    """
    Manages patch indices for a sliding window approach with defined strides.

    This index manager calculates the total number of patches and their
    locations based on the shape of a pre-padded image, the desired
    patch size, and the stride of the window.

    Attributes:
    ----------
    padded_data_shape (tuple): The shape of the padded source image (e.g., N, Z, H, W, C).
    patch_shape (tuple): The shape of the patches to extract.
    stride (tuple): The stride to use when moving the window across spatial dimensions.
    """
    padded_data_shape: tuple
    patch_shape: tuple
    stride: tuple

    def __post_init__(self):
        """
        Initializes attributes and calculates the number of patches.
        """
        # We only apply striding to spatial dimensions (all but first and last).
        self.spatial_dims = self.padded_data_shape[1:-1]
        self.patch_spatial_dims = self.patch_shape[1:-1]
        self.stride_spatial = self.stride[1:-1]

        # Validate that dimensions are compatible.
        assert len(self.spatial_dims) == len(self.patch_spatial_dims) == len(self.stride_spatial)
        for i, (dim, patch_dim) in enumerate(zip(self.spatial_dims, self.patch_spatial_dims)):
            if patch_dim > dim:
                raise ValueError(
                    f"Patch dimension {i} ({patch_dim}) cannot be larger than "
                    f"the padded image dimension ({dim})."
                )

        # Calculate the number of patches for each spatial dimension.
        self.n_patches_per_dim = []
        for i in range(len(self.spatial_dims)):
            padded_dim = self.spatial_dims[i]
            patch_dim = self.patch_spatial_dims[i]
            stride_dim = self.stride_spatial[i]
            # Standard formula for strided convolutions/windows.
            num_patches = np.floor((padded_dim - patch_dim) / stride_dim).astype(int) + 1
            self.n_patches_per_dim.append(num_patches)

        # Non-spatial dims (N, C) are not tiled, their count is their size.
        self.grid_counts_per_dim = [self.padded_data_shape[0]] + self.n_patches_per_dim + [self.padded_data_shape[-1]]
        self._strides_for_flat_idx = self._calculate_strides_for_lookup()

    def _calculate_strides_for_lookup(self) -> list:
        """
        Calculates strides needed to convert a flat index to a multi-dimensional grid index.
        This is for internal use to quickly find a patch's location from its index.
        """
        strides = [1] * len(self.grid_counts_per_dim)
        for i in range(len(self.grid_counts_per_dim) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.grid_counts_per_dim[i + 1]
        return strides

    def total_patch_count(self) -> int:
        """Returns the total number of patches that can be extracted."""
        return int(np.prod(self.grid_counts_per_dim))

    def get_patch_location_from_dataset_idx(self, index: int) -> tuple:
        """
        Converts a flat dataset index to the top-left coordinate of the patch
        in the padded image.

        Args:
            index (int): The flat index of the patch in the dataset.

        Returns:
            A tuple representing the location, e.g., (N_idx, Z_coord, H_coord, W_coord).
        """
        if index >= self.total_patch_count():
            raise IndexError(f"Index {index} is out of bounds for total patches {self.total_patch_count()}")

        grid_indices = []
        remaining_index = index
        for i in range(len(self.grid_counts_per_dim)):
            grid_idx = remaining_index // self._strides_for_flat_idx[i]
            grid_indices.append(grid_idx)
            remaining_index %= self._strides_for_flat_idx[i]

        # First and last indices are for N and C dimensions.
        n_coord = grid_indices[0]

        # Convert spatial grid indices to pixel coordinates using the stride.
        spatial_coords = []
        for i, grid_idx in enumerate(grid_indices[1:-1]):
            coord = grid_idx * self.stride_spatial[i]
            spatial_coords.append(coord)

        return (n_coord, *spatial_coords)

#
# Part 2: New Dataset Class with Pre-Padding and Windowed Tiling
#

class WindowedTilingDataset(MultiChDloader):
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
        self._pad_data()

    def _pad_data(self):
        """
        Pads the loaded image data. The padding amount is 1.5 times the
        grid_size on each side of the spatial dimensions.
        """
        if self._grid_sz is None:
            raise ValueError("`grid_size` must be set in the config to calculate padding.")

        # Handle both integer (2D) and tuple (3D) grid sizes.
        spatial_grid_size = (self._grid_sz, self._grid_sz) if isinstance(self._grid_sz, int) else self._grid_sz

        # Calculate padding width for each side of each spatial dimension.
        self.pad_width_spatial = [(int(1.5 * s), int(1.5 * s)) for s in spatial_grid_size]

        # Construct the full padding tuple for np.pad, with no padding on N and C dims.
        pad_width_full = [(0, 0)]  # N dimension
        pad_width_full.extend(self.pad_width_spatial)
        pad_width_full.append((0, 0))  # C dimension

        print(f"Original data shape: {self._data.shape}")
        print(f"Padding spatial dimensions with: {self.pad_width_spatial}")

        self._padded_data = np.pad(self._data, pad_width=pad_width_full, mode='reflect')
        print(f"Padded data shape: {self._padded_data.shape}")

    def set_img_sz(self, image_size, grid_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]]):
        """
        Overrides the parent method to set up the SlidingWindowIndexManager.

        Args:
            image_size: The size of the patch to extract (patch_size).
            grid_size: Used to confirm padding, but not for tiling.
            stride: The stride of the sliding window.
        """
        self._img_sz = image_size[-1] if isinstance(image_size, list) else image_size
        self._stride_sz = stride
        
        # Ensure grid_size hasn't changed, as padding depends on it.
        assert self._grid_sz == grid_size, "Changing grid_size after initialization is not supported."

        numC = self._padded_data.shape[-1]
        is_3d = len(self._padded_data.shape) == 5

        # Define patch and stride shapes based on data dimensionality.
        if is_3d:
            patch_shape = (1, self._depth3D, self._img_sz, self._img_sz, numC)
            stride_shape = (1, *(stride if isinstance(stride, tuple) else (1, stride, stride)), numC)
        else:
            patch_shape = (1, self._img_sz, self._img_sz, numC)
            stride_shape = (1, stride, stride, numC)

        self.idx_manager = SlidingWindowIndexManager(
            padded_data_shape=self._padded_data.shape,
            patch_shape=patch_shape,
            stride=stride_shape,
        )
        print(f"Initialized SlidingWindowIndexManager with {self.idx_manager.total_patch_count()} patches.")

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

#
# Part 3: New Stitching Function
#

def stitch_and_crop_predictions(predictions: np.ndarray, dset: WindowedTilingDataset) -> np.ndarray:
    """
    Stitches predicted patches from a sliding window and crops the result
    back to the original image size. Overlapping regions are averaged.

    Args:
        predictions: A batch of predicted patches, shape (num_patches, C, [Z], H, W).
        dset: The WindowedTilingDataset object used, which contains padding info.

    Returns:
        The final stitched and cropped image, in (N, [Z], H, W, C) format.
    """
    if not isinstance(dset, WindowedTilingDataset):
        raise TypeError("This stitching function requires a WindowedTilingDataset instance.")

    padded_shape = dset._padded_data.shape
    idx_manager = dset.idx_manager

    # Create a canvas for the stitched padded image and a counter for averaging.
    stitched_padded_image = np.zeros((predictions.shape[1], *padded_shape[1:-1]), dtype=predictions.dtype)
    counts = np.zeros_like(stitched_padded_image)
    
    # Pre-create a patch of ones for efficient counting.
    patch_ones = np.ones(predictions[0].shape, dtype=predictions.dtype)

    for i in range(len(predictions)):
        patch = predictions[i]
        
        # Get the location of this patch in the padded image canvas.
        loc = idx_manager.get_patch_location_from_dataset_idx(i)
        _, *spatial_loc = loc

        # Define slices for placing the patch on the canvas.
        slices = [slice(None)] # Channel slice
        for s_loc, s_dim in zip(spatial_loc, idx_manager.patch_spatial_dims):
            slices.append(slice(int(s_loc), int(s_loc + s_dim)))

        # Add the patch to the canvas and increment the count for averaging.
        stitched_padded_image[tuple(slices)] += patch
        counts[tuple(slices)] += patch_ones

    # Average the overlapping regions.
    counts[counts == 0] = 1  # Avoid division by zero.
    stitched_padded_image /= counts

    # Crop the stitched image back to the original size using stored padding info.
    pad_width = dset.pad_width_spatial
    crop_slices = [slice(None)] # Channel slice
    for pad_before, pad_after in pad_width:
        crop_slices.append(slice(pad_before, -pad_after if pad_after > 0 else None))

    final_image = stitched_padded_image[tuple(crop_slices)]
    
    # Transpose back to original data format (e.g., H, W, C)
    is_3d = len(dset.original_data_shape) == 5
    axes_order = (1, 2, 3, 0) if is_3d else (1, 2, 0)
    final_image_transposed = np.transpose(final_image, axes_order)

    # Add back the N dimension and return.
    return final_image_transposed[np.newaxis, ...]

