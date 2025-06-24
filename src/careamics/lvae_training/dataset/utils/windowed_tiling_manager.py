import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union
from careamics.lvae_training.dataset.types import TilingMode
from careamics.lvae_training.dataset.utils.index_manager import GridIndexManager 


@dataclass
class WindowedTilingGridIndexManager(GridIndexManager):
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
        self.grid_counts_for_flat_idx = [self.padded_data_shape[0]] + self.n_patches_per_dim
        self._strides_for_flat_idx = self._calculate_strides_for_lookup(self.grid_counts_for_flat_idx)

    def _calculate_strides_for_lookup(self,grid_counts_per_dim) -> list:
        """
        Calculates strides needed to convert a flat index to a multi-dimensional grid index.
        This is for internal use to quickly find a patch's location from its index.
        """
        strides = [1] * len(grid_counts_per_dim) # Use the corrected list
        for i in range(len(grid_counts_per_dim) - 2, -1, -1):
            strides[i] = strides[i + 1] * grid_counts_per_dim[i + 1] # Use the corrected list
        return strides

    def total_patch_count(self) -> int:
        """Returns the total number of patches that can be extracted."""
        return int(np.prod(self.grid_counts_for_flat_idx))

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
        
        for i in range(len(self.grid_counts_for_flat_idx)): # Iterate only over N + spatial dims
            grid_idx = remaining_index // self._strides_for_flat_idx[i]
            grid_indices.append(grid_idx)
            remaining_index %= self._strides_for_flat_idx[i]

        # First index is for N dimension.
        n_coord = grid_indices[0]

        # Convert spatial grid indices to pixel coordinates using the stride.
        # Now, grid_indices[1:] directly corresponds to the spatial dimensions (Z, H, W or H, W).
        spatial_coords = []
        for i, grid_idx in enumerate(grid_indices[1:]): # Iterate over spatial grid indices
            coord = grid_idx * self.stride_spatial[i]
            spatial_coords.append(coord)

        return (n_coord, *spatial_coords)