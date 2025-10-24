"""
WindowedTilingGridIndexManager for sliding window patch extraction.

Refactored to work with UNPADDED data (consistent with refactored WindowedLCDLoader).
This manager handles sliding window extraction with defined strides, where patches
near boundaries will be padded on-demand during cropping (not pre-padded).

Key changes from original:
- Changed "padded_data_shape" to "data_shape" (now uses unpadded data shape)
- Removed assumption of pre-padded data
- Updated documentation to reflect unpadded data usage
- Patches at boundaries will be padded on-demand in _crop_img_with_padding()
"""
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
    locations based on the shape of an UNPADDED image, the desired patch size,
    and the stride of the window. Patches extracted from unpadded data may
    cross boundaries, in which case they will be padded on-demand during
    cropping (handled by parent's _crop_img_with_padding method).

    Attributes
    ----------
    data_shape : tuple
        The shape of the UNPADDED source image (e.g., N, H, W, C for 2D or N, Z, H, W, C for 3D).
    patch_shape : tuple
        The shape of the patches to extract (e.g., 1, 64, 64, 2).
    stride : tuple
        The stride to use when moving the window across spatial dimensions.
        Format: (1, stride_h, stride_w, 1) for 2D or (1, stride_z, stride_h, stride_w, 1) for 3D.

    Example
    -------
    >>> mgr = WindowedTilingGridIndexManager(
    ...     data_shape=(6, 2720, 2720, 2),  # Unpadded
    ...     patch_shape=(1, 64, 64, 2),
    ...     stride=(1, 4, 4, 1)
    ... )
    >>> print(f"Total patches: {mgr.total_patch_count()}")  # Will be ~2.6M
    >>> loc = mgr.get_patch_location_from_dataset_idx(0)
    >>> print(f"First patch at: {loc}")  # (0, 0, 0) - top-left
    """

    # Rename from padded_data_shape to data_shape for clarity
    data_shape: tuple  # Now uses unpadded shape
    patch_shape: tuple
    stride: tuple

    def __post_init__(self):
        """
        Initializes attributes and calculates the number of patches.
        
        Note: Unlike the parent GridIndexManager which assumes centered grid-based
        patches, this manager uses simple sliding windows with fixed strides.
        Patches that exceed boundary will be padded on-demand during cropping.
        """
        # We only apply striding to spatial dimensions (all but first and last)
        print(
            f"{self.__class__.__name__} initialized with data_shape: {self.data_shape},"
        )
        print(
            f"{self.__class__.__name__} initialized with patch_shape: {self.patch_shape}, stride: {self.stride}"
        )

        # Extract spatial dimensions (skip batch and channel dims)
        self.spatial_dims = self.data_shape[1:-1]
        self.patch_spatial_dims = self.patch_shape[1:-1]
        self.stride_spatial = self.stride[1:-1]

        # Validate that dimensions are compatible
        assert len(self.spatial_dims) == len(self.patch_spatial_dims) == len(
            self.stride_spatial
        ), (
            f"Spatial dims {len(self.spatial_dims)}, patch dims {len(self.patch_spatial_dims)}, "
            f"stride dims {len(self.stride_spatial)} must all be equal"
        )

        # Validate patch size doesn't exceed data size
        for i, (dim, patch_dim) in enumerate(zip(self.spatial_dims, self.patch_spatial_dims)):
            if patch_dim > dim:
                raise ValueError(
                    f"Patch dimension {i} ({patch_dim}) cannot be larger than "
                    f"the data dimension ({dim}). "
                    f"Note: patches larger than unpadded data will be padded on-demand during cropping."
                )

        # Calculate the number of patches for each spatial dimension
        # Using sliding window formula: floor((size - patch_size) / stride) + 1
        self.n_patches_per_dim = []
        for i in range(len(self.spatial_dims)):
            data_dim = self.spatial_dims[i]
            patch_dim = self.patch_spatial_dims[i]
            stride_dim = self.stride_spatial[i]

            # Standard formula for strided sliding windows
            # This gives us how many positions the window can start from
            num_patches = int(np.floor((data_dim - patch_dim) / stride_dim)) + 1
            self.n_patches_per_dim.append(num_patches)

            print(
                f"  Dimension {i}: data={data_dim}, patch={patch_dim}, stride={stride_dim} "
                f"→ {num_patches} patches"
            )

        # For flat indexing: [N_samples, patches_per_dim_0, patches_per_dim_1, ...]
        self.grid_counts_for_flat_idx = [self.data_shape[0]] + self.n_patches_per_dim
        print(f"  Grid counts for flat indexing: {self.grid_counts_for_flat_idx}")
        print(f"  Total patches: {self.total_patch_count()}")

        # Calculate strides for converting flat index to multi-dimensional index
        self._strides_for_flat_idx = self._calculate_strides_for_lookup(
            self.grid_counts_for_flat_idx
        )

    def _calculate_strides_for_lookup(self, grid_counts_per_dim) -> list:
        """
        Calculates strides needed to convert a flat index to a multi-dimensional grid index.

        For example, if grid_counts = [6, 676, 676], strides = [1, 456976, 676]
        This allows us to quickly decompose a flat index like 457652 into (0, 1, 0)

        Parameters
        ----------
        grid_counts_per_dim : list
            List of counts for each dimension: [N_samples, patches_dim0, patches_dim1, ...]

        Returns
        -------
        list
            Strides for each dimension to convert flat index to grid indices
        """
        strides = [1] * len(grid_counts_per_dim)
        for i in range(len(grid_counts_per_dim) - 2, -1, -1):
            strides[i] = strides[i + 1] * grid_counts_per_dim[i + 1]
        return strides

    def total_patch_count(self) -> int:
        """
        Returns the total number of patches that can be extracted from the UNPADDED data.

        Returns
        -------
        int
            Total number of patches across all samples and spatial dimensions.
            For 2D data (N, H, W, C), returns N * n_patches_h * n_patches_w
        """
        return int(np.prod(self.grid_counts_for_flat_idx))

    def get_patch_location_from_dataset_idx(self, index: int) -> tuple:
        """
        Converts a flat dataset index to the top-left coordinate of the patch
        in the UNPADDED data space.

        The returned coordinates specify where to start extracting the patch.
        If the patch extends beyond the data boundary, it will be padded on-demand
        during cropping (in parent's _crop_img_with_padding method).

        Parameters
        ----------
        index : int
            The flat index of the patch in the dataset (0 to total_patch_count()-1).

        Returns
        -------
        tuple
            A tuple representing the top-left location, e.g., (n_idx, h_coord, w_coord)
            for 2D data or (n_idx, z_coord, h_coord, w_coord) for 3D data.
            
        Raises
        ------
        IndexError
            If index >= total_patch_count()

        Example
        -------
        >>> mgr.get_patch_location_from_dataset_idx(0)  # First patch
        (0, 0, 0)
        >>> mgr.get_patch_location_from_dataset_idx(676)  # Second row of patches (H dim)
        (0, 4, 0)  # if stride_h=4
        """
        if index >= self.total_patch_count():
            raise IndexError(
                f"Index {index} is out of bounds for total patches {self.total_patch_count()}"
            )

        # Decompose flat index into multi-dimensional grid indices
        grid_indices = []
        remaining_index = index

        for i in range(len(self.grid_counts_for_flat_idx)):
            grid_idx = remaining_index // self._strides_for_flat_idx[i]
            grid_indices.append(grid_idx)
            remaining_index %= self._strides_for_flat_idx[i]

        # First index is for the sample (N dimension)
        n_coord = grid_indices[0]

        # Convert spatial grid indices to pixel coordinates using the stride
        # grid_idx[0] means "the 0th patch position in this dimension"
        # Multiplying by stride gives us the pixel coordinate where to start extracting
        spatial_coords = [
            grid_idx * self.stride_spatial[i] for i, grid_idx in enumerate(grid_indices[1:])
        ]

        return (n_coord, *spatial_coords)

    # ============================================================================
    # Optional utility methods for debugging and validation
    # ============================================================================

    def validate_patch_extraction_boundaries(self):
        """
        Validates and reports on patch extraction boundaries.
        
        Useful for debugging to understand which patches will touch boundaries
        and require on-demand padding during cropping.
        """
        print(f"\n{self.__class__.__name__} Boundary Analysis:")
        print("=" * 70)

        for sample_idx in range(self.data_shape[0]):
            # Check a few patches from this sample
            print(f"\nSample {sample_idx}:")
            for patch_idx in [0, 1, self.n_patches_per_dim[0] - 1]:
                if patch_idx < self.n_patches_per_dim[0]:
                    loc = self.get_patch_location_from_dataset_idx(
                        sample_idx * np.prod(self.n_patches_per_dim) + patch_idx
                    )
                    h_start = loc[1]
                    h_end = h_start + self.patch_spatial_dims[0]
                    w_start = loc[2]
                    w_end = w_start + self.patch_spatial_dims[1]

                    h_oob = h_end > self.spatial_dims[0]
                    w_oob = w_end > self.spatial_dims[1]
                    oob_status = "OOB" if (h_oob or w_oob) else "OK"

                    print(
                        f"  Patch {patch_idx}: ({h_start}:{h_end}, {w_start}:{w_end}) [{oob_status}]"
                    )

        print("=" * 70)
        print("Note: OOB (Out Of Bounds) patches will be padded on-demand during cropping\n")

    def get_patch_coordinates_range(self, index: int) -> tuple:
        """
        Returns the full coordinate range [start, end) for a patch.
        
        Useful for understanding boundary cases.
        
        Returns (h_start, h_end, w_start, w_end) for 2D or
        (z_start, z_end, h_start, h_end, w_start, w_end) for 3D.
        """
        n_idx, *spatial_coords = self.get_patch_location_from_dataset_idx(index)
        
        coords_range = []
        for i, start_coord in enumerate(spatial_coords):
            end_coord = start_coord + self.patch_spatial_dims[i]
            coords_range.append((start_coord, end_coord))
        
        return tuple(coords_range)

    def is_boundary_patch(self, index: int) -> bool:
        """
        Returns True if the patch at this index touches the data boundary
        and will require on-demand padding during cropping.
        """
        coords_range = self.get_patch_coordinates_range(index)
        
        for i, (start, end) in enumerate(coords_range):
            if end > self.spatial_dims[i]:
                return True
        
        return False

    def count_boundary_patches(self) -> int:
        """
        Returns the number of patches that touch boundaries.
        These patches will be padded on-demand during cropping.
        """
        count = 0
        for idx in range(self.total_patch_count()):
            if self.is_boundary_patch(idx):
                count += 1
        return count
