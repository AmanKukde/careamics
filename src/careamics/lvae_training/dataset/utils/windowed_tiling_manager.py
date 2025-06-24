import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union
from careamics.lvae_training.dataset.types import TilingMode
from careamics.lvae_training.dataset.utils.index_manager import GridIndexManager 


@dataclass
class WindowedTilingGridIndexManager(GridIndexManager):
    """
    A specialized GridIndexManager for WindowedTiling mode.

    This class overrides the default GridIndexManager behavior for WindowedTiling
    to implement specific padding and fixed-stride tiling.
    """
    # No new attributes are strictly needed here, as _fixed_stride and _padding
    # are inherited, but we will manage them specifically for WindowedTiling.

    def __post_init__(self):
        # Call the parent's __post_init__ for basic assertions and default initialization
        super().__post_init__()

        # Only apply custom logic if the tiling_mode is WindowedTiling for this instance
        if self.tiling_mode == TilingMode.WindowedTiling:
            # Validate patch and grid shapes - still relevant for conceptual inner tile
            innerpad_check = np.array(self.patch_shape) - np.array(self.grid_shape)
            for dim, pad in enumerate(innerpad_check):
                if pad < 0:
                    raise ValueError(
                        f"Patch shape:{self.patch_shape} must be greater than or equal to grid shape:{self.grid_shape} in dimension {dim}"
                    )
                if pad % 2 != 0:
                    raise ValueError(
                        f"Patch shape - Grid shape difference ({pad}) must be even in dimension {dim} for proper inner tile centering (even if not used as stride)."
                    )

            _padding_list = []
            _effective_data_shape_list = []
            _fixed_stride_list = []
            
            # Determine which dimensions are spatial/depth (to apply padding to and calculate stride)
            is_spatial_dim = [False] * len(self.data_shape)
            if len(self.data_shape) == 5: # N Z H W C
                for dim_idx in range(1, 4): # Z, H, W
                    is_spatial_dim[dim_idx] = True
            elif len(self.data_shape) == 4: # N H W C
                for dim_idx in range(1, 3): # H, W
                    is_spatial_dim[dim_idx] = True

            for dim in range(len(self.data_shape)):
                if is_spatial_dim[dim]:
                    # Apply 1.5 * grid_size padding to spatial/depth dimensions
                    padding_amount = int(self.grid_shape[dim] * 1.5) # grid_shape is inner_tile_dimension
                    _padding_list.append((padding_amount, padding_amount))
                    _effective_data_shape_list.append(self.data_shape[dim] + 2 * padding_amount)
                    _fixed_stride_list.append(5) # Hardcoded stride for spatial/depth dimensions
                else: # Batch (N) and Channel (C) dimensions get no padding and no stride (full extent)
                    _padding_list.append((0, 0))
                    _effective_data_shape_list.append(self.data_shape[dim])
                    _fixed_stride_list.append(1) # Stride of 1 for N and C to represent taking the full extent
            
            self._padding = tuple(_padding_list)
            self._effective_data_shape = tuple(_effective_data_shape_list)
            self._fixed_stride = tuple(_fixed_stride_list)
        else:
            # If not WindowedTiling, ensure inherited _fixed_stride is None or appropriate for parent behavior
            # This is important so methods relying on it don't break if this class is accidentally used
            # with other tiling modes.
            self._fixed_stride = None # Or some other default if parent needs it for non-WindowedTiling

    def get_individual_dim_grid_count(self, dim: int):
        """
        Overrides to use the fixed stride for WindowedTiling.
        """
        if self.tiling_mode == TilingMode.WindowedTiling:
            current_data_len = self._effective_data_shape[dim]
            current_patch_len = self.patch_shape[dim]
            stride_for_dim = self._fixed_stride[dim]

            if dim == 0 or dim == len(self.data_shape) - 1: # Batch (N) and Channel (C) dimensions
                return self.data_shape[dim] # Each entry/channel is its own "grid" in this context

            if stride_for_dim <= 0:
                raise ValueError(f"Stride for dimension {dim} must be positive for WindowedTiling.")
            
            if current_data_len < current_patch_len: # If padded data is smaller than patch
                return 1 
            # Calculate how many full strides fit, plus one for the first patch
            return int(np.floor((current_data_len - current_patch_len) / stride_for_dim)) + 1
        else:
            # For other tiling modes, defer to the base class implementation
            return super().get_individual_dim_grid_count(dim)

    def get_gridstart_location_from_dim_index(self, dim: int, dim_index: int):
        """
        Overrides to use the fixed stride for WindowedTiling.
        """
        if self.tiling_mode == TilingMode.WindowedTiling:
            # Basic assertions from parent
            assert dim < len(self.data_shape)
            assert dim >= 0
            total_grids_in_dim = self.get_individual_dim_grid_count(dim)
            assert dim_index < total_grids_in_dim

            if dim == 0 or dim == len(self.data_shape) - 1: # Batch (N) and Channel (C) dimensions
                return dim_index 
            
            # For windowed tiling, the start position is dim_index * fixed_stride.
            return dim_index * self._fixed_stride[dim]
        else:
            # For other tiling modes, defer to the base class implementation
            return super().get_gridstart_location_from_dim_index(dim, dim_index)

    def get_patch_location_from_dataset_idx(self, dataset_idx: int):
        """
        Overrides to simplify for WindowedTiling: patch start is directly the grid start.
        """
        if self.tiling_mode == TilingMode.WindowedTiling:
            # For windowed tiling, get_location_from_dataset_idx already returns the patch start.
            return self.get_location_from_dataset_idx(dataset_idx)
        else:
            # For other tiling modes, defer to the base class implementation
            return super().get_patch_location_from_dataset_idx(dataset_idx)
    def get_dataset_idx_from_grid_location(self, location: tuple) -> int:
        """
        Overrides for reverse lookup for WindowedTiling using fixed stride.
        Converts a full N, [Z], H, W, C patch location tuple back to a flat dataset index.
        """
        if self.tiling_mode == TilingMode.WindowedTiling:
            # Ensure the length of the input 'location' tuple matches the expected data dimensions.
            # Using self._data_shape which is correctly set up in __post_init__
            if len(location) != len(self.data_shape): # Use self.data_shape as the source of truth for dim count
                raise ValueError(f"Location tuple length {len(location)} "
                                 f"must match data_shape length {len(self.data_shape)}.")
            
            grid_idx = []
            
            # Iterate through each dimension (N, [Z], H, W, C) using its original index.
            # The loop variable 'dim_original_idx' directly corresponds to the dimension's position.
            for dim_original_idx in range(len(self.data_shape)): # <<-- CRITICAL: Iterate over len(self.data_shape)
                
                # Retrieve the coordinate for the current dimension from the 'location' tuple.
                location_coord = location[dim_original_idx]
                
                # Determine the grid index based on the dimension type and fixed stride.
                # N (Batch) and C (Channel) dimensions usually have a stride of 1, 
                # meaning their grid index is just their coordinate.
                if dim_original_idx == 0: # N dimension
                    grid_idx.append(location_coord)
                elif dim_original_idx == (len(self.data_shape) - 1): # C dimension (last dimension)
                    grid_idx.append(location_coord)
                elif hasattr(self, 'mode_3D') and self.mode_3D and dim_original_idx == 1: # Z dimension if 3D
                    # Apply stride calculation for Z
                    grid_idx.append(int(np.floor(location_coord / self._fixed_stride[dim_original_idx])))
                else: # H or W dimensions (and Z if 2D context handles it as spatial implicitly)
                    # Use _fixed_stride for reverse calculation for spatial dimensions
                    grid_idx.append(int(np.floor(location_coord / self._fixed_stride[dim_original_idx])))
            
            # Convert the list of calculated grid indices to a tuple and pass to dataset_idx_from_grid_idx.
            # This method should be inherited from the base GridIndexManager.
            return self.dataset_idx_from_grid_idx(tuple(grid_idx))
        else:
            # If not WindowedTiling, defer to the base class implementation.
            # This assumes the parent class's get_dataset_idx_from_grid_location handles other modes.
            return super().get_dataset_idx_from_grid_location(location)