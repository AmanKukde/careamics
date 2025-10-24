import numpy as np
import math
from typing import Tuple, Union, Callable
from .types import TilingMode, DataSplitType

import torch
import numpy as np
from skimage.transform import resize
# Import the necessary classes
from .multich_dataset import MultiChDloader
from .utils.windowed_tiling_manager import WindowedTilingGridIndexManager

class WindowedTilingDloader(MultiChDloader):
    """
    A dataset class that inherits from MultiChDloader and implements a
    sliding window tiling strategy. It pads the entire image first, then
    extracts patches using a sliding window with a specified stride.

    This version is consistent with MultiChDloader's data processing,
    including noise handling, alpha blending, and augmentations.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the dataset. The data loading is handled by the parent
        class. After the data is loaded, it is padded.
        """
        # Initialize attributes to None
        self.original_data_shape = None
        self._padded_data = None
        self._padded_noise_data = None
        self.pad_width_spatial = None
        self.patch_shape = None

        data_config = args[0]

        self.multiscale_lowres_count = data_config.multiscale_lowres_count
        assert self.multiscale_lowres_count is not None
        # Call parent constructor to load data, handle noise, set up augmentations, etc.
        # The parent's __init__ will call our overridden set_img_sz.
        super().__init__(*args, **kwargs)

        # After super().__init__() returns, self._data, self._noise_data, and self._img_sz are loaded and set.
        # Now we can safely pad the data.

    def _pad_data(self, data, target_shape, noise_data=None):
        """
        Pads 2D (4D) or 3D (5D) data to reach the desired target_shape.
        Pads symmetrically; if padding is odd, the extra pixel goes to the 'end' side.

        Args:
            data: np.ndarray, input data (N,H,W,C) for 2D or (N,Z,H,W,C) for 3D
            target_shape: tuple, desired final shape
            noise_data: optional np.ndarray of same shape as data, will be padded the same way

        Returns:
            padded_data, padded_noise_data (if provided)
        """
        current_shape = data.shape
        ndim = data.ndim

        # Determine spatial axes
        if ndim == 5:  # 3D data: (N,Z,H,W,C)
            spatial_axes = [1,2,3]
        elif ndim == 4:  # 2D data: (N,H,W,C)
            spatial_axes = [1,2]
        else:
            raise ValueError(f"Unsupported data shape {current_shape}")

        # Compute padding for each axis
        pad_width_full = []
        for i in range(ndim):
            if i in spatial_axes:
                target = target_shape[i]
                current = current_shape[i]
                total_pad = max(0, target - current)
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width_full.append((pad_before, pad_after))
            else:
                pad_width_full.append((0,0))  # no padding for N or C

        print(f"[{self.__class__.__name__}] Padding data from {current_shape} to {target_shape}")
        print(f"[{self.__class__.__name__}] pad_width: {pad_width_full}")

        # Use padding kwargs if available
        padding_kwargs = getattr(self, '_overlapping_padding_kwargs', {'mode': 'reflect'})

        padded_data = np.pad(data, pad_width=pad_width_full, **padding_kwargs)

        if noise_data is not None:
            padded_noise_data = np.pad(noise_data, pad_width=pad_width_full, **padding_kwargs)
        else:
            padded_noise_data = None

        print(f"[{self.__class__.__name__}] Padded data shape: {padded_data.shape}")
        return padded_data, padded_noise_data

    def set_img_sz(self, image_size, grid_size: Union[int, Tuple[int, int, int]]):
        """
        Overrides the parent method to set up the WindowedTilingGridIndexManager.
        This is called by the parent's `__init__`. It configures the patch extraction strategy.
        """
        # Set patch size and grid size from config
        self._img_sz = image_size if isinstance(image_size, int) else image_size[-1]
        self._grid_sz = grid_size
        self.original_data_shape = self._data.shape
        # Determine a sensible stride based on the grid size. This can be configured.
        # A stride of half the grid size is a common choice for overlapping tiles.

        if isinstance(grid_size, int): # 2D case
            stride_val = grid_size // 8
            assert stride_val == 4
            stride_spatial = (stride_val, stride_val)
        else: # 3D case
            stride_val = grid_size[-1] // 8
            # assert stride_val == 4
            stride_spatial = [grid_size[i] // 8 for i in range(len(grid_size))]
            stride_spatial = [9,4,4]  #!AMAN hardcoded for testing
        
        print("From inside set_img_sz of WindowedTilingDloader:")
        print(f"[{self.__class__.__name__}] Data Size {self._data.shape}")
        print(f"[{self.__class__.__name__}] Image size (patch size): {self._img_sz}")
        print(f"[{self.__class__.__name__}] Grid size: {self._grid_sz}")
        print(f"[{self.__class__.__name__}] Using stride spatial: {stride_spatial}")

        # Define patch shape and stride shape for the index manager
        numC = self._data.shape[-1]
        if self._5Ddata:
            self.patch_shape = (1, self._depth3D, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1) # (N, Z, H, W, C)
        else:
            self.patch_shape = (1, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1) # (N, H, W, C)


        boundary_discard_fraction = getattr(self, 'boundary_discard_fraction', 0.5)
        
        self.padding_amount, self.padded_data_shape, pad_width, boundary_pad = self.get_padding_dimensions_and_shape(
                                                                                    data_shape = self._data.shape,
                                                                                    grid_sz = self._grid_sz,stride =  stride_spatial,
                                                                                    boundary_discard_fraction = boundary_discard_fraction,
                                                                                    is_5D = self._5Ddata)
        self.pad_width_spatial = pad_width[1:-1]  # Extract spatial padding only
        self.boundary_pad = boundary_pad  # Store for later use in unpadding

        self._padded_data, self._padded_noise_data = self._pad_data(data = self._data, target_shape = self.padded_data_shape, noise_data = self._noise_data)
        assert self._padded_data.shape == self.padded_data_shape, f"Expected padded shape {self.padded_data_shape}, got {self._padded_data.shape}"
        print(f"\n[{self.__class__.__name__}] Padded data shape: {self._padded_data.shape}, with padding of {self.padding_amount} on each edge thus {self.padding_amount*2} in total")
        print(f"[{self.__class__.__name__}] Padded noise data shape: {self._padded_noise_data.shape if self._noise_data is not None else 'None'}")

        # Initialize our special windowed index manager
        self.idx_manager = WindowedTilingGridIndexManager(
            data_shape=self._data.shape,
            grid_shape = self._grid_sz,
            tiling_mode= TilingMode.ShiftBoundary,
            padded_data_shape=self.padded_data_shape,
            patch_shape=self.patch_shape,
            stride=stride_full_shape,
        )
        
        print(f"[{self.__class__.__name__}] Finished Windowed Tiling with {self.idx_manager.total_patch_count()} patches.\n")


    def get_padding_dimensions_and_shape(
        self,
        data_shape,
        grid_sz,
        stride=None,
        boundary_discard_fraction=0.5,
        is_5D=False
    ):
        """
        Compute padding that accounts for:
        1. Boundary padding needed to avoid edge artifacts during stitching
        2. Additional padding to ensure the padded image can be properly tiled with stride
        
        Args:
            data_shape: tuple, shape of data (e.g., (N, H, W, C) or (N, Z, H, W, C))
            grid_sz: int or list, tile size (e.g., 64 or [9, 64, 64])
            stride: list or None, explicit stride per axis (if None, uses grid_sz // 8)
            boundary_discard_fraction: float, fraction of tile that gets stitched 
                                    (default 0.5 for center 50%)
            is_5D: bool, whether data is 3D (5D array) or 2D (4D array)
        
        Returns:
            total_padding: list of ints, total padding per spatial axis
            pad_width: list of tuples for np.pad format
            padded_data_shape: tuple, shape after padding
            boundary_pad: list of ints, the boundary padding component per axis
        
        Example:
            For 2D with 64x64 tiles, stitching center 32x32:
            >>> data_shape = (10, 512, 512, 1)
            >>> total_pad, pad_width, padded_shape, boundary = \\
            ...     get_padding_dimensions_and_shape_with_boundary_padding(
            ...         data_shape, 64, [8, 8], 0.5, False)
            >>> # boundary_pad = [16, 16] (half of discarded 32 pixels per side)
            >>> # pad_width = [(0,0), (16,16), (16,16), (0,0)]
            >>> # padded_shape = (10, 544, 544, 1)
            
            For 3D with 9x64x64 tiles, stitching center 9x32x32:
            >>> data_shape = (5, 32, 256, 256, 1)
            >>> total_pad, pad_width, padded_shape, boundary = \\
            ...     get_padding_dimensions_and_shape_with_boundary_padding(
            ...         data_shape, [9, 64, 64], [1, 8, 8], 0.5, True)
            >>> # boundary_pad = [2, 16, 16] (half of discarded regions per side)
            >>> # pad_width = [(0,0), (2,2), (16,16), (16,16), (0,0)]
        """
        
        # Determine spatial dimensions
        if is_5D:
            grid_sz = [grid_sz] * 3 if isinstance(grid_sz, int) else list(grid_sz)
            spatial_dims = 3
            data_spatial_shape = data_shape[1:-1]  # (Z, H, W)
        else:
            grid_sz = [grid_sz] * 2 if isinstance(grid_sz, int) else list(grid_sz)
            spatial_dims = 2
            data_spatial_shape = data_shape[1:3]  # (H, W)
        
        # Compute stride
        if stride is not None:
            step_size = list(stride)
        else:
            step_size = [max(1, g // 8) for g in grid_sz]
        
        # Compute boundary padding: half of the discarded region on each side
        # If stitching center 50%, we discard 25% on each side
        # So boundary_pad = tile_size * (1 - discard_fraction) / 2
        boundary_pad = [int(g * (1 - boundary_discard_fraction) / 2) for g in grid_sz]
        
        # Add boundary padding to data shape
        padded_for_boundary = [
            data_spatial_shape[i] + 2 * boundary_pad[i] 
            for i in range(spatial_dims)
        ]
        
        # Now compute how many tiles we need for this boundary-padded shape
        n_tiles = [
            max(1, int(np.ceil((padded_for_boundary[i] - grid_sz[i]) / step_size[i])) + 1)
            for i in range(spatial_dims)
        ]
        
        # Compute the needed length to fit all tiles
        needed_length = [
            step_size[i] * (n_tiles[i] - 1) + grid_sz[i]
            for i in range(spatial_dims)
        ]
        
        # Additional padding needed beyond boundary padding for stride alignment
        extra_padding = [
            max(0, needed_length[i] - padded_for_boundary[i])
            for i in range(spatial_dims)
        ]
        
        # Total padding (boundary + extra for stride alignment)
        total_padding = [
            boundary_pad[i] * 2 + extra_padding[i] 
            for i in range(spatial_dims)
        ]
        
        # Distribute padding symmetrically
        pad_per_side = []
        for i in range(spatial_dims):
            # Boundary padding is symmetric
            left_pad = boundary_pad[i] + extra_padding[i] // 2
            right_pad = boundary_pad[i] + (extra_padding[i] - extra_padding[i] // 2)
            pad_per_side.append((left_pad, right_pad))
        
        # Construct pad_width for np.pad
        # Format: [(N_before, N_after), (spatial_0_before, spatial_0_after), ..., (C_before, C_after)]
        pad_width = [(0, 0)]  # No padding for N dimension
        pad_width.extend(pad_per_side)  # Add spatial padding
        pad_width.append((0, 0))  # No padding for C dimension
        
        # Compute final padded shape
        padded_shape_list = list(data_shape)
        for i in range(spatial_dims):
            axis_idx = i + 1  # Skip N dimension
            padded_shape_list[axis_idx] += total_padding[i]
        padded_data_shape = tuple(padded_shape_list)
        
        return total_padding, padded_data_shape, pad_width, boundary_pad        

    def __len__(self):
        """Returns the total number of patches that can be extracted."""
        return self.idx_manager.total_patch_count() if hasattr(self, 'idx_manager') else 0

    def _get_img(self, index: int) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Overrides parent `_load_img` to extract a patch from the pre-padded data.
        This is the core of the sliding window mechanism.
        """
        # Get the top-left coordinate of the patch in the padded data space.
        patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(index)
        n_idx, *spatial_loc = patch_loc
        
        # Define the slice ranges for the patch extraction based on its location and shape.
        patch_spatial_dims = self.patch_shape[1:-1]

        slices = [slice(int(loc), int(loc + dim)) for loc, dim in zip(spatial_loc, patch_spatial_dims)]

        # Extract the patch from the padded data tensor.
        if self._5Ddata: # (N, Z, H, W, C)
            patch = self._padded_data[n_idx, slices[0], slices[1], slices[2], :]
        else: # (N, H, W, C)
            patch = self._padded_data[n_idx, slices[0], slices[1], :]

        # Split the patch into a tuple of single-channel images, as the parent logic expects.
        img_tuples = tuple(patch[None, ..., i] for i in range(patch.shape[-1]))
        
        noise_tuples = ()
        if self._padded_noise_data is not None and not self._disable_noise:
            # Extract corresponding noise patch
            if self._5Ddata:
                noise_patch = self._padded_noise_data[n_idx, slices[0], slices[1], slices[2], :]
            else:
                noise_patch = self._padded_noise_data[n_idx, slices[0], slices[1], :]
            # Split into a tuple of noise channels
            noise_tuples = tuple(noise_patch[None, ..., i] for i in range(noise_patch.shape[-1]))
            
        return img_tuples, noise_tuples

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        """
        Returns a single processed data sample (input, target).
        
        This method is now consistent with the parent `MultiChDloader`, applying
        the same augmentations, noise handling, and alpha blending logic, but on
        patches that are extracted using the sliding window method defined in `_get_img`.
        """
        # This implementation mirrors the parent `__getitem__` method,
        # but it will automatically use our overridden `_get_img` method.
        
        if self._train_index_switcher is not None:
            index = self._get_index_from_valid_target_logic(index)

        if (self._uncorrelated_channels and np.random.rand() < self._uncorrelated_channel_probab):
            # This logic calls `_get_img` multiple times to fetch channels from different locations.
            # Our overridden version will be used, making it work seamlessly.
            img_tuples, noise_tuples = self.get_uncorrelated_img_tuples(index)
        else:
            img_tuples, noise_tuples = self._get_img(index)

        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                # This also relies on `_get_img` to fetch an empty patch.
                img_tuples = self.replace_with_empty_patch(img_tuples)

        # Apply rotation and flip augmentations if enabled.
        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # --- The rest of this logic is inherited from and identical to the parent class ---

        # Add synthetic noise to create the input channels.
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = [x + noise_tuples[0] * factor for x in img_tuples]
        else:
            input_tuples = img_tuples

        # Compute the model input by weighting/blending the (potentially noisy) input channels.
        inp, alpha = self._compute_input(input_tuples)

        # Add synthetic noise to the clean channels to create the target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]
        # Compute the final target from the (now noisy) image tuples.
        target = self._compute_target(img_tuples, alpha)
        
        # Normalize the target.
        norm_target = self.normalize_target(target)

        # Prepare the final output tuple.
        output = [inp, norm_target]

        if self._return_alpha:
            output.append(np.array(alpha, dtype=np.float32))

        if self._return_index:
            output.append(index)

        return tuple(output)
