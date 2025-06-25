import numpy as np
import math
from typing import Tuple, Union, Callable
from .types import TilingMode, DataSplitType

# Import the necessary classes
from .multich_dataset import MultiChDloader
from .utils.windowed_tiling_manager import WindowedTilingGridIndexManager

class WindowedTilingLCMultiChDloader(MultiChDloader):
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
        
        # Call parent constructor to load data, handle noise, set up augmentations, etc.
        # The parent's __init__ will call our overridden set_img_sz.
        super().__init__(*args, **kwargs)

        # After super().__init__() returns, self._data, self._noise_data, and self._img_sz are loaded and set.
        # Now we can safely pad the data.
        self.original_data_shape = self._data.shape
        self._pad_data()

    def _pad_data(self):
        """
        Pads the loaded image data and, if it exists, the noise data.
        The padding amount is based on the patch size (`self._img_sz`) to ensure
        that any patch can be extracted from the edges of the original image.
        """
        if self._img_sz is None:
            raise ValueError("Image size (self._img_sz) must be set before padding. Check config.")

        # Use half the patch size for padding on each side of the spatial dimensions.
        pad_amount = self._img_sz // 2
        
        # Determine spatial dimensions based on whether data is 3D (5D tensor) or 2D (4D tensor)
        if self._5Ddata: # N, Z, H, W, C
            # Spatial dimensions are Z, H, W
            self.pad_width_spatial = [(pad_amount, pad_amount)] * 3
        else: # N, H, W, C
            # Spatial dimensions are H, W
            self.pad_width_spatial = [(pad_amount, pad_amount)] * 2

        # Construct the full padding tuple for np.pad. No padding on Batch (N) and Channel (C) dimensions.
        pad_width_full = [(0, 0)] + self.pad_width_spatial + [(0, 0)]

        print(f"[{self.__class__.__name__}] Original data shape: {self._data.shape}")
        print(f"[{self.__class__.__name__}] Padding spatial dimensions with: {self.pad_width_spatial}")

        # Use the same padding mode as the parent class for consistency (e.g., 'reflect', 'symmetric').
        # This is stored in _overlapping_padding_kwargs. Fallback to a sensible default.
        padding_kwargs = getattr(self, '_overlapping_padding_kwargs', {'mode': 'symmetric'})

        # Pad the main data
        self._padded_data = np.pad(self._data, pad_width=pad_width_full, **padding_kwargs)
        print(f"[{self.__class__.__name__}] Padded data shape: {self._padded_data.shape}")

        # Pad the noise data if it exists, using the same parameters
        if self._noise_data is not None:
            self._padded_noise_data = np.pad(self._noise_data, pad_width=pad_width_full, **padding_kwargs)
            print(f"[{self.__class__.__name__}] Padded noise data shape: {self._padded_noise_data.shape}")


    def set_img_sz(self, image_size, grid_size: Union[int, Tuple[int, int, int]]):
        """
        Overrides the parent method to set up the WindowedTilingGridIndexManager.
        This is called by the parent's `__init__`. It configures the patch extraction strategy.
        """
        # Set patch size and grid size from config
        self._img_sz = image_size if isinstance(image_size, int) else image_size[-1]
        self._grid_sz = grid_size
        
        # Determine a sensible stride based on the grid size. This can be configured.
        # A stride of half the grid size is a common choice for overlapping tiles.

        if isinstance(grid_size, int): # 2D case
            stride_val = grid_size // 4
            stride_spatial = (stride_val, stride_val)
        else: # 3D case
            stride_spatial = tuple(s // 4 for s in grid_size)
        
        print(f"[{self.__class__.__name__}] Image size (patch size): {self._img_sz}")
        print(f"[{self.__class__.__name__}] Grid size: {self._grid_sz}")
        print(f"[{self.__class__.__name__}] Using stride: {stride_spatial}")

        # Define patch shape and stride shape for the index manager
        numC = self._data.shape[-1]
        if self._5Ddata:
            self.patch_shape = (1, self._depth3D, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1) # (N, Z, H, W, C)
        else:
            self.patch_shape = (1, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1) # (N, H, W, C)

        # Calculate the final shape of the data *after* it will be padded.
        # This is needed to initialize the index manager correctly.
        pad_amount = self._img_sz // 2
        padded_shape_list = list(self._data.shape)
        # Iterate over spatial dimensions (all but first and last)
        for i in range(1, len(padded_shape_list) - 1):
            padded_shape_list[i] += 2 * pad_amount
        padded_shape = tuple(padded_shape_list)

        # Initialize our special windowed index manager
        self.idx_manager = WindowedTilingGridIndexManager(
            data_shape=self.original_data_shape,
            grid_shape = self._grid_sz,
            tiling_mode= TilingMode.WindowedTiling,
            padded_data_shape=padded_shape,
            patch_shape=self.patch_shape,
            stride=stride_full_shape,
        )
        
        print(f"[{self.__class__.__name__}] Initialized Windowed Tiling with {self.idx_manager.total_patch_count()} patches.")

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
