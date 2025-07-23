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

    def _pad_data(self, data, noise_data, pad_amount_one_edge: int = None):
        """
        Pads the loaded image data and, if it exists, the noise data.
        The padding amount is based on the patch size (`self._img_sz`) to ensure
        that any patch can be extracted from the edges of the original image.
        """
        print(f"[{self.__class__.__name__}] Padding Initialised. Data shape Recieved: {data.shape}, padding amount: {pad_amount_one_edge}")

        if self._img_sz is None:
            raise ValueError("Image size (self._img_sz) must be set before padding. Check config.")

        # Use half the patch size for padding on each side of the spatial dimensions.
        
        # Determine spatial dimensions based on whether data is 3D (5D tensor) or 2D (4D tensor)
        if self._5Ddata: # N, Z, H, W, C
            # Spatial dimensions are Z, H, W
            self.pad_width_spatial = [(pad_amount_one_edge, pad_amount_one_edge)] * 3
        else: # N, H, W, C
            # Spatial dimensions are H, W
            self.pad_width_spatial = [(pad_amount_one_edge, pad_amount_one_edge)] * 2

        # Construct the full padding tuple for np.pad. No padding on Batch (N) and Channel (C) dimensions.
        pad_width_full = [(0, 0)] + self.pad_width_spatial + [(0, 0)]

        print(f"[{self.__class__.__name__}] Padding spatial dimensions with: {self.pad_width_spatial}")

        # Use the same padding mode as the parent class for consistency (e.g., 'reflect', 'symmetric').
        # This is stored in _overlapping_padding_kwargs. Fallback to a sensible default.
        padding_kwargs = getattr(self, '_overlapping_padding_kwargs', {'mode': 'reflect'})

        # Pad the main data
        padded_data = np.pad(data, pad_width=pad_width_full, **padding_kwargs)

        # Pad the noise data if it exists, using the same parameters
        if noise_data is not None:
            padded_noise_data = np.pad(noise_data, pad_width=pad_width_full, **padding_kwargs)
        else:
            padded_noise_data = None

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
            stride_val = grid_size // 8
            assert stride_val == 4
            stride_spatial = (stride_val, stride_val, stride_val)
        
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

        # Calculate the final shape of the data *after* it will be padded.
        # This is needed to initialize the index manager correctly.
        #!AMAN
        #* !! TODO FIX PADDING AMOUNT OR REMOVE GRID SIZE PARAMETER 
        #* !! FURTHER MAKE THIS LOGIC NOT HARD CODED

        # Calculate the padding amount based on the data size (at this time).
        self.pad_amount_one_edge, self.padded_data_shape = self.get_padding_dimensions_and_shape(self._grid_sz)
       
        self._padded_data, self._padded_noise_data = self._pad_data(data = self._data, noise_data = self._noise_data, pad_amount_one_edge = self.pad_amount_one_edge)
        assert self._padded_data.shape[1:-1] == self.padded_data_shape[1:-1]
        print(f"\n[{self.__class__.__name__}] Padded data shape: {self._padded_data.shape}, with padding of {self.pad_amount_one_edge} on each edge thus {self.pad_amount_one_edge*2} in total")
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
        
        print(f"[{self.__class__.__name__}] Finished Windowed Tiling with {self.idx_manager.total_patch_count()} patches.")

    def get_padding_dimensions_and_shape(self, mmse_repetitions = 64):
        
        step_size = self._grid_sz//math.sqrt(mmse_repetitions)
        n_tiles = ((self._data.shape[1] - self._grid_sz)/step_size) + 1

        needed_length = step_size * (n_tiles - 1) + self._grid_sz
        total_padding = needed_length - self._data.shape[1]
        if self.multiscale_lowres_count != None:
            biggest_patch_size_possible = self._img_sz * (2**(self.multiscale_lowres_count - 1))
        biggest_patch_size_possible = self._img_sz

        total_padding += (biggest_patch_size_possible + self._grid_sz) * 0.5 * 2
        pad_amount_one_edge = int(total_padding // 2)
        #half of grid + half of patch
        padded_shape_list = list(self._data.shape)

        # Iterate over spatial dimensions (all but first and last)
        for i in range(1, len(padded_shape_list) - 1):
            padded_shape_list[i] += int(total_padding)

        padded_data_shape = tuple(padded_shape_list)
        return pad_amount_one_edge, padded_data_shape

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

import numpy as np
from typing import Tuple, Union, Callable

from skimage.transform import resize
from .multich_dataset import MultiChDloader  # base loader
from .utils.windowed_tiling_manager import WindowedTilingGridIndexManager
from .config import DatasetConfig

import numpy as np
import math
from typing import Union, Tuple, Callable
from skimage.transform import resize

from .multich_dataset import MultiChDloader
from .utils.windowed_tiling_manager import WindowedTilingGridIndexManager
from .types import TilingMode
from .config import DatasetConfig


class WindowedLCDLoader(MultiChDloader):
    """
    A dataloader that combines:
    - Multi-resolution patch extraction (as in LCMultiChDloader)
    - Sliding window tiling (as in WindowedTilingDloader)
    
    Patches are extracted at multiple downsampled resolutions, all centered on the same location
    to allow concentric input pyramids.
    """
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        # Store multiscale count and padding configs
        self.multiscale_lowres_count = data_config.multiscale_lowres_count
        assert isinstance(self.multiscale_lowres_count, int) and self.multiscale_lowres_count >= 1

        self._padding_kwargs = data_config.padding_kwargs or {"mode": "reflect"}
        self._overlapping_padding_kwargs = (
            data_config.overlapping_padding_kwargs or self._padding_kwargs
        )

        # Load base data
        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        # Compute needed padding & padded data shape
        self.pad_each_edge, padded_shape = self.get_padding_dimensions_and_shape()

        # Pad data and noise
        self._padded_data = np.pad(
            self._data,
            ((0, 0), (self.pad_each_edge, self.pad_each_edge), (self.pad_each_edge, self.pad_each_edge), (0, 0)),
            **self._padding_kwargs,
        )
        if self._noise_data is not None:
            self._padded_noise_data = np.pad(
                self._noise_data,
                ((0, 0), (self.pad_each_edge, self.pad_each_edge), (self.pad_each_edge, self.pad_each_edge), (0, 0)),
                **self._padding_kwargs,
            )
        else:
            self._padded_noise_data = None

        # Build downsampled pyramid from padded data
        self._scaled_data = [self._padded_data]
        for _ in range(1, self.multiscale_lowres_count):
            prev = self._scaled_data[-1]
            new_shape = (
                prev.shape[0],
                prev.shape[1] // 2,
                prev.shape[2] // 2,
                prev.shape[3],
            )
            resized = resize(
                prev.astype(np.float32), new_shape, preserve_range=True, anti_aliasing=True
            ).astype(prev.dtype)
            self._scaled_data.append(resized)

        # Build noise pyramid if noise is present
        if self._padded_noise_data is not None:
            self._scaled_noise_data = [self._padded_noise_data]
            for _ in range(1, self.multiscale_lowres_count):
                prev = self._scaled_noise_data[-1]
                new_shape = (
                    prev.shape[0],
                    prev.shape[1] // 2,
                    prev.shape[2] // 2,
                    prev.shape[3],
                )
                resized = resize(
                    prev.astype(np.float32), new_shape, preserve_range=True, anti_aliasing=True
                ).astype(prev.dtype)
                self._scaled_noise_data.append(resized)
        else:
            self._scaled_noise_data = None

        # Reinit index manager with stride
        stride = self._grid_sz // 8  # use configurable logic if needed
        stride_shape = (1, stride, stride, 1)
        self.patch_shape = (1, self._img_sz, self._img_sz, self._padded_data.shape[-1])

        self.idx_manager = WindowedTilingGridIndexManager(
            data_shape=self._data.shape,
            padded_data_shape=self._padded_data.shape,
            grid_shape=self._grid_sz,
            stride=stride_shape,
            patch_shape=self.patch_shape,
            tiling_mode=TilingMode.ShiftBoundary,
        )

    # ------------------------------------------------------------------
    def get_padding_dimensions_and_shape(self) -> Tuple[int, Tuple]:
        """
        Computes required padding on each edge to ensure:
        - All patches at all resolutions can be extracted
        - All patch centers (from tiling grid) remain valid even at coarsest scale
        """
        max_scale = 2 ** (self.multiscale_lowres_count - 1)
        full_reach = (self._img_sz // 2) * max_scale
        stride = self._grid_sz // 8
        pad_needed = int(np.ceil(full_reach + stride))

        padded_shape = list(self._data.shape)
        for i in range(1, len(padded_shape) - 1):  # spatial dims only
            padded_shape[i] += pad_needed * 2
        return pad_needed, tuple(padded_shape)

    def _extract_patch_at_level(self, image, center_hw, level) -> np.ndarray:
        """
        Given a center coordinate (in full-res coordinates), and a resolution level,
        returns a patch of shape (_img_sz, _img_sz) centered on the corresponding location.
        """
        factor = 2 ** level
        center = (center_hw[0] // factor, center_hw[1] // factor)
        start_h = center[0] - self._img_sz // 2
        end_h = center[0] + self._img_sz // 2
        start_w = center[1] - self._img_sz // 2
        end_w = center[1] + self._img_sz // 2
        return image[start_h:end_h, start_w:end_w]

    def _get_img(self, index: int):
        """
        Extracts a pyramid of concentric patches aligned to the same spatial center,
        determined by the grid index.
        """
        n_idx, h, w = self.idx_manager.get_patch_location_from_dataset_idx(index)

        h_padded = h + self.pad_each_edge
        w_padded = w + self.pad_each_edge
        center_hw = (h_padded + self._img_sz // 2, w_padded + self._img_sz // 2)

        img_tuples = []
        noise_tuples = [] if self._scaled_noise_data is not None else []

        for ch in range(self._padded_data.shape[-1]):
            multiscale_stack = []
            for level in range(self.multiscale_lowres_count):
                patch = self._extract_patch_at_level(
                    self._scaled_data[level][n_idx], center_hw, level
                )
                multiscale_stack.append(patch[..., ch][None])

            img_tuples.append(np.concatenate(multiscale_stack, axis=0))

            if self._scaled_noise_data is not None:
                noise_stack = []
                for level in range(self.multiscale_lowres_count):
                    patch = self._extract_patch_at_level(
                        self._scaled_noise_data[level][n_idx], center_hw, level
                    )
                    noise_stack.append(patch[..., ch][None])
                noise_tuples.append(np.concatenate(noise_stack, axis=0))

        return tuple(img_tuples), tuple(noise_tuples)

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if self._train_index_switcher is not None:
            index = self._get_index_from_valid_target_logic(index)

        img_tuples, noise_tuples = self._get_img(index)

        # -- Optional enhancements --
        if self._uncorrelated_channels and np.random.rand() < self._uncorrelated_channel_probab:
            img_tuples, noise_tuples = self.get_uncorrelated_img_tuples(index)

        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # -- Form inputs & targets --
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = [x + noise_tuples[0] * factor for x in img_tuples]
        else:
            input_tuples = img_tuples

        inp, alpha = self._compute_input(input_tuples)
        # target = self._compute_target(img_tuples[:][0], alpha)
        # self.plot(input_tuples)
        target = np.array(img_tuples)[:,0]
        norm_target = self.normalize_target(target)

        output = [inp, norm_target]
        if self._return_alpha:
            output.append(alpha)
        if self._return_index:
            output.append(index)

        return tuple(output)

    def plot(self,img_tuples):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(self._data[0,...,0])
        plt.subplot(1,2,2)
        plt.imshow(self._data[0,...,1])
        plt.show()
        plt.figure()
        plt.subplot(1,5,1)
        plt.imshow(img_tuples[0][0])
        plt.subplot(1,5,2)
        plt.imshow(img_tuples[0][1])
        plt.subplot(1,5,3)
        plt.imshow(img_tuples[0][2])
        plt.subplot(1,5,4)
        plt.imshow(img_tuples[0][3])
        plt.subplot(1,5,5)
        plt.imshow(img_tuples[0][4])
        plt.show()
        plt.figure()
        plt.subplot(1,5,1)
        plt.imshow(img_tuples[1][0])
        plt.subplot(1,5,2)
        plt.imshow(img_tuples[1][1])
        plt.subplot(1,5,3)
        plt.imshow(img_tuples[1][2])
        plt.subplot(1,5,4)
        plt.imshow(img_tuples[1][3])
        plt.subplot(1,5,5)
        plt.imshow(img_tuples[1][4])
        plt.show()
# class WindowedLCDLoader(WindowedTilingDloader):

    # def __init__(self, config, *args, **kwargs):
    #     # assert self.multiscale_lowres_count >= 1, "self.multiscale_lowres_count must be >= 1"

    #     self.base_patch_size = config.image_size[0]  # usually 64
    #     self.resize_shape = (self.base_patch_size, self.base_patch_size)

    #     # Set patch size to the max required so the grid_index_manager works correctly
    #     max_required_patch = self.base_patch_size * (2**(config.multiscale_lowres_count - 1))
    #     config.image_size = (max_required_patch, max_required_patch)

    #     super().__init__(config, *args, **kwargs)

    #     # Restore original patch size for use in resizing
    #     self.patch_size = self.base_patch_size
    #     for _ in range(1, self.multiscale_lowres_count):
    #         shape = self._scaled_data[-1].shape
    #         assert len(shape) == 4
    #         new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
    #         ds_data = resize(
    #             self._scaled_data[-1].astype(np.float32), new_shape
    #         ).astype(self._scaled_data[-1].dtype)
    #         # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
    #         assert (
    #             ds_data.max() / self._scaled_data[-1].max() < 5
    #         ), "Downsampled image should not have very different values"
    #         assert (
    #             ds_data.max() / self._scaled_data[-1].max() > 0.2
    #         ), "Downsampled image should not have very different values"

    #         self._scaled_data.append(ds_data)
    #         # do the same for noise
    #         if self._noise_data is not None:
    #             noise_data = resize(self._scaled_noise_data[-1], new_shape)
    #             self._scaled_noise_data.append(noise_data)


    # def get_multi_resolution(self, idx):
    #     x, y = self.get_location(idx)

    #     n, H, W, c = self.input_image.shape
    #     base = self.base_patch_size
    #     res_count = self.multiscale_res_count

    #     all_patches = []
    #     highest_res_np = None

    #     for i in range(res_count):
    #         scale = 2 ** (res_count - 1 - i)  # decreasing resolution
    #         curr_size = base * scale

    #         # Compute center
    #         center_top = x + base * scale // 2
    #         center_left = y + base * scale // 2

    #         half = curr_size // 2
    #         crop_top = max(center_top - half, 0)
    #         crop_left = max(center_left - half, 0)
    #         crop_bottom = crop_top + curr_size
    #         crop_right = crop_left + curr_size

    #         # Crop: (n, curr_size, curr_size, c)
    #         inp_crop = self.input_image[:, crop_top:crop_bottom, crop_left:crop_right, :]

    #         # Save highest-res for assert
    #         if i == res_count - 1:
    #             highest_res_np = inp_crop.astype(np.float32)

    #         # Resize each image in the batch
    #         resized_batch = []
    #         for img in inp_crop:
    #             resized = resize(
    #                 img.astype(np.float32),
    #                 (base, base, c),
    #                 order=3,
    #                 mode='reflect',
    #                 anti_aliasing=True
    #             ).astype(self.input_image.dtype)
    #             resized_batch.append(resized)

    #         resized_batch = np.stack(resized_batch, axis=0)  # shape: (n, base, base, c)

    #         # Assertions
    #         if highest_res_np is not None:
    #             ref_max = highest_res_np.max()
    #             curr_max = resized_batch.max()
    #             assert curr_max / ref_max < 5, "Downsampled image should not have very different values"
    #             assert curr_max / ref_max > 0.2, "Downsampled image should not have very different values"

    #         all_patches.append(resized_batch)  # List of (n, base, base, c)

    #     # Stack across resolution axis: result shape → (res_count, n, base, base, c)
    #     all_patches = np.stack(all_patches, axis=0)

    #     # Move resolution axis after batch: (n, res_count, base, base, c)
    #     all_patches = np.transpose(all_patches, (1, 0, 2, 3, 4))

    #     return all_patches  # shape: (n, res_count, base, base, c) #! IS THIS SHAPE COMPATIBLE TO MAKE INPUTS (merged) and extract target from this without overloading more functions ?

    
    # def _get_img(self, index: int) -> Tuple[np.ndarray, ...]:
    #     patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(index)
    #     n_idx, *spatial_loc = patch_loc

    #     img_array = self._get_multires_patch(n_idx, spatial_loc)  # (num_scales, H, W, C)
    #     img_tuples = tuple(img_array[..., i] for i in range(img_array.shape[-1]))  # tuple of (num_scales, H, W)

    #     noise_tuples = ()
    #     if self._padded_noise_data is not None and not self._disable_noise:
    #         noise_array = self._get_multires_patch(n_idx, spatial_loc)  # use same logic for noise
    #         noise_tuples = tuple(noise_array[..., i] for i in range(noise_array.shape[-1]))

    #     return img_tuples, noise_tuples

    # def _get_img(self, index: int) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    #     """
    #     Extracts a patch and applies multi-resolution downsampling.
    #     Returns:
    #         img_tuples: Tuple of (base_size, H, W) arrays, one per channel
    #         noise_tuples: Same structure, if available
    #     """
    #     patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(index)
    #     n_idx, *spatial_loc = patch_loc

    #     patch_spatial_dims = self.patch_shape[1:-1]
    #     slices = [slice(int(loc), int(loc + dim)) for loc, dim in zip(spatial_loc, patch_spatial_dims)]

    #     if self._5Ddata:
    #         patch = self._padded_data[n_idx, slices[0], slices[1], slices[2], :]
    #     else:
    #         patch = self._padded_data[n_idx, slices[0], slices[1], :]

    #     # Convert to torch tensor: (C, H, W)
    #     patch = torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1)

    #     patches = []
    #     highest_res_patch = None

    #     for i in range(self.multiscale_lowres_count):
    #         scale = 2 ** (self.multiscale_lowres_count - 1 - i)
    #         curr_size = self.base_patch_size * scale

    #         half = curr_size // 2
    #         center_top = patch.shape[1] // 2
    #         center_left = patch.shape[2] // 2
    #         crop_top = max(center_top - half, 0)
    #         crop_left = max(center_left - half, 0)
    #         crop_bottom = crop_top + curr_size
    #         crop_right = crop_left + curr_size

    #         inp_crop = patch[:, crop_top:crop_bottom, crop_left:crop_right]

    #         if i == self.multiscale_lowres_count - 1:
    #             highest_res_patch = inp_crop.clone()

    #         # Resize to base_patch_size
    #         np_crop = inp_crop.permute(1, 2, 0).numpy()
    #         resized_np = resize(
    #             np_crop,
    #             (self.base_patch_size, self.base_patch_size, np_crop.shape[2]),
    #             order=3, mode='reflect', anti_aliasing=True
    #         ).astype(np.float32)

    #         if highest_res_patch is not None:
    #             ref_max = highest_res_patch.max().item()
    #             curr_max = resized_np.max()
    #             assert 0.2 < curr_max / ref_max < 5, "Abnormal scale difference"

    #         patches.append(resized_np)
    #     patches = patches[::-1]
    #     stacked_patch = np.stack(patches, axis=0)  # (num_scales, H, W, C)
    #     stacked_patch = stacked_patch.transpose(0, 3, 1, 2)  # (num_scales, C, H, W)

    #     img_tuples = tuple(stacked_patch[:, i] for i in range(stacked_patch.shape[1]))  # Tuple[num_scales, H, W]

    #     # Do the same for noise
    #     noise_tuples = ()
    #     if self._padded_noise_data is not None and not self._disable_noise:
    #         if self._5Ddata:
    #             noise_patch = self._padded_noise_data[n_idx, slices[0], slices[1], slices[2], :]
    #         else:
    #             noise_patch = self._padded_noise_data[n_idx, slices[0], slices[1], :]

    #         noise_patch = torch.from_numpy(noise_patch.astype(np.float32)).permute(2, 0, 1)
    #         noise_patches = []

    #         for i in range(self.multiscale_lowres_count):
    #             scale = 2 ** (self.multiscale_lowres_count - 1 - i)
    #             curr_size = self.base_patch_size * scale

    #             half = curr_size // 2
    #             center_top = noise_patch.shape[1] // 2
    #             center_left = noise_patch.shape[2] // 2
    #             crop_top = max(center_top - half, 0)
    #             crop_left = max(center_left - half, 0)
    #             crop_bottom = crop_top + curr_size
    #             crop_right = crop_left + curr_size

    #             noise_crop = noise_patch[:, crop_top:crop_bottom, crop_left:crop_right]
    #             np_crop = noise_crop.permute(1, 2, 0).numpy()
    #             resized_np = resize(
    #                 np_crop,
    #                 (self.base_patch_size, self.base_patch_size, np_crop.shape[2]),
    #                 order=3, mode='reflect', anti_aliasing=True
    #             ).astype(np.float32)

    #             noise_patches.append(resized_np)

    #         stacked_noise = np.stack(noise_patches, axis=0).transpose(0, 3, 1, 2)
    #         noise_tuples = tuple(stacked_noise[:, i] for i in range(stacked_noise.shape[1]))

    #     return img_tuples, noise_tuples
    
    # def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        """
        Returns a single processed data sample (input, target).
        
        This method is now consistent with the parent `MultiChDloader`, applying
        the same augmentations, noise handling, and alpha blending logic, but on
        patches that are extracted using the sliding window method defined in `_get_img`.
        """
        # This implementation mirrors the parent `__getitem__` method,
        # but it will automatically use our overridden `_get_img` method.
        
        # if self._train_index_switcher is not None:
        #     index = self._get_index_from_valid_target_logic(index)

        # if (self._uncorrelated_channels and np.random.rand() < self._uncorrelated_channel_probab):
        #     # This logic calls `_get_img` multiple times to fetch channels from different locations.
        #     # Our overridden version will be used, making it work seamlessly.
        #     img_tuples, noise_tuples = self.get_uncorrelated_img_tuples(index)
        # else:
        #     img_tuples, noise_tuples = self._get_img(index)

        # if self._empty_patch_replacement_enabled:
        #     if np.random.rand() < self._empty_patch_replacement_probab:
        #         # This also relies on `_get_img` to fetch an empty patch.
        #         img_tuples = self.replace_with_empty_patch(img_tuples)

        # # Apply rotation and flip augmentations if enabled.
        # if self._enable_rotation:
        #     img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # # --- The rest of this logic is inherited from and identical to the parent class ---

        # # Add synthetic noise to create the input channels.
        # if len(noise_tuples) > 0:
        #     factor = np.sqrt(2) if self._input_is_sum else 1.0
        #     input_tuples = [x + noise_tuples[0] * factor for x in img_tuples]
        # else:
        #     input_tuples = img_tuples

        # # Compute the model input by weighting/blending the (potentially noisy) input channels.
        # inp, alpha = self._compute_input(input_tuples)

        # # Add synthetic noise to the clean channels to create the target.
        # if len(noise_tuples) >= 1:
        #     target_tuples = [
        #         x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
        #     ]
        # # Compute the final target from the (now noisy) image tuples.
        # # target = self._compute_target(img_tuples[:,0,:], alpha)
        # target = np.array(img_tuples)[:,0,:]
        # # Normalize the target.
        # norm_target = self.normalize_target(target)

        # # Prepare the final output tuple.
        # output = [inp, norm_target]

        # if self._return_alpha:
        #     output.append(np.array(alpha, dtype=np.float32))

        # if self._return_index:
        #     output.append(index)

        # return tuple(output)