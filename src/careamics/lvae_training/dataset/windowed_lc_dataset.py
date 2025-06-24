import numpy as np
from typing import Tuple, Union, Callable
from skimage.transform import resize # You'll need this import

# Import the necessary classes
from .config import DatasetConfig
from .lc_dataset import LCMultiChDloader
from .types import TilingMode
from careamics.lvae_training.dataset.utils.windowed_tiling_manager import WindowedTilingGridIndexManager

class WindowedTilingLCMultiChDloader(LCMultiChDloader):
    """
    A specialized LCMultiChDloader for WindowedTiling mode.

    This class pre-pads the highest resolution data and adjusts the
    GridIndexManager to work on the padded data. Lower resolutions are
    handled by downsampling the *padded* high-resolution data.
    """
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        # Initial call to parent's constructor. This will load _data, _noise_data (original size),
        # set up _tiling_mode, and crucially, populate _scaled_data and _scaled_noise_data
        # with downsampled versions of the *original unpadded* data.
        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        print("Windowed Tiling Initialised")

        if self._tiling_mode == TilingMode.WindowedTiling:
            # Store original data (unpadded) for reference if needed,
            # though after padding, all operations should use padded data.
            self._original_unpadded_data = self._data 
            self._original_unpadded_noise_data = self._noise_data

            # Calculate padding from the manager (based on original data shape and config)
            # This padding is for the highest resolution.
            # _padding will be e.g., ((0,0), (48,48), (48,48), (0,0)) for 2D HWC
            padding_config_high_res = self.idx_manager._padding 

            # Apply reflect padding to the *highest resolution* data.
            # self._data now holds the padded high-res image.
            self._data = np.pad(self._original_unpadded_data, pad_width=padding_config_high_res, mode=self._overlapping_padding_kwargs.get('mode', 'reflect'))
            if self._original_unpadded_noise_data is not None:
                self._noise_data = np.pad(self._original_unpadded_noise_data, pad_width=padding_config_high_res, mode=self._overlapping_padding_kwargs.get('mode', 'reflect'))
            
            print(f"[WindowedTilingLCMultiChDloader] Padded highest resolution data. Original unpadded shape: {self._original_unpadded_data.shape}, Padded shape: {self._data.shape}")

            # RE-INITIALIZE idx_manager with the *padded* data shape.
            # This is CRUCIAL. It ensures total_grid_count() and get_patch_location_from_dataset_idx()
            # are calculated based on the dimensions of the padded image.
            # The manager's internal stride and padding calculations will now assume `self._data.shape`
            # as the base image to tile.
        
            self.set_img_sz(data_config.image_size, data_config.grid_size)


            # CRITICAL CHANGE: RE-GENERATE _scaled_data and _scaled_noise_data
            # using the *already padded* self._data as the source for downsampling.
            # This ensures all scales are proportionally padded.

            # Initialize lists with None, then populate
            self._scaled_data = [None] * (self.multiscale_lowres_count + 1)
            self._scaled_noise_data = [None] * (self.multiscale_lowres_count + 1)

            # Assign the padded high-res data as the first scale
            self._scaled_data[0] = self._data
            if self._noise_data is not None:
                self._scaled_noise_data[0] = self._noise_data

            # Generate lower resolution versions from the *padded* highest resolution data
            for scale_idx in range(1, self.multiscale_lowres_count + 1):
                current_scale_factor = 2**scale_idx
                
                # Calculate target shape for the downsampled *padded* image
                # Assuming data is (N, D, H, W, C) or (N, H, W, C)
                target_shape = list(self._data.shape) # Start with the padded high-res shape

                if self._5Ddata:
                    # Downsample Z, H, W dimensions
                    target_shape[1] = max(1, int(np.ceil(target_shape[1] / current_scale_factor))) # Z
                    target_shape[2] = max(1, int(np.ceil(target_shape[2] / current_scale_factor))) # H
                    target_shape[3] = max(1, int(np.ceil(target_shape[3] / current_scale_factor))) # W
                else: # 2D
                    # Downsample H, W dimensions
                    target_shape[1] = max(1, int(np.ceil(target_shape[1] / current_scale_factor))) # H
                    target_shape[2] = max(1, int(np.ceil(target_shape[2] / current_scale_factor))) # W
                
                # Convert to float for resize, then back to original dtype
                self._scaled_data[scale_idx] = resize(
                    self._data.astype(np.float32), # Source is the padded high-res data
                    output_shape=tuple(target_shape),
                    anti_aliasing=True,
                    preserve_range=True # Important to keep pixel values in original range
                ).astype(self._data.dtype)

                if self._noise_data is not None:
                    self._scaled_noise_data[scale_idx] = resize(
                        self._noise_data.astype(np.float32), # Source is the padded high-res noise
                        output_shape=tuple(target_shape),
                        anti_aliasing=True,
                        preserve_range=True
                    ).astype(self._noise_data.dtype)

            print(f"[WindowedTilingLCMultiChDloader] Generated {self.multiscale_lowres_count} lower resolution scales from padded data.")
            # Optional: Verify shapes for debugging
            for i, img_s in enumerate(self._scaled_data):
                print(f"  _scaled_data[{i}].shape: {img_s.shape}")

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """ Overrides the parent's _load_img to handle pre-padded data for WindowedTiling.
        For WindowedTiling, it directly slices from the pre-padded self._data.
        This method is primarily for loading the highest resolution (scale 0) patch.
        For other tiling modes, it defers to the parent's _load_img.
        """
        if self._tiling_mode == TilingMode.WindowedTiling:
            if isinstance(index, int) or isinstance(index, np.int64):
                idx = index
            else:
                idx = index[0]  # Assume index[0] is the batch index if tuple

            # Get the patch location relative to the *padded* highest-res image
            patch_start_loc_padded_img = self.idx_manager.get_patch_location_from_dataset_idx(idx)
            n_idx = patch_start_loc_padded_img[0]  # Batch index

            # spatial_patch_start_loc_tuple will be (Z,H,W) for 5D or (H,W) for 2D, but it's a tuple from manager
            spatial_patch_start_loc_tuple = patch_start_loc_padded_img[1:]

            # The patch_shape is from the idx_manager (e.g., (1, 64, 64, 1) for 2D, or (1, D, H, W, 1) for 3D)
            patch_spatial_shape = self.idx_manager.patch_shape[1:-1]  # (Z,H,W) or (H,W)

            # Use direct slicing on the already padded self._data
            slices = [n_idx]
            for dim_idx, start_coord in enumerate(spatial_patch_start_loc_tuple):
                # Ensure dim_idx is within the range of patch_spatial_shape
                if dim_idx >= len(patch_spatial_shape):
                    raise IndexError(f"Dimension index {dim_idx} is out of range for patch_spatial_shape {patch_spatial_shape}")
                slices.append(slice(start_coord, start_coord + patch_spatial_shape[dim_idx]))
            slices.append(slice(None))  # All channels

            imgs = self._data[tuple(slices)]

            # Safety check: Ensure the extracted patch is exactly the image_size/patch_shape.
            expected_spatial_shape_for_concat = self.idx_manager.patch_shape[1:-1]  # (Z, H, W) or (H, W)
            expected_full_shape = expected_spatial_shape_for_concat + (imgs.shape[-1],)  # (Z, H, W, C) or (H, W, C)
            if imgs.shape != expected_full_shape:
                print(f"Warning: Extracted high-res patch shape {imgs.shape} does not match expected {expected_full_shape}. Resizing.")
                imgs = resize(imgs.astype(np.float32), output_shape=expected_full_shape, anti_aliasing=True, preserve_range=True).astype(imgs.dtype)

            # Reformat to tuples of (1, Z, H, W, 1) or (1, H, W, 1) for each channel
            loaded_imgs = [imgs[None, ..., i] for i in range(imgs.shape[-1])]

            noise = []
            if self._noise_data is not None and not self._disable_noise:
                noise_data_patch = self._noise_data[tuple(slices)]  # Apply same slices to noise
                if noise_data_patch.shape != expected_full_shape:
                    print(f"Warning: Extracted high-res noise patch shape {noise_data_patch.shape} does not match expected {expected_full_shape}. Resizing.")
                    noise_data_patch = resize(noise_data_patch.astype(np.float32), output_shape=expected_full_shape, anti_aliasing=True, preserve_range=True).astype(noise_data_patch.dtype)
                noise = [noise_data_patch[None, ..., i] for i in range(noise_data_patch.shape[-1])]

            return tuple(loaded_imgs), tuple(noise)
        else:
            # For other tiling modes, use the parent's _load_img logic.
            return super()._load_img(index)


    def _get_img(self, index: int):
        """
        Overrides parent's _get_img. For WindowedTiling, it loads the highest resolution patch
        from `_load_img`, and then directly extracts patches from the *already downsampled and proportionally padded*
        lower resolution images in `self._scaled_data`.
        """
        if self._tiling_mode == TilingMode.WindowedTiling:
            # _load_img already returns the fully cropped highest resolution patch (scale 0)
            img_tuples_high_res, noise_tuples_high_res = self._load_img(index)

            allres_versions = {
                i: [img_tuples_high_res[i]] for i in range(len(img_tuples_high_res))
            }
            allres_noise_versions = { # Initialize noise versions too if needed
                i: [noise_tuples_high_res[i]] for i in range(len(noise_tuples_high_res))
            }


            # Get the patch location for the highest resolution *relative to the padded image*.
            # This is what idx_manager returns.
            patch_start_loc_padded_img = self.idx_manager.get_patch_location_from_dataset_idx(index)
            
            # Loop through lower resolutions (scale_idx represents the factor for 2^scale_idx)
            for scale_idx in range(1, self.multiscale_lowres_count + 1): # +1 because 2**0 is high-res, 2**1 is 1st low-res etc.
                # Get the downsampled version of the *padded* image for this scale
                full_scaled_padded_img = self._scaled_data[scale_idx] # This is (N, D_scaled, H_scaled, W_scaled, C) or (N, H_scaled, W_scaled, C)

                # Calculate the start location *within this downsampled padded image*
                current_scale_factor = 2**scale_idx
                
                # Apply scaling to each coordinate from the highest-res padded patch start.
                # N_idx is the first one, then spatial dimensions.
                n_idx_scaled = patch_start_loc_padded_img[0] # N dimension not scaled
                
                # Spatial coordinates are scaled
                spatial_patch_start_loc_scaled_tuple = tuple(
                    coord_val // current_scale_factor for coord_val in patch_start_loc_padded_img[1:]
                )
                
                # The patch size for each output resolution is self._img_sz (or patch_shape).
                # Example: for image_size 64, it's always a 64x64 patch.
                patch_spatial_shape = self.idx_manager.patch_shape[1:-1] # (Z,H,W) or (H,W)

                # Slice the lower-resolution patch from the *already scaled and proportionally padded* image
                slices = [n_idx_scaled]
                for dim_idx, start_coord in enumerate(spatial_patch_start_loc_scaled_tuple):
                    slices.append(slice(start_coord, start_coord + patch_spatial_shape[dim_idx]))
                slices.append(slice(None)) # All channels

                current_scale_patch_img = full_scaled_padded_img[tuple(slices)]

                # Safety check: Ensure the extracted patch has the correct final dimensions.
                # This is crucial. If downsampling or slicing introduced a discrepancy, fix it here.
                expected_spatial_shape_for_concat = self.idx_manager.patch_shape[1:-1] # (Z, H, W) or (H, W)
                expected_full_shape = expected_spatial_shape_for_concat + (current_scale_patch_img.shape[-1],) # (Z, H, W, C) or (H, W, C)

                if current_scale_patch_img.shape != expected_full_shape:
                    print(f"Warning: Extracted low-res patch shape {current_scale_patch_img.shape} does not match expected {expected_full_shape} for scale {scale_idx}. Resizing.")
                    current_scale_patch_img = resize(current_scale_patch_img.astype(np.float32), 
                                                        output_shape=expected_full_shape, # Resize to (Z, H, W, C) or (H, W, C)
                                                        anti_aliasing=True, 
                                                        preserve_range=True).astype(current_scale_patch_img.dtype)


                # Reformat to (1, Z, H, W, 1) or (1, H, W, 1) per channel for concatenation later
                current_scale_img_tuples = [current_scale_patch_img[None, ..., i] for i in range(current_scale_patch_img.shape[-1])]
                
                # Append to allres_versions for each channel
                for ch_idx in range(len(img_tuples_high_res)):
                    allres_versions[ch_idx].append(current_scale_img_tuples[ch_idx])

                # Handle noise similarly if it needs to be multiscale
                if self._noise_data is not None and not self._disable_noise:
                    full_scaled_padded_noise = self._scaled_noise_data[scale_idx]
                    current_scale_patch_noise = full_scaled_padded_noise[tuple(slices)]

                    if current_scale_patch_noise.shape != expected_full_shape:
                         print(f"Warning: Extracted low-res noise patch shape {current_scale_patch_noise.shape} does not match expected {expected_full_shape} for scale {scale_idx}. Resizing.")
                         current_scale_patch_noise = resize(current_scale_patch_noise.astype(np.float32), 
                                                            output_shape=expected_full_shape, 
                                                            anti_aliasing=True, 
                                                            preserve_range=True).astype(current_scale_patch_noise.dtype)

                    current_scale_noise_tuples = [current_scale_patch_noise[None, ..., i] for i in range(current_scale_patch_noise.shape[-1])]
                    for ch_idx in range(len(noise_tuples_high_res)):
                        allres_noise_versions[ch_idx].append(current_scale_noise_tuples[ch_idx])

            # Concatenate all resolution versions for each channel
            output_img_tuples = tuple(
                [
                    np.concatenate(allres_versions[ch_idx])
                    for ch_idx in range(len(img_tuples_high_res))
                ]
            )
            output_noise_tuples = tuple(
                [
                    np.concatenate(allres_noise_versions[ch_idx])
                    for ch_idx in range(len(noise_tuples_high_res))
                ]
            )

            return output_img_tuples, output_noise_tuples
        else:
            # For other tiling modes, defer to the parent's _get_img which handles _crop_imgs etc.
            return super()._get_img(index)