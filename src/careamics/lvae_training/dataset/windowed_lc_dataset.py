"""
A place for Datasets and Dataloaders.
"""

from typing import Tuple, Union, Callable

import numpy as np
from skimage.transform import resize
from careamics.lvae_training.dataset.types import TilingMode
from .config import DatasetConfig
from .multich_dataset import MultiChDloader
from .utils.windowed_tiling_manager import WindowedTilingGridIndexManager


class WindowedLCDLoader(MultiChDloader):
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        # Initialize attributes to None (similar to WindowedTilingDloader)
        self.original_data_shape = None
        self._padded_data = None
        self._padded_noise_data = None
        self.pad_width_spatial = None
        self.patch_shape = None
        
        self._padding_kwargs = (
            data_config.padding_kwargs  # mode=padding_mode, constant_values=constant_value
        )
        self._uncorrelated_channel_probab = data_config.uncorrelated_channel_probab

        self.multiscale_lowres_count = data_config.multiscale_lowres_count
        assert self.multiscale_lowres_count is not None

        # Call parent constructor - this will call our overridden set_img_sz
        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        if data_config.overlapping_padding_kwargs is not None:
            assert (
                self._padding_kwargs == data_config.overlapping_padding_kwargs
            ), "During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None"
        else:
            self._overlapping_padding_kwargs = data_config.padding_kwargs

        # After parent init, create scaled versions from padded data
        self._scaled_data = [self._padded_data]
        self._scaled_noise_data = [self._padded_noise_data]

        assert (
            isinstance(self.multiscale_lowres_count, int)
            and self.multiscale_lowres_count >= 1
        )
        assert isinstance(self._padding_kwargs, dict)
        assert "mode" in self._padding_kwargs

        print(f"[{self.__class__.__name__}] Creating {self.multiscale_lowres_count} scaled versions of padded data")
        
        for _ in range(1, self.multiscale_lowres_count):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            
            print(f"[{self.__class__.__name__}] Scale {len(self._scaled_data)}: Downsampling from {shape} to {new_shape}")
            
            ds_data = resize(
                self._scaled_data[-1].astype(np.float32), new_shape
            ).astype(self._scaled_data[-1].dtype)
            # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
            assert (
                ds_data.max() / self._scaled_data[-1].max() < 5
            ), "Downsampled image should not have very different values"
            assert (
                ds_data.max() / self._scaled_data[-1].max() > 0.2
            ), "Downsampled image should not have very different values"

            self._scaled_data.append(ds_data)
            # do the same for noise
            if self._padded_noise_data is not None:
                noise_data = resize(
                    self._scaled_noise_data[-1].astype(np.float32), new_shape
                ).astype(self._scaled_noise_data[-1].dtype)
                self._scaled_noise_data.append(noise_data)
        
        print(f"[{self.__class__.__name__}] Finished creating scaled versions\n")

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
            spatial_axes = [1, 2, 3]
        elif ndim == 4:  # 2D data: (N,H,W,C)
            spatial_axes = [1, 2]
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
                pad_width_full.append((0, 0))  # no padding for N or C

        print(f"[{self.__class__.__name__}] Padding data from {current_shape} to {target_shape}")
        print(f"[{self.__class__.__name__}] pad_width: {pad_width_full}")

        # Use padding kwargs
        padded_data = np.pad(data, pad_width=pad_width_full, **self._padding_kwargs)

        if noise_data is not None:
            padded_noise_data = np.pad(noise_data, pad_width=pad_width_full, **self._padding_kwargs)
        else:
            padded_noise_data = None

        print(f"[{self.__class__.__name__}] Padded data shape: {padded_data.shape}")
        return padded_data, padded_noise_data

    def get_padding_dimensions_and_shape(self, stride=None):
        """
        Compute per-edge padding and padded data shape for 2D/3D data,
        using explicit stride if provided. Pads only as necessary to fit tiles.

        Args:
            stride: list or tuple of ints [H, W] or [Z,H,W] to explicitly set step size.
                    If None, it will be computed from grid_sz // sqrt(mmse_repetitions)

        Returns:
            total_padding: list of ints, total padding per axis [H,W] or [Z,H,W]
            padded_data_shape: tuple, new shape after padding
        """
        # --- Ensure grid size iterable ---
        if self._5Ddata:
            grid_sz = [self._grid_sz] * 3 if isinstance(self._grid_sz, int) else list(self._grid_sz)
            spatial_dims = len(grid_sz)
        else:
            grid_sz = [self._grid_sz, self._grid_sz] if isinstance(self._grid_sz, int) else list(self._grid_sz)
            spatial_dims = 2

        # --- Use explicit stride or compute default ---
        if stride is not None:
            step_size = list(stride)
        else:
            step_size = [max(1, g // 8) for g in grid_sz]

        # --- Compute number of tiles per axis (ceil division) ---
        data_spatial_shape = self._data.shape[1:-1] if self._5Ddata else self._data.shape[1:3]
        n_tiles = [
            max(1, int(np.ceil((data_spatial_shape[i] - grid_sz[i]) / step_size[i])) + 1)
            for i in range(spatial_dims)
        ]

        # --- Compute needed length along each axis ---
        needed_length = [
            step_size[i] * (n_tiles[i] - 1) + grid_sz[i]
            for i in range(spatial_dims)
        ]

        # --- Total padding needed ---
        total_padding = [
            max(0, needed_length[i] - data_spatial_shape[i])
            for i in range(spatial_dims)
        ]

        # --- Compute padded shape ---
        padded_shape_list = list(self._data.shape)
        for i in range(spatial_dims):
            axis_idx = i + 1  # Skip N dimension
            padded_shape_list[axis_idx] += total_padding[i]
        padded_data_shape = tuple(padded_shape_list)

        return total_padding, padded_data_shape

    def set_img_sz(self, image_size, grid_size: Union[int, Tuple[int, int, int]]):
        """
        Overrides the parent method to set up padding and the WindowedTilingGridIndexManager.
        This is called by the parent's `__init__`. It configures the patch extraction strategy.
        """
        # Set patch size and grid size from config
        self._img_sz = image_size if isinstance(image_size, int) else image_size[-1]
        self._grid_sz = grid_size
        self.original_data_shape = self._data.shape

        # Determine stride based on the grid size
        if isinstance(grid_size, int):  # 2D case
            stride_val = grid_size // 8
            stride_spatial = (stride_val, stride_val)
        else:  # 3D case
            stride_spatial = [grid_size[i] // 8 for i in range(len(grid_size))]

        print("From inside set_img_sz of WindowedLCDLoader:")
        print(f"[{self.__class__.__name__}] Data Size {self._data.shape}")
        print(f"[{self.__class__.__name__}] Image size (patch size): {self._img_sz}")
        print(f"[{self.__class__.__name__}] Grid size: {self._grid_sz}")
        print(f"[{self.__class__.__name__}] Using stride spatial: {stride_spatial}")

        # Define patch shape and stride shape for the index manager
        numC = self._data.shape[-1]
        if self._5Ddata:
            self.patch_shape = (1, self._depth3D, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1)  # (N, Z, H, W, C)
        else:
            self.patch_shape = (1, self._img_sz, self._img_sz, numC)
            stride_full_shape = (1, *stride_spatial, 1)  # (N, H, W, C)

        # Calculate the padding amount and final padded shape
        self.pad_amount_one_edge, self.padded_data_shape = self.get_padding_dimensions_and_shape(
            stride=stride_spatial
        )

        # Pad the data
        self._padded_data, self._padded_noise_data = self._pad_data(
            data=self._data, target_shape=self.padded_data_shape, noise_data=self._noise_data
        )
        
        assert self._padded_data.shape == self.padded_data_shape, \
            f"Expected padded shape {self.padded_data_shape}, got {self._padded_data.shape}"
        
        print(f"\n[{self.__class__.__name__}] Padded data shape: {self._padded_data.shape}, "
              f"with padding of {self.pad_amount_one_edge} on each edge thus {[p*2 for p in self.pad_amount_one_edge]} in total")
        print(f"[{self.__class__.__name__}] Padded noise data shape: "
              f"{self._padded_noise_data.shape if self._noise_data is not None else 'None'}")

        # Initialize our special windowed index manager
        self.idx_manager = WindowedTilingGridIndexManager(
            data_shape=self._data.shape,
            grid_shape=self._grid_sz,
            tiling_mode=TilingMode.ShiftBoundary,
            padded_data_shape=self.padded_data_shape,
            patch_shape=self.patch_shape,
            stride=stride_full_shape,
        )

        print(f"[{self.__class__.__name__}] Finished Windowed Tiling with {self.idx_manager.total_patch_count()} patches.\n")

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        assert t_list is not None
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None

        # Reduce original data
        self._data = self._data[t_list].copy()
        
        # Reduce padded data
        self._padded_data = self._padded_data[t_list].copy()
        
        # Reduce scaled versions
        self._scaled_data = [
            self._scaled_data[i][t_list].copy() for i in range(len(self._scaled_data))
        ]

        if self._noise_data is not None:
            self._noise_data = self._noise_data[t_list].copy()
            self._padded_noise_data = self._padded_noise_data[t_list].copy()
            self._scaled_noise_data = [
                self._scaled_noise_data[i][t_list].copy()
                for i in range(len(self._scaled_noise_data))
            ]

        self.N = len(t_list)
        # TODO where tf is self._img_sz defined?
        self.set_img_sz([self._img_sz, self._img_sz], self._grid_sz)
        print(
            f"[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}"
        )

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f" Pad:{self._padding_kwargs}"
        if self._uncorrelated_channels:
            msg += f" UncorrChProbab:{self._uncorrelated_channel_probab}"
        return msg

    def _load_scaled_img(
        self, scaled_index, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index

        # tidx = self.idx_manager.get_t(idx)
        patch_loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        nidx = patch_loc_list[0]

        imgs = self._scaled_data[scaled_index][nidx]
        imgs = tuple([imgs[None, ..., i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][nidx]
            noise = tuple([noisedata[None, ..., i] for i in range(noisedata.shape[-1])])
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Overrides parent `_load_img` to extract a patch from the pre-padded data.
        This is the core of the sliding window mechanism.
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]

        # Get the top-left coordinate of the patch in the padded data space.
        patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        n_idx, *spatial_loc = patch_loc

        # Define the slice ranges for the patch extraction based on its location and shape.
        patch_spatial_dims = self.patch_shape[1:-1]

        slices = [slice(int(loc), int(loc + dim)) for loc, dim in zip(spatial_loc, patch_spatial_dims)]

        # Extract the patch from the padded data tensor.
        if self._5Ddata:  # (N, Z, H, W, C)
            patch = self._padded_data[n_idx, slices[0], slices[1], slices[2], :]
        else:  # (N, H, W, C)
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

    def _crop_img(self, img: np.ndarray, patch_start_loc: Tuple):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        max_len_vals = list(self.idx_manager.data_shape[1:-1])
        max_len_vals[-2:] = img.shape[-2:]
        return self._crop_img_with_padding(
            img, patch_start_loc, max_len_vals=max_len_vals
        )

    def __len__(self):
        """Returns the total number of patches that can be extracted."""
        return self.idx_manager.total_patch_count() if hasattr(self, 'idx_manager') else 0

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        # Noise_tuples is populated when there is synthetic noise in training
        # Should have similar type of noise with the noise model
        # Starting with microsplit, dump the noise, use it instead as an augmentation if nessesary
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            patch_start_loc = self._get_random_hw(h, w)
            if self._5Ddata:
                patch_start_loc = (
                    np.random.choice(img_tuples[0].shape[-3] - self._depth3D),
                ) + patch_start_loc
        else:
            patch_start_loc = self._get_deterministic_loc(index)

        # LC logic is located here, the function crops the image of the highest resolution
        cropped_img_tuples = [
            self._crop_flip_img(img, patch_start_loc, False, False)
            for img in img_tuples
        ]
        cropped_noise_tuples = [
            self._crop_flip_img(noise, patch_start_loc, False, False)
            for noise in noise_tuples
        ]
        patch_start_loc = list(patch_start_loc)
        h_start, w_start = patch_start_loc[-2], patch_start_loc[-1]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {
            i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))
        }
        for scale_idx in range(1, self.multiscale_lowres_count):
            # Returning the image of the lower resolution
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            patch_start_loc[-2:] = [h_start, w_start]
            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, patch_start_loc, False, False)
                for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple(
            [
                np.concatenate(allres_versions[ch_idx])
                for ch_idx in range(len(img_tuples))
            ]
        )
        return output_img_tuples, cropped_noise_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)
        if self._uncorrelated_channels:
            assert (
                self._input_idx is None
            ), "Uncorrelated channels is not implemented when there is a separate input channel."
            if np.random.rand() < self._uncorrelated_channel_probab:
                img_tuples_new = [None] * len(img_tuples)
                img_tuples_new[0] = img_tuples[0]
                for i in range(1, len(img_tuples)):
                    new_index = np.random.randint(len(self))
                    img_tuples_tmp, _ = self._get_img(new_index)
                    img_tuples_new[i] = img_tuples_tmp[i]
                img_tuples = img_tuples_new

        if self._is_train:
            if self._empty_patch_replacement_enabled:
                if np.random.rand() < self._empty_patch_replacement_probab:
                    img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # add noise to input, if noise is present combine it with the image
        # factor is for the compute input not to have too much noise because the average of two gaussians
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = []
            for x in img_tuples:
                x = (
                    x.copy()
                )  # to avoid changing the original image since it is later used for target
                # NOTE: other LC levels already have noise added. So, we just need to add noise to the highest resolution.
                x[0] = x[0] + noise_tuples[0] * factor
                input_tuples.append(x)
        else:
            input_tuples = img_tuples

        # Compute the input by sum / average the channels
        # Alpha is an amount of weight which is applied to the channels when combining them
        # How to sample alpha is still under research
        inp, alpha = self._compute_input(input_tuples)
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]

        target = self._compute_target(target_tuples, alpha)

        norm_target = self.normalize_target(target)

        output = [inp, norm_target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)