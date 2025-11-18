import numpy as np
from typing import Iterator, Union, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tf

# ============================================================
# Helper utilities
# ============================================================

def _parse_inner_fractions(inner_fraction, num_dims: int):
    """Convert inner_fraction into a per-axis list."""
    if isinstance(inner_fraction, (int, float)):
        return [inner_fraction] * num_dims
    if isinstance(inner_fraction, (list, tuple)):
        if len(inner_fraction) != num_dims:
            raise ValueError(
                f"Expected {num_dims} inner fractions, got {len(inner_fraction)}."
            )
        return list(inner_fraction)
    raise TypeError("inner_fraction must be float or list.")


def _compute_inner_crop_params(patch_spatial_dims, inner_fractions):
    """Compute start/end crop indices and inner tile sizes per axis."""
    start = []
    end = []
    size = []
    for full, frac in zip(patch_spatial_dims, inner_fractions):
        inner_size = int(full * frac)
        s = (full - inner_size) // 2
        e = s + inner_size
        size.append(inner_size)
        start.append(s)
        end.append(e)
    return start, end, size


def _ensure_channel_last(pred: np.ndarray, is_3d: bool):
    """Convert prediction to channel-last format."""
    if is_3d:
        # Expect (Z,H,W,C). If (C,Z,H,W) swap.
        if pred.ndim == 4 and pred.shape[0] < min(pred.shape[1:]):
            return np.transpose(pred, (1, 2, 3, 0))
        return pred

    else:
        # 2D expects (H,W,C). If (C,H,W), swap.
        if pred.ndim == 3 and pred.shape[0] < min(pred.shape[1:]):
            return np.transpose(pred, (1, 2, 0))
        return pred


# ============================================================
# 2D stitching helper
# ============================================================

def _apply_crop_and_stitch_2d(
    pred, stitched, counts,
    loc, start_inners, inner_tile_sizes,
    original_shape
):
    """Apply inner crop and place inside stitched canvas (2D)."""
    h_start, w_start = loc[1], loc[2]

    # Compute inner coords
    hs = h_start + start_inners[0]
    ws = w_start + start_inners[1]
    he = min(hs + inner_tile_sizes[0], original_shape[1])
    we = min(ws + inner_tile_sizes[1], original_shape[2])

    # Actual usable size
    hh = max(0, he - hs)
    ww = max(0, we - ws)
    if hh <= 0 or ww <= 0:
        return

    pred_crop = pred[:hh, :ww, :]
    stitched[loc[0], hs:he, ws:we, :] += pred_crop
    counts[loc[0], hs:he, ws:we, :] += 1


# ============================================================
# 3D stitching helper
# ============================================================

def _apply_crop_and_stitch_3d(
    pred, stitched, counts,
    loc, start_inners, inner_tile_sizes,
    original_shape
):
    """Apply inner crop and place into stitched canvas (3D)."""
    z_start, h_start, w_start = loc[1], loc[2], loc[3]

    zs = z_start + start_inners[0]
    hs = h_start + start_inners[1]
    ws = w_start + start_inners[2]

    ze = min(zs + inner_tile_sizes[0], original_shape[1])
    he = min(hs + inner_tile_sizes[1], original_shape[2])
    we = min(ws + inner_tile_sizes[2], original_shape[3])

    # Actual crop sizes
    zz = max(0, ze - zs)
    hh = max(0, he - hs)
    ww = max(0, we - ws)
    if zz <= 0 or hh <= 0 or ww <= 0:
        return

    pred_crop = pred[:zz, :hh, :ww, :]
    stitched[loc[0], zs:ze, hs:he, ws:we, :] += pred_crop
    counts[loc[0], zs:ze, hs:he, ws:we, :] += 1
    

# ============================================================
# MAIN 2D FUNCTION
# ============================================================

def stitch_predictions_2d(
    generator: Iterator[np.ndarray],
    dset,
    inner_fraction: Union[float, List[float]] = 0.5,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    original_shape = dset._data.shape
    idx_manager = dset.idx_manager
    patch_spatial_dims = idx_manager.patch_spatial_dims  # (H, W)
    num_patches = len(dset)

    # init buffers
    stitched = np.zeros(original_shape, np.float32)
    counts = np.zeros_like(stitched)

    # crop params
    inner_fractions = _parse_inner_fractions(inner_fraction, 2)
    start_inners, end_inners, inner_tile_sizes = _compute_inner_crop_params(
        patch_spatial_dims, inner_fractions
    )

    patch_idx = 0
    for pred in tqdm(generator, total=num_patches):
        if patch_idx >= num_patches:
            break

        pred = _ensure_channel_last(pred, is_3d=False)
        loc = idx_manager.get_patch_location_from_dataset_idx(patch_idx)

        # crop + stitch
        _apply_crop_and_stitch_2d(
            pred, stitched, counts,
            loc, start_inners, inner_tile_sizes,
            original_shape
        )

        patch_idx += 1

    counts[counts == 0] = 1
    stitched /= counts
    return stitched, counts


# ============================================================
# MAIN 3D FUNCTION
# ============================================================

def stitch_predictions_3d(
    generator: Iterator[np.ndarray],
    dset,
    inner_fraction: Union[float, List[float]] = 0.5,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    original_shape = dset._data.shape
    idx_manager = dset.idx_manager
    patch_spatial_dims = idx_manager.patch_spatial_dims  # (Z,H,W)
    num_patches = len(dset)

    # init buffers
    stitched = np.zeros(original_shape)
    counts = np.zeros_like(stitched)

    # crop params
    inner_fractions = _parse_inner_fractions(inner_fraction, 3)
    start_inners, end_inners, inner_tile_sizes = _compute_inner_crop_params(
        patch_spatial_dims, inner_fractions
    )

    patch_idx = 0
    for pred in tqdm(generator, total=num_patches):
        if patch_idx >= num_patches:
            break

        pred = _ensure_channel_last(pred, is_3d=True)
        loc = idx_manager.get_patch_location_from_dataset_idx(patch_idx)

        _apply_crop_and_stitch_3d(
            pred, stitched, counts,
            loc, start_inners, inner_tile_sizes,
            original_shape
        )

        patch_idx += 1
        if debug and patch_idx%1000 == 0:
            intermediate =  stitched/np.maximum(1,counts)
            tf.imwrite("my_image.tiff",intermediate.transpose(0,4,1,2,3))
    counts[counts == 0] = 1
    stitched /= counts
    return stitched, counts


# ============================================================
# OPTIONAL DISPATCHER (keeps your old interface)
# ============================================================

def stitch_predictions_windowed(
    generator,
    dset,
    inner_fraction=0.5,
    debug=False
):
    dims = len(dset.idx_manager.patch_spatial_dims)
    if dims == 2:
        return stitch_predictions_2d(generator, dset, inner_fraction, debug)
    elif dims == 3:
        return stitch_predictions_3d(generator, dset, inner_fraction, debug)
    else:
        raise ValueError(f"Unsupported spatial dims={dims}")
