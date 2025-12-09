import numpy as np
from typing import Iterator, Union, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tf

# ============================================================
# Helper Utilities
# ============================================================

def _parse_inner_fractions(inner_fraction, num_dims: int):
    if isinstance(inner_fraction, (int, float)):
        return [inner_fraction] * num_dims
    if isinstance(inner_fraction, (list, tuple)):
        if len(inner_fraction) != num_dims:
            raise ValueError(f"Expected {num_dims} inner fractions, got {len(inner_fraction)}")
        return list(inner_fraction)
    raise TypeError("inner_fraction must be float or list")


def _compute_inner_crop_params(patch_spatial_dims, inner_fractions):
    start, end, size = [], [], []
    for full, frac in zip(patch_spatial_dims, inner_fractions):
        inner_size = int(full * frac)
        s = (full - inner_size) // 2
        e = s + inner_size
        start.append(s)
        end.append(e)
        size.append(inner_size)
    return start, end, size


def compute_needed_patch_indices(
    dataset_shape,   # (Z, Y, X)
    patch_size,      # (Zp, Yp, Xp)
    stride,          # (Zs, Ys, Xs)
    pad_per_side     # (Zpad, Ypad, Xpad)
):
    Zdim, Ydim, Xdim = dataset_shape
    Zp, Yp, Xp = patch_size
    Zs, Ys, Xs = stride
    Zpad, Ypad, Xpad = pad_per_side

    # Compute start coords of all patches in each dimension
    z_starts = np.arange(0, Zdim - Zp + 1, Zs)
    y_starts = np.arange(0, Ydim - Yp + 1, Ys)
    x_starts = np.arange(0, Xdim - Xp + 1, Xs)

    # Keep only patches that overlap valid (non-padded) area
    z_valid = np.where((z_starts + Zp > Zpad) & (z_starts < Zdim - Zpad))[0]
    y_valid = np.where((y_starts + Yp > Ypad) & (y_starts < Ydim - Ypad))[0]
    x_valid = np.where((x_starts + Xp > Xpad) & (x_starts < Xdim - Xpad))[0]

    # Compute the cartesian product of valid patch indices
    zz, yy, xx = np.meshgrid(z_valid, y_valid, x_valid, indexing="ij")
    # Flatten and convert to linear indices
    patch_indices = (zz * len(y_starts) * len(x_starts) + yy * len(x_starts) + xx).ravel()

    return patch_indices.astype(int)



def _ensure_channel_last(pred: np.ndarray, is_3d: bool):
    """Convert prediction to channel-last format"""
    if is_3d:
        if pred.ndim == 4 and pred.shape[0] < min(pred.shape[1:]):
            return np.transpose(pred, (1,2,3,0))
        return pred
    else:
        if pred.ndim == 3 and pred.shape[0] < min(pred.shape[1:]):
            return np.transpose(pred, (1,2,0))
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
    for output in tqdm(generator, total=num_patches):
        pred , idx = output
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
# CPU Stitcher
# ============================================================

def stitch_predictions_3d_vectorized(generator, dset, inner_fraction=0.5, debug=False):
    """Vectorized CPU stitcher"""
    original_shape = dset._data.shape
    Z,H,W,C = original_shape[1:5]

    stitched = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros_like(stitched)

    idx_manager = dset.idx_manager
    inner_fractions = _parse_inner_fractions(inner_fraction, 3)
    start_inners, _, inner_tile_sizes = _compute_inner_crop_params(idx_manager.patch_spatial_dims, inner_fractions)

    for batch_pred, batch_indices in tqdm(generator):
        B = batch_pred.shape[0]
        for i in range(B):
            pred = _ensure_channel_last(batch_pred[i], is_3d=True)
            loc = idx_manager.get_patch_location_from_dataset_idx(int(batch_indices[i]))
            _, z0, y0, x0 = loc

            zs = z0 + start_inners[0]
            hs = y0 + start_inners[1]
            ws = x0 + start_inners[2]

            ze = min(zs + inner_tile_sizes[0], Z)
            he = min(hs + inner_tile_sizes[1], H)
            we = min(ws + inner_tile_sizes[2], W)

            zz = min(pred.shape[0], ze - zs)
            hh = min(pred.shape[1], he - hs)
            ww = min(pred.shape[2], we - ws)

            if zz <=0 or hh <=0 or ww <=0:
                continue

            stitched[loc[0], zs:zs+zz, hs:hs+hh, ws:ws+ww, :] += pred[:zz, :hh, :ww, :]
            counts[loc[0], zs:zs+zz, hs:hs+hh, ws:ws+ww, :] += 1

    counts[counts==0] = 1
    return stitched / counts, counts


# ============================================================
# GPU Stitcher
# ============================================================

def stitch_predictions_3d_gpu(generator, dset, inner_fraction=0.5, debug=False, device="cuda"):
    """GPU stitcher using torch tensors"""
    idx_manager = dset.idx_manager
    Z,H,W,C = dset._data.shape[1:5]

    stitched = torch.zeros(dset._data.shape, device=device)
    counts = torch.zeros_like(stitched)

    inner_fractions = _parse_inner_fractions(inner_fraction, 3)
    start_inners, _, inner_tile_sizes = _compute_inner_crop_params(idx_manager.patch_spatial_dims, inner_fractions)

    for batch_pred, batch_indices in tqdm(generator):
        B = batch_pred.shape[0]
        for i in range(B):
            pred = batch_pred[i].to(device)
            loc = idx_manager.get_patch_location_from_dataset_idx(int(batch_indices[i]))
            _, z0, y0, x0 = loc

            zs = z0 + start_inners[0]
            hs = y0 + start_inners[1]
            ws = x0 + start_inners[2]

            ze = min(zs + inner_tile_sizes[0], Z)
            he = min(hs + inner_tile_sizes[1], H)
            we = min(ws + inner_tile_sizes[2], W)

            zz = min(pred.shape[0], ze - zs)
            hh = min(pred.shape[1], he - hs)
            ww = min(pred.shape[2], we - ws)

            if zz <=0 or hh <=0 or ww <=0:
                continue

            stitched[:, zs:zs+zz, hs:hs+hh, ws:ws+ww, :] += pred[:zz,:hh,:ww,:]
            counts[:, zs:zs+zz, hs:hs+hh, ws:ws+ww, :] += 1

    counts[counts==0] = 1
    return stitched / counts, counts


# ============================================================
# Main Dispatcher
# ============================================================

def stitch_predictions_windowed(generator, dset, inner_fraction=0.5, debug=False,
                                use_gpu=False, vectorized=False):
    dims = len(dset.idx_manager.patch_spatial_dims)
    if dims != 3:
        raise ValueError("Only 3D stitching implemented in this version")
    if use_gpu:
        return stitch_predictions_3d_gpu(generator, dset, inner_fraction, debug)
    elif vectorized:
        return stitch_predictions_3d_vectorized(generator, dset, inner_fraction, debug)
    else:
        raise ValueError("Non-vectorized CPU stitching not implemented")