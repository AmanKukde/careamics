
import time
import numpy as np
from typing import Iterator, Union, List, Tuple
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import tifffile as tf
import torch
from pathlib import Path

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



def _ensure_channel_last(pred, is_3d: bool):
    """Convert prediction to channel-last format"""
    is_torch = isinstance(pred, torch.Tensor)
    
    if is_3d:
        if pred.ndim == 4 and pred.shape[0] < min(pred.shape[1:]):
            if is_torch:
                return pred.permute(1, 2, 3, 0)
            else:
                return np.transpose(pred, (1, 2, 3, 0))
        return pred
    else:
        if pred.ndim == 3 and pred.shape[0] < min(pred.shape[1:]):
            if is_torch:
                return pred.permute(1, 2, 0)
            else:
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

    for batch_pred, batch_indices in tqdm(generator,dynamic_ncols=True):
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



def _compute_inner_crop_params(patch_spatial_dims, inner_fractions, debug=False):
    start, end, size = [], [], []
    for i, (full, frac) in enumerate(zip(patch_spatial_dims, inner_fractions)):
        inner_size = int(full * frac)
        s = (full - inner_size) // 2
        e = s + inner_size
        start.append(s)
        end.append(e)
        size.append(inner_size)
        if debug:
            print(f"[DEBUG] Dim {i}: full={full}, frac={frac}, inner_size={inner_size}, start={s}, end={e}")
    return start, end, size

# ============================================================
# CPU Stitcher (Vectorized) with Debug
# ============================================================
def stitch_predictions_3d_gpu(
    generator, dset, inner_fraction=0.5, debug=False, device="cuda"
):
    """Optimized GPU stitcher using scatter operations instead of indexing."""
    idx_manager = dset.idx_manager
    original_shape = dset._data.shape
    Z, H, W, C = original_shape[1:5]
    
    stitched = torch.zeros(original_shape, device=device, dtype=torch.float32)
    counts = torch.zeros_like(stitched)
    
    # Parse fractions & compute crop params
    inner_fractions = _parse_inner_fractions(inner_fraction, 3)
    start_inners, end_inners, inner_tile_sizes = _compute_inner_crop_params(
        idx_manager.patch_spatial_dims, inner_fractions, debug=debug
    )
    
    cz0, cy0, cx0 = start_inners
    zz, hh, ww = inner_tile_sizes
    
    # Pre-compute ALL patch locations
    num_patches = len(dset)
    

    print("Computing all locations now...")
    t1 = time.time()

    LOC_PATH = Path("./patch_locations.pt")

    if LOC_PATH.exists():
        all_locs = torch.load(LOC_PATH).to(device)
    else:
        all_locs = torch.zeros((num_patches, 4), dtype=torch.long, device=device)
        for i in range(num_patches):
            all_locs[i] = torch.tensor(idx_manager.get_patch_location_from_dataset_idx(i), device=device)
        torch.save(all_locs.cpu(), LOC_PATH)
    
    for batch_pred, batch_indices in generator:
        B = batch_pred.shape[0]
        batch_pred = _ensure_channel_last(batch_pred, is_3d=True).to(device).float()
        
        # Get locations for this batch

        locs = all_locs[batch_indices.long()]  # [B, 4]
        # Crop all patches at once
        crops = batch_pred[:, cz0:cz0+zz, cy0:cy0+hh, cx0:cx0+ww, :]  # [B, zz, hh, ww, C]
        
        # Process batch in parallel
        for i in range(B):
            b0, z0, y0, x0 = locs[i].tolist()
            zs, hs, ws = z0 + cz0, y0 + cy0, x0 + cx0
            ze, he, we = min(zs + zz, Z), min(hs + hh, H), min(ws + ww, W)
            
            actual_zz = ze - zs
            actual_hh = he - hs
            actual_ww = we - ws
            
            if actual_zz > 0 and actual_hh > 0 and actual_ww > 0:
                crop = crops[i, :actual_zz, :actual_hh, :actual_ww, :]
                
                # Use add_ (in-place) instead of += to avoid copy
                stitched[b0, zs:ze, hs:he, ws:we, :].add_(crop)
                counts[b0, zs:ze, hs:he, ws:we, :].add_(1)
    
    # Avoid divide-by-zero
    counts.clamp_(min=1)
    stitched.div_(counts)
    
    if debug:
        print("[DEBUG] Final GPU stitching complete")
    
    return stitched, counts

# ============================================================
# Main Dispatcher
# ============================================================
def _parse_inner_fractions(inner_fraction, ndim):
    """Parse inner_fraction into a tuple of fractions per dimension."""
    if isinstance(inner_fraction, (list, tuple)):
        return tuple(inner_fraction)
    return tuple([inner_fraction] * ndim)


def _compute_inner_crop_params(patch_dims, inner_fractions, debug=False):
    """Compute crop parameters for inner region extraction."""
    ndim = len(patch_dims)
    start_inners = []
    end_inners = []
    inner_sizes = []
    
    for i in range(ndim):
        margin = int(patch_dims[i] * (1 - inner_fractions[i]) / 2)
        start_inner = margin
        end_inner = patch_dims[i] - margin
        inner_size = end_inner - start_inner
        
        start_inners.append(start_inner)
        end_inners.append(end_inner)
        inner_sizes.append(inner_size)
        
        if debug:
            print(f"Dim {i}: patch={patch_dims[i]}, margin={margin}, "
                  f"inner=[{start_inner}:{end_inner}], size={inner_size}")
    
    return start_inners, end_inners, inner_sizes


def stitch_predictions_gpu(
    predictions: torch.Tensor,
    dset,
    inner_fraction: float = 0.5,
    device: str = "cuda",
    debug: bool = False,
) -> torch.Tensor:
    """
    GPU-based stitcher that maintains compatibility with existing stitching logic.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predictions tensor of shape [N, C, *spatial_dims] where N is number of tiles.
    dset : Dataset
        Dataset with idx_manager containing grid information.
    inner_fraction : float
        Fraction of inner tile to use (0.5 = use middle 50% of each tile).
    device : str
        Device to use for stitching.
    debug : bool
        Enable debug output.
        
    Returns
    -------
    torch.Tensor
        Stitched predictions matching the original data shape.
    """
    predictions = predictions.to(device)
    mng = dset.idx_manager
    
    # Get data shape and determine dimensionality
    data_shape = list(dset.get_data_shape())
    num_channels = max(data_shape[-1], predictions.shape[1])
    data_shape[-1] = num_channels
    
    # Determine spatial dimensions (exclude channel)
    spatial_ndim = len(data_shape) - 1
    
    if debug:
        print(f"Data shape: {data_shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Spatial dimensions: {spatial_ndim}")
    
    # Initialize output tensors
    output = torch.zeros(data_shape, device=device, dtype=predictions.dtype)
    counts = torch.zeros(data_shape, device=device, dtype=torch.float32)
    
    # Compute inner crop parameters
    patch_spatial_dims = list(predictions.shape[2:])  # Exclude batch and channel
    inner_fractions = _parse_inner_fractions(inner_fraction, len(patch_spatial_dims))
    start_inners, end_inners, inner_sizes = _compute_inner_crop_params(
        patch_spatial_dims, inner_fractions, debug=debug
    )
    
    # Process each tile
    for dset_idx in tqdm(range(predictions.shape[0]), desc="GPU Stitching", disable=not debug):
        # Get grid location for this tile
        gs = torch.tensor(mng.get_location_from_dataset_idx(dset_idx), 
                         dtype=torch.long, device=device)
        ge = gs + torch.tensor(mng.grid_shape, dtype=torch.long, device=device)
        
        # Patch start and end
        ps = gs - torch.tensor(mng.patch_offset(), dtype=torch.long, device=device)
        pe = ps + torch.tensor(mng.patch_shape, dtype=torch.long, device=device)
        
        # Valid grid start and end (clipped to data bounds)
        vgs = torch.maximum(gs, torch.zeros_like(gs))
        vge = torch.minimum(ge, torch.tensor(mng.data_shape, dtype=torch.long, device=device))
        
        # Handle boundary shift mode
        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]
        
        # Relative start and end in prediction tile
        rs = vgs - ps
        re = rs + (vge - vgs)
        
        # Apply inner crop offset
        crop_offset = torch.tensor(start_inners, dtype=torch.long, device=device)
        rs_cropped = rs + crop_offset
        re_cropped = torch.minimum(
            re,
            torch.tensor([s + inner_sizes[i] for i, s in enumerate(start_inners)],
                        dtype=torch.long, device=device)
        )
        
        # Ensure we don't go out of bounds
        re_cropped = torch.minimum(re_cropped, torch.tensor(patch_spatial_dims, dtype=torch.long, device=device))
        
        # Adjust output region accordingly
        vgs_adj = vgs + (rs_cropped - rs)
        vge_adj = vge - (re - re_cropped)
        
        # Process each channel
        for ch_idx in range(predictions.shape[1]):
            if spatial_ndim == 3:  # 4D output: [H, W, D, C]
                h_s, h_e = vgs_adj[0].item(), vge_adj[0].item()
                w_s, w_e = vgs_adj[1].item(), vge_adj[1].item()
                d_s, d_e = vgs_adj[2].item(), vge_adj[2].item()
                
                rh_s, rh_e = rs_cropped[0].item(), re_cropped[0].item()
                rw_s, rw_e = rs_cropped[1].item(), re_cropped[1].item()
                rd_s, rd_e = rs_cropped[2].item(), re_cropped[2].item()
                
                if h_e > h_s and w_e > w_s and d_e > d_s:
                    crop = predictions[dset_idx, ch_idx, rh_s:rh_e, rw_s:rw_e, rd_s:rd_e]
                    output[h_s:h_e, w_s:w_e, d_s:d_e, ch_idx].add_(crop)
                    counts[h_s:h_e, w_s:w_e, d_s:d_e, ch_idx].add_(1)
                    
            elif spatial_ndim == 4:  # 5D output: [T, H, W, D, C]
                t_s, t_e = vgs_adj[0].item(), vge_adj[0].item()
                h_s, h_e = vgs_adj[1].item(), vge_adj[1].item()
                w_s, w_e = vgs_adj[2].item(), vge_adj[2].item()
                d_s, d_e = vgs_adj[3].item(), vge_adj[3].item()
                
                rt_s, rt_e = rs_cropped[0].item(), re_cropped[0].item()
                rh_s, rh_e = rs_cropped[1].item(), re_cropped[1].item()
                rw_s, rw_e = rs_cropped[2].item(), re_cropped[2].item()
                rd_s, rd_e = rs_cropped[3].item(), re_cropped[3].item()
                
                assert t_e - t_s <= 1, "Only one frame per tile is supported"
                
                if t_e > t_s and h_e > h_s and w_e > w_s and d_e > d_s:
                    crop = predictions[dset_idx, ch_idx, rt_s:rt_e, rh_s:rh_e, rw_s:rw_e, rd_s:rd_e]
                    output[t_s, h_s:h_e, w_s:w_e, d_s:d_e, ch_idx].add_(crop)
                    counts[t_s, h_s:h_e, w_s:w_e, d_s:d_e, ch_idx].add_(1)
            else:
                raise ValueError(f"Unsupported spatial dimensions: {spatial_ndim}")
    
    # Average by count, avoiding division by zero
    counts.clamp_(min=1)
    output.div_(counts)
    
    if debug:
        print(f"[DEBUG] GPU stitching complete. Output shape: {output.shape}")
        print(f"[DEBUG] Overlap counts - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean()}")
    
    return output


def stitch_predictions_new(predictions, dset):
    """
    Original CPU-based stitcher for backward compatibility.
    Args:
        predictions: numpy array of predictions
        dset: dataset with idx_manager
    """
    mng = dset.idx_manager
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])
    output = np.zeros(shape, dtype=predictions.dtype)
    
    for dset_idx in range(predictions.shape[0]):
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape
        
        vgs = np.array([max(0, x) for x in gs], dtype=int)
        vge = np.array([min(x, y) for x, y in zip(ge, mng.data_shape)], dtype=int)
        
        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]
        
        rs = vgs - ps
        re = rs + (vge - vgs)
        
        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                output[vgs[0]:vge[0], vgs[1]:vge[1], vgs[2]:vge[2], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1]:re[1], rs[2]:re[2]]
                )
            elif len(output.shape) == 5:
                assert vge[0] - vgs[0] == 1, "Only one frame is supported"
                output[vgs[0], vgs[1]:vge[1], vgs[2]:vge[2], vgs[3]:vge[3], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1]:re[1], rs[2]:re[2], rs[3]:re[3]]
                )
            else:
                raise ValueError(f"Unsupported shape {output.shape}")
    
    return output

def stitch_predictions_windowed(generator, dset, inner_fraction=0.5, debug=False,
                                gpu=False, vectorized=False):
    dims = len(dset.idx_manager.patch_spatial_dims)
    if dims != 3:
        raise ValueError("Only 3D stitching implemented in this version")
    if gpu:
        return stitch_predictions_3d_gpu(generator, dset, inner_fraction, debug)
    elif vectorized:
        return stitch_predictions_3d_vectorized(generator, dset, inner_fraction, debug)
    else:
        raise ValueError("Non-vectorized CPU stitching not implemented")
    
