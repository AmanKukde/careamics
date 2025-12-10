import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Debug Coverage Generator
# ============================================================

def debug_coverage_generator(dset, batch_size=96, use_gpu=True):
    """
    Generator that only yields indices and dummy shapes,
    skipping all model inference.
    
    Args:
        dset: The dataset to iterate over
        batch_size: Batch size for iteration
        use_gpu: Whether to use GPU tensors
    
    Yields:
        dummy_batch, indices: Dummy tensor/array with correct shape and batch indices
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    num_samples = len(dset)
    
    # Get expected patch shape from dataset
    sample = dset[0]
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    
    # Get patch spatial dimensions from idx_manager
    patch_spatial_dims = dset.idx_manager.patch_spatial_dims  # (Z, H, W)
    
    # Infer channel count from sample
    C = 2
    
    # Create dummy shape in channel-last format: (Z, H, W, C)
    Z, H, W = patch_spatial_dims
    dummy_shape = (Z, H, W, C)
    
    print(f"[DEBUG] Dataset size: {num_samples}")
    print(f"[DEBUG] Patch spatial dims from idx_manager: {patch_spatial_dims}")
    print(f"[DEBUG] Inferred channels: {C}")
    print(f"[DEBUG] Dummy output shape per sample: {dummy_shape} (channel-last)")
    print(f"[DEBUG] Using device: {device}")
    
    global_idx = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_num in tqdm(range(num_batches), desc="Debug coverage iteration", dynamic_ncols=True):
        current_batch_size = min(batch_size, num_samples - global_idx)
        
        # Create dummy batch (all zeros, we don't care about values)
        dummy_batch = torch.zeros((current_batch_size, *dummy_shape), 
                                  dtype=torch.float32, device=device)
        indices = torch.arange(global_idx, global_idx + current_batch_size, 
                              dtype=torch.long, device=device)
        
        global_idx += current_batch_size
        yield dummy_batch, indices


# ============================================================
# Helper Functions
# ============================================================

def _parse_inner_fractions(inner_fraction, ndim):
    """Parse inner_fraction into a list of fractions."""
    if isinstance(inner_fraction, (int, float)):
        return [inner_fraction] * ndim
    else:
        return list(inner_fraction)


def _compute_inner_crop_params(patch_spatial_dims, inner_fractions, debug=False):
    """Compute crop parameters matching the stitching function."""
    start_inners = []
    end_inners = []
    inner_tile_sizes = []
    
    for full, frac in zip(patch_spatial_dims, inner_fractions):
        inner_size = int(full * frac)
        start = (full - inner_size) // 2
        end = start + inner_size
        
        start_inners.append(start)
        end_inners.append(end)
        inner_tile_sizes.append(inner_size)
    
    if debug:
        print(f"[DEBUG] Crop params - start: {start_inners}, end: {end_inners}, sizes: {inner_tile_sizes}")
    
    return start_inners, end_inners, inner_tile_sizes


def unpad(arr, pad_width):
    """Remove padding from array."""
    slices = []
    for (before, after) in pad_width:
        if before == 0 and after == 0:
            slices.append(slice(None))
        elif after == 0:
            slices.append(slice(before, None))
        else:
            slices.append(slice(before, -after))
    return arr[tuple(slices)]


# ============================================================
# Debug Coverage Stitcher (Matches Real Stitcher Logic)
# ============================================================

def stitch_coverage_only(generator, dset, inner_fraction=0.5, use_gpu=True, debug=False):
    """
    Stitches only the counts array to visualize coverage,
    matching the exact logic of stitch_predictions_3d_gpu.
    
    Args:
        generator: Generator yielding (dummy_batch, indices)
        dset: Dataset object with idx_manager
        inner_fraction: Fraction of patch to use (same as normal stitching)
        use_gpu: Whether to use GPU tensors
        debug: Whether to print debug information
    
    Returns:
        counts: Array showing how many times each voxel was covered
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    idx_manager = dset.idx_manager
    original_shape = dset._data.shape
    
    # Extract spatial dimensions: shape is (B, Z, H, W, C)
    Z = original_shape[1]
    H = original_shape[2]
    W = original_shape[3]
    C = original_shape[4]
    
    counts = torch.zeros(original_shape, device=device, dtype=torch.float32)
    
    if debug:
        print(f"[DEBUG] Original data shape: {original_shape}")
        print(f"[DEBUG] Spatial dims (Z, H, W, C): {Z}, {H}, {W}, {C}")
    
    # Parse fractions & compute crop params (matching real stitcher)
    inner_fractions = _parse_inner_fractions(inner_fraction, 3)
    start_inners, end_inners, inner_tile_sizes = _compute_inner_crop_params(
        idx_manager.patch_spatial_dims, inner_fractions, debug=debug
    )
    
    cz0, cy0, cx0 = start_inners
    zz, hh, ww = inner_tile_sizes
    
    if debug:
        print(f"[DEBUG] Patch spatial dims (Z,H,W): {idx_manager.patch_spatial_dims}")
        print(f"[DEBUG] Inner fractions: {inner_fractions}")
        print(f"[DEBUG] Start offsets (cz0, cy0, cx0): {cz0}, {cy0}, {cx0}")
        print(f"[DEBUG] Inner tile sizes (zz, hh, ww): {zz}, {hh}, {ww}")
    
    # Pre-compute ALL patch locations (matching real stitcher)
    num_patches = len(dset)
    all_locs = torch.zeros((num_patches, 4), dtype=torch.long, device=device)
    for i in range(num_patches):
        loc = idx_manager.get_patch_location_from_dataset_idx(i)
        all_locs[i] = torch.tensor(loc, device=device)
    
    if debug:
        print(f"[DEBUG] Pre-computed {num_patches} patch locations")
    
    # Track statistics
    total_patches = 0
    skipped_patches = 0
    out_of_bounds_patches = 0
    sample_locations = []  # Store first few for debugging
    
    # Iterate through generator (matching real stitcher)
    for batch_dummy, batch_indices in generator:
        B = batch_dummy.shape[0]
        
        # Get locations for this batch
        locs = all_locs[batch_indices.long()]  # [B, 4]
        
        # Crop all patches at once (matching real stitcher)
        crops = batch_dummy[:, cz0:cz0+zz, cy0:cy0+hh, cx0:cx0+ww, :]  # [B, zz, hh, ww, C]
        
        # Process batch in parallel (matching real stitcher)
        for i in range(B):
            b0, z0, y0, x0 = locs[i].tolist()
            zs, hs, ws = z0 + cz0, y0 + cy0, x0 + cx0
            ze, he, we = min(zs + zz, Z), min(hs + hh, H), min(ws + ww, W)
            
            actual_zz = ze - zs
            actual_hh = he - hs
            actual_ww = we - ws
            
            # Store first few locations for debugging
            if len(sample_locations) < 5:
                sample_locations.append({
                    'patch_idx': int(batch_indices[i]),
                    'b0': b0, 'z0': z0, 'y0': y0, 'x0': x0,
                    'zs': zs, 'hs': hs, 'ws': ws,
                    'ze': ze, 'he': he, 'we': we,
                    'actual_zz': actual_zz, 'actual_hh': actual_hh, 'actual_ww': actual_ww,
                    'crop_shape': crops[i].shape
                })
            
            # Check if patch is out of bounds
            if zs < 0 or hs < 0 or ws < 0 or ze > Z or he > H or we > W:
                out_of_bounds_patches += 1
            
            if actual_zz > 0 and actual_hh > 0 and actual_ww > 0:
                # Use add_ (in-place) matching real stitcher
                counts[b0, zs:ze, hs:he, ws:we, :].add_(1)
                total_patches += 1
            else:
                skipped_patches += 1
    
    # Convert to numpy
    counts = counts.cpu().numpy()
    
    print(f"\n[DEBUG] Coverage complete!")
    print(f"[DEBUG] Coverage shape (padded): {counts.shape}")
    
    # CRITICAL DEBUG: Analyze padding and patch distribution
    if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
        pad_z_start, pad_z_end = dset.explicit_pad_width[1]
        pad_h_start, pad_h_end = dset.explicit_pad_width[2]
        pad_w_start, pad_w_end = dset.explicit_pad_width[3]
        
        print(f"\n[DEBUG] PADDING ANALYSIS:")
        print(f"  Padding: {dset.explicit_pad_width}")
        print(f"  Padded shape: {counts.shape}")
        print(f"  Actual data region (in padded coords):")
        print(f"    Z: [{pad_z_start}:{counts.shape[1]-pad_z_end}]  (size: {counts.shape[1]-pad_z_start-pad_z_end})")
        print(f"    H: [{pad_h_start}:{counts.shape[2]-pad_h_end}]  (size: {counts.shape[2]-pad_h_start-pad_h_end})")
        print(f"    W: [{pad_w_start}:{counts.shape[3]-pad_w_end}]  (size: {counts.shape[3]-pad_w_start-pad_w_end})")
        
        # Analyze where patches are actually landing
        data_z_start, data_z_end = pad_z_start, counts.shape[1] - pad_z_end
        data_h_start, data_h_end = pad_h_start, counts.shape[2] - pad_h_end
        data_w_start, data_w_end = pad_w_start, counts.shape[3] - pad_w_end
        
        # Count coverage in different regions
        counts_in_data = counts[:, data_z_start:data_z_end, data_h_start:data_h_end, data_w_start:data_w_end, :]
        counts_in_padding = counts.copy()
        counts_in_padding[:, data_z_start:data_z_end, data_h_start:data_h_end, data_w_start:data_w_end, :] = 0
        
        print(f"\n[DEBUG] COVERAGE DISTRIBUTION:")
        print(f"  Actual data region coverage:")
        print(f"    Min: {counts_in_data.min()}, Max: {counts_in_data.max()}, Mean: {counts_in_data.mean():.2f}")
        print(f"    Uncovered: {(counts_in_data == 0).sum()} / {counts_in_data.size} ({100*(counts_in_data == 0).sum()/counts_in_data.size:.2f}%)")
        print(f"  Padding region coverage:")
        print(f"    Min: {counts_in_padding.min()}, Max: {counts_in_padding.max()}, Mean: {counts_in_padding.mean():.2f}")
        print(f"    Total voxels in padding: {counts_in_padding.size - (counts_in_padding == 0).sum()}")
    
    # Check idx_manager configuration
    print(f"\n[DEBUG] IDX_MANAGER CONFIGURATION:")
    print(f"  Patch spatial dims: {idx_manager.patch_spatial_dims}")
    if hasattr(idx_manager, 'stride'):
        print(f"  Stride: {idx_manager.stride}")
    if hasattr(idx_manager, 'overlap'):
        print(f"  Overlap: {idx_manager.overlap}")
    if hasattr(idx_manager, 'patch_shape'):
        print(f"  Patch shape: {idx_manager.patch_shape}")
    if hasattr(idx_manager, 'data_shape'):
        print(f"  Data shape idx_manager sees: {idx_manager.data_shape}")
    
    # Compute effective stride after inner cropping
    effective_coverage_per_patch = inner_tile_sizes
    print(f"\n[DEBUG] STRIDE ANALYSIS:")
    print(f"  Patch size: {idx_manager.patch_spatial_dims}")
    print(f"  Inner crop size (actual contribution): {inner_tile_sizes}")
    print(f"  Inner fraction: {inner_fractions}")
    
    # Try to infer stride from first few patches
    if len(sample_locations) >= 2:
        stride_z = sample_locations[1]['z0'] - sample_locations[0]['z0']
        stride_h = sample_locations[1]['y0'] - sample_locations[0]['y0']
        stride_w = sample_locations[1]['x0'] - sample_locations[0]['x0']
        print(f"  Inferred stride from patches: Z={stride_z}, H={stride_h}, W={stride_w}")
        
        # Check if stride matches inner tile size (needed for full coverage)
        print(f"\n[DEBUG] COVERAGE FEASIBILITY:")
        if stride_z == 0 and stride_h == 0 and stride_w != 0:
            print(f"  Stride in W direction: {stride_w}, Inner tile W: {inner_tile_sizes[2]}")
            if stride_w > inner_tile_sizes[2]:
                print(f"  ⚠️  WARNING: Stride ({stride_w}) > Inner tile ({inner_tile_sizes[2]}) - GAPS EXPECTED!")
            elif stride_w == inner_tile_sizes[2]:
                print(f"  ✓ Stride matches inner tile - should have full coverage (gaps may be due to padding)")
            else:
                print(f"  ✓ Stride < Inner tile - should have overlap")
    
    if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
        pad_z_start, pad_z_end = dset.explicit_pad_width[1]
        pad_h_start, pad_h_end = dset.explicit_pad_width[2]
        pad_w_start, pad_w_end = dset.explicit_pad_width[3]
        
        print(f"\n[DEBUG] PADDING ANALYSIS:")
        print(f"  Padding: {dset.explicit_pad_width}")
        print(f"  Padded shape: {counts.shape}")
        print(f"  Actual data region (in padded coords):")
        print(f"    Z: [{pad_z_start}:{counts.shape[1]-pad_z_end}]  (size: {counts.shape[1]-pad_z_start-pad_z_end})")
        print(f"    H: [{pad_h_start}:{counts.shape[2]-pad_h_end}]  (size: {counts.shape[2]-pad_h_start-pad_h_end})")
        print(f"    W: [{pad_w_start}:{counts.shape[3]-pad_w_end}]  (size: {counts.shape[3]-pad_w_start-pad_w_end})")
        
        # Analyze where patches are actually landing
        data_z_start, data_z_end = pad_z_start, counts.shape[1] - pad_z_end
        data_h_start, data_h_end = pad_h_start, counts.shape[2] - pad_h_end
        data_w_start, data_w_end = pad_w_start, counts.shape[3] - pad_w_end
        
        # Count coverage in different regions
        counts_in_data = counts[:, data_z_start:data_z_end, data_h_start:data_h_end, data_w_start:data_w_end, :]
        counts_in_padding = counts.copy()
        counts_in_padding[:, data_z_start:data_z_end, data_h_start:data_h_end, data_w_start:data_w_end, :] = 0
        
        print(f"\n[DEBUG] COVERAGE DISTRIBUTION:")
        print(f"  Actual data region coverage:")
        print(f"    Min: {counts_in_data.min()}, Max: {counts_in_data.max()}, Mean: {counts_in_data.mean():.2f}")
        print(f"    Uncovered: {(counts_in_data == 0).sum()} / {counts_in_data.size} ({100*(counts_in_data == 0).sum()/counts_in_data.size:.2f}%)")
        print(f"  Padding region coverage:")
        print(f"    Min: {counts_in_padding.min()}, Max: {counts_in_padding.max()}, Mean: {counts_in_padding.mean():.2f}")
        print(f"    Covered voxels in padding: {(counts_in_padding > 0).sum()} / {counts_in_padding.size} ({100*(counts_in_padding > 0).sum()/counts_in_padding.size:.2f}%)")
        
        # Count how many patches are wasted on padding
        patches_in_padding = 0
        patches_overlapping = 0
        patches_in_data = 0
        
        # Sample patches more intelligently - spread across the volume
        sample_indices = np.linspace(0, num_patches-1, min(10000, num_patches), dtype=int)
        
        for i in sample_indices:
            loc = idx_manager.get_patch_location_from_dataset_idx(i)
            b0, z0, y0, x0 = loc
            zs, hs, ws = z0 + cz0, y0 + cy0, x0 + cx0
            ze, he, we = min(zs + zz, Z), min(hs + hh, H), min(ws + ww, W)
            
            # Check if patch is entirely in padding, overlapping, or entirely in data
            if ze <= data_z_start or zs >= data_z_end or \
               he <= data_h_start or hs >= data_h_end or \
               we <= data_w_start or ws >= data_w_end:
                patches_in_padding += 1
            elif zs < data_z_start or hs < data_h_start or ws < data_w_start or \
                 ze > data_z_end or he > data_h_end or we > data_w_end:
                patches_overlapping += 1
            else:
                patches_in_data += 1
        
        total_sampled = len(sample_indices)
        print(f"\n[DEBUG] PATCH DISTRIBUTION (sampled {total_sampled} patches across full volume):")
        print(f"  Entirely in padding: {patches_in_padding} ({100*patches_in_padding/total_sampled:.1f}%)")
        print(f"  Overlapping padding/data: {patches_overlapping} ({100*patches_overlapping/total_sampled:.1f}%)")
        print(f"  Entirely in data: {patches_in_data} ({100*patches_in_data/total_sampled:.1f}%)")
        
        if total_sampled > 0:
            est_in_padding = int(patches_in_padding * num_patches / total_sampled)
            est_overlapping = int(patches_overlapping * num_patches / total_sampled)
            est_in_data = int(patches_in_data * num_patches / total_sampled)
            print(f"  Estimated total distribution:")
            print(f"    - In padding: ~{est_in_padding} / {num_patches} ({100*est_in_padding/num_patches:.1f}%)")
            print(f"    - Overlapping: ~{est_overlapping} / {num_patches} ({100*est_overlapping/num_patches:.1f}%)")
            print(f"    - In data: ~{est_in_data} / {num_patches} ({100*est_in_data/num_patches:.1f}%)")
    
    # Print sample locations for debugging (BEFORE unpadding)
    if debug and sample_locations:
        print(f"\n[DEBUG] Sample patch locations in PADDED space (first 5):")
        if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
            pad_z_start = dset.explicit_pad_width[1][0]
            pad_h_start = dset.explicit_pad_width[2][0]
            pad_w_start = dset.explicit_pad_width[3][0]
            print(f"  (Data region starts at Z={pad_z_start}, H={pad_h_start}, W={pad_w_start})")
        
        for loc in sample_locations:
            in_padding = ""
            if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
                # Check if patch overlaps with actual data
                if loc['ze'] <= pad_z_start or loc['zs'] >= counts.shape[1] - pad_z_end:
                    in_padding = " [ENTIRELY IN Z-PADDING]"
                elif loc['he'] <= pad_h_start or loc['hs'] >= counts.shape[2] - pad_h_end:
                    in_padding = " [ENTIRELY IN H-PADDING]"
                elif loc['we'] <= pad_w_start or loc['ws'] >= counts.shape[3] - pad_w_end:
                    in_padding = " [ENTIRELY IN W-PADDING]"
                elif loc['zs'] < pad_z_start or loc['hs'] < pad_h_start or loc['ws'] < pad_w_start:
                    in_padding = " [PARTIAL PADDING OVERLAP]"
            
            print(f"  Patch {loc['patch_idx']}: origin=({loc['b0']},{loc['z0']},{loc['y0']},{loc['x0']}), "
                  f"canvas=({loc['zs']}:{loc['ze']},{loc['hs']}:{loc['he']},{loc['ws']}:{loc['we']}){in_padding}")
    
    print(f"\n[DEBUG] Out of bounds patches: {out_of_bounds_patches}")
    print(f"[DEBUG] Total patches processed: {total_patches}")
    print(f"[DEBUG] Patches skipped (zero size): {skipped_patches}")
    
    # Show coverage stats BEFORE unpadding
    print(f"\n[DEBUG] Coverage stats (PADDED space):")
    print(f"  Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")
    uncovered_padded = (counts == 0).sum()
    total_voxels_padded = counts.size
    print(f"  Uncovered voxels: {uncovered_padded} / {total_voxels_padded} ({100*uncovered_padded/total_voxels_padded:.2f}%)")
    
    # Unpad if necessary
    if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
        print(f"\n[DEBUG] Removing padding: {dset.explicit_pad_width}")
        counts_ = counts.copy()
        print(f"padded counts shape : {counts_.shape}")
        counts = unpad(counts, dset.explicit_pad_width)
        print(f"[DEBUG] Coverage unpadded: {counts.shape}")
        
        # Print sample locations in UNPADDED space
        if debug and sample_locations and dset.explicit_pad_width is not None:
            pad_z_start = dset.explicit_pad_width[1][0]
            pad_h_start = dset.explicit_pad_width[2][0]
            pad_w_start = dset.explicit_pad_width[3][0]
            print(f"\n[DEBUG] Sample patch locations in UNPADDED space (first 5):")
            print(f"  (Subtracting padding offsets: Z-{pad_z_start}, H-{pad_h_start}, W-{pad_w_start})")
            for loc in sample_locations:
                unpad_zs = loc['zs'] - pad_z_start
                unpad_ze = loc['ze'] - pad_z_start
                unpad_hs = loc['hs'] - pad_h_start
                unpad_he = loc['he'] - pad_h_start
                unpad_ws = loc['ws'] - pad_w_start
                unpad_we = loc['we'] - pad_w_start
                print(f"  Patch {loc['patch_idx']}: canvas=({unpad_zs}:{unpad_ze},{unpad_hs}:{unpad_he},{unpad_ws}:{unpad_we})")
    
    print(f"\n[DEBUG] Coverage stats (UNPADDED space):")
    print(f"  Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")
    
    # Check for uncovered regions
    uncovered = (counts == 0).sum()
    total_voxels = counts.size
    print(f"[DEBUG] Uncovered voxels: {uncovered} / {total_voxels} ({100*uncovered/total_voxels:.2f}%)")
    
    # RECOMMENDATIONS
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    if hasattr(dset, 'explicit_pad_width') and dset.explicit_pad_width is not None:
        pad_info = dset.explicit_pad_width
        has_padding = any(sum(p) > 0 for p in pad_info)
        
        if has_padding:
            print("\n⚠️  ISSUE DETECTED: Many patches are being placed in padding regions!")
            print("\nPossible solutions:")
            print("1. Configure idx_manager to EXCLUDE padding regions when generating patches")
            print("   - idx_manager should only generate patches over the unpadded data region")
            print("   - Patches should have origins in range Z:[9,24], H:[48,1656], W:[48,1656]")
            print("\n2. OR adjust your workflow:")
            print("   - Don't pad before patching")
            print("   - Apply padding after inference/stitching instead")
            print("\n3. OR if you need padding for context:")
            print("   - Keep padding but adjust idx_manager to start patches at the data boundary")
            print("   - This wastes fewer patches on pure padding regions")
    
    if uncovered / total_voxels > 0.1:  # More than 10% uncovered
        print("\n⚠️  HIGH UNCOVERED PERCENTAGE DETECTED!")
        print("\nCheck your stride settings:")
        print(f"  - Current inner tile size: {inner_tile_sizes}")
        print(f"  - Stride should be ≤ inner tile size for full coverage")
        print(f"  - Try reducing stride or increasing inner_fraction")
    
    return counts, counts_

# ============================================================
# Visualization Helper
# ============================================================

def visualize_coverage(counts, slice_idx=None, channel_idx=0, batch_idx=0):
    """
    Visualize the coverage map.
    
    Args:
        counts: Coverage array from stitch_coverage_only
        slice_idx: Which z-slice to show (None = middle slice)
        channel_idx: Which channel to show
        batch_idx: Which batch element to show
    """
    # Assume shape is (B, Z, H, W, C)
    if counts.ndim == 5:
        if slice_idx is None:
            slice_idx = counts.shape[1] // 2
        
        coverage_slice = counts[batch_idx, slice_idx, :, :, channel_idx]
    else:
        raise ValueError(f"Unexpected counts shape: {counts.shape}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coverage heatmap
    im1 = axes[0].imshow(coverage_slice, cmap='hot', interpolation='nearest')
    axes[0].set_title(f'Coverage Map (B={batch_idx}, Z={slice_idx}, C={channel_idx})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Number of overlaps')
    
    # Coverage histogram
    axes[1].hist(coverage_slice.flatten(), bins=50, edgecolor='black')
    axes[1].set_xlabel('Number of overlaps')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Coverage Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCoverage Statistics for B={batch_idx}, Z={slice_idx}, C={channel_idx}:")
    print(f"  Min coverage: {coverage_slice.min()}")
    print(f"  Max coverage: {coverage_slice.max()}")
    print(f"  Mean coverage: {coverage_slice.mean():.2f}")
    print(f"  Std coverage: {coverage_slice.std():.2f}")
    print(f"  Zero coverage voxels: {(coverage_slice == 0).sum()}")


def visualize_coverage_3d(counts, channel_idx=0, batch_idx=0, num_slices=5):
    """
    Visualize multiple Z-slices at once.
    
    Args:
        counts: Coverage array
        channel_idx: Which channel to show
        batch_idx: Which batch element to show
        num_slices: How many slices to show
    """
    Z = counts.shape[1]
    slice_indices = np.linspace(0, Z-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
    if num_slices == 1:
        axes = [axes]
    
    for idx, z_idx in enumerate(slice_indices):
        coverage_slice = counts[batch_idx, z_idx, :, :, channel_idx]
        im = axes[idx].imshow(coverage_slice, cmap='hot', interpolation='nearest')
        axes[idx].set_title(f'Z={z_idx}')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        plt.colorbar(im, ax=axes[idx], label='Overlaps')
    
    plt.suptitle(f'Coverage across Z-slices (B={batch_idx}, C={channel_idx})')
    plt.tight_layout()
    plt.show()


# ============================================================
# Usage Example
# ============================================================

def debug_coverage_workflow(test_dset, batch_size=96, inner_fraction=[0.5, 0.5, 0.5], 
                           use_gpu=True, visualize=True, show_3d=True, debug=False):
    """
    Complete workflow to debug coverage without running the model.
    
    Args:
        test_dset: Your test dataset
        batch_size: Batch size for iteration
        inner_fraction: Inner crop fraction
        use_gpu: Whether to use GPU
        visualize: Whether to visualize results
        show_3d: Whether to show multiple Z-slices
        debug: Whether to print debug information
    
    Returns:
        counts: Coverage map
    """
    print("="*60)
    print("DEBUG COVERAGE WORKFLOW")
    print("="*60)
    
    # Create debug generator
    gen = debug_coverage_generator(test_dset, batch_size=batch_size, use_gpu=use_gpu)
    
    # Run coverage stitching
    counts, padded_counts = stitch_coverage_only(gen, test_dset, inner_fraction=inner_fraction, 
                                  use_gpu=use_gpu, debug=debug)
    
    # Visualize
    if visualize:
        print("\n" + "="*60)
        print("VISUALIZATION")
        print("="*60)
        visualize_coverage(counts)
        
        if show_3d:
            visualize_coverage_3d(counts, num_slices=5)
    
    return counts, padded_counts