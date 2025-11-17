"""Prediction utility functions."""

import builtins
from typing import Union

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation


# TODO: why not allow input and output of torch.tensor ?
def stitch_prediction(
    tiles: list[np.ndarray],
    tile_infos: list[TileInformation],
) -> list[np.ndarray]:
    """
    Stitch tiles back together to form a full image(s).

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates. Can contain tiles
        from multiple images.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    list of numpy.ndarray
        Full image(s).
    """
    # Find where to split the lists so that only info from one image is contained.
    # Do this by locating the last tiles of each image.
    last_tiles = [tile_info.last_tile for tile_info in tile_infos]
    last_tile_position = np.where(last_tiles)[0]
    image_slices = [
        slice(
            None if i == 0 else last_tile_position[i - 1] + 1, last_tile_position[i] + 1
        )
        for i in range(len(last_tile_position))
    ]
    image_predictions = []
    # slice the lists and apply stitch_prediction_single to each in turn.
    swt = False
    # swt = False
    if swt == False:
        for image_slice in image_slices:
            image_predictions.append(
                stitch_prediction_single(tiles[image_slice], tile_infos[image_slice])
            )
    else:
        for image_slice in image_slices:
            image_predictions.append(
                stitch_prediction_single_(tiles[image_slice], tile_infos[image_slice])
            )
    return image_predictions


def stitch_prediction_single(
    tiles: list[NDArray],
    tile_infos: list[TileInformation],
) -> NDArray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    numpy.ndarray
        Full image, with dimensions SC(Z)YX.
    """
    # TODO: this is hacky... need a better way to deal with when input channels and
    #   target channels do not match
    if len(tile_infos[0].array_shape) == 4:
        # 4 dimensions => 3 spatial dimensions so -4 is channel dimension
        tile_channels = tiles[0].shape[-4]
    elif len(tile_infos[0].array_shape) == 3:
        # 3 dimensions => 2 spatial dimensions so -3 is channel dimension
        tile_channels = tiles[0].shape[-3]
    else:
        # Note pretty sure this is unreachable because array shape is already
        #   validated by TileInformation
        raise ValueError(
            f"Unsupported number of output dimension {len(tile_infos[0].array_shape)}"
        )
    # retrieve whole array size, add S dim and use number of channels in tile
    input_shape = (1, tile_channels, *tile_infos[0].array_shape[1:])
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile, tile_info in zip(tiles, tile_infos, strict=False):

        # Compute coordinates for cropping predicted tile
        crop_slices: tuple[Union[builtins.ellipsis, slice], ...] = (
            ...,
            *[slice(c[0], c[1]) for c in tile_info.overlap_crop_coords],
        )

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile[crop_slices]

        # Insert cropped tile into predicted image using stitch coordinates
        image_slices = (..., *[slice(c[0], c[1]) for c in tile_info.stitch_coords])
        predicted_image[image_slices] = cropped_tile.astype(np.float32)

    return predicted_image


from typing import List
import numpy as np

def stitch_prediction_single_(
    tiles: List[np.ndarray],
    tile_infos: List['TileInformation'],
    inner_fraction: float = 0.5
) -> np.ndarray:
    """
    Stitches tiles into a full prediction, handling both 3D (SYX) and 4D (N,Y,X) outputs.
    
    Args:
        tiles: List of tile arrays. Can be 2D, 3D, or 4D (with leading singleton axes)
        tile_infos: List of TileInformation with source_crop_coords ((y0,y1),(x0,x1))
        inner_fraction: Fraction of the tile to use as the central crop (square)
    
    Returns:
        pred_img: Stitched prediction array
    """

    # Determine output shape from first tile_info
    out_shape = tile_infos[0].array_shape
    pred_img = np.zeros(out_shape, dtype=np.float32)
    coverage = np.zeros(out_shape, dtype=np.float32)

    for i, (tile, info) in enumerate(zip(tiles, tile_infos)):
        # ---- Identify spatial axes ----
        spatial_axes = tile.shape[-len(info.source_crop_coords):]  # last axes are spatial
        num_spatial = len(spatial_axes)

        # ---- Determine square inner crop ----
        min_spatial = min(spatial_axes)
        crop_len = max(1, int(round(min_spatial * inner_fraction)))

        # ---- Build crop slices ----
        # Start with leading axes (batch, channel) if any
        crop_slices = [slice(None)] * (tile.ndim - num_spatial)
        out_slices = [slice(None)] * (pred_img.ndim - num_spatial)

        for ax, (src_start, src_end) in enumerate(info.source_crop_coords):
            axis_len = spatial_axes[ax]
            start = (axis_len - crop_len) // 2
            end = start + crop_len
            crop_slices.append(slice(start, end))
            out_slices.append(slice(src_start + start, src_start + end))

        # ---- Extract inner tile ----
        inner_tile = tile[tuple(crop_slices)]

        # ---- Squeeze extra leading singleton axes to match pred_img ----
        while inner_tile.ndim > pred_img.ndim:
            inner_tile = np.squeeze(inner_tile, axis=0)

        # ---- Debug prints ----
        print(f"\n--- Tile {i} ---")
        print(f"Tile shape: {tile.shape}")
        print(f"Spatial axes: {spatial_axes}")
        print(f"Source crop coords: {info.source_crop_coords}")
        print(f"Crop slices: {crop_slices}")
        print(f"Output slices: {out_slices}")
        print(f"Inner tile shape after squeeze: {inner_tile.shape}")
        try:
            out_region_shape = pred_img[tuple(out_slices)].shape
            print(f"Output region shape: {out_region_shape}")
        except IndexError as e:
            print(f"IndexError accessing pred_img with out_slices: {e}")
            continue

        # ---- Skip if shapes do not match ----
        if inner_tile.shape != out_region_shape:
            print(f"[SKIP] Shape mismatch: inner_tile {inner_tile.shape}, output {out_region_shape}")
            continue

        # ---- Add to prediction and track coverage ----
        pred_img[tuple(out_slices)] += inner_tile
        coverage[tuple(out_slices)] += 1

    # ---- Average overlapping regions ----
    pred_img = pred_img / np.maximum(coverage, 1)
    return pred_img
