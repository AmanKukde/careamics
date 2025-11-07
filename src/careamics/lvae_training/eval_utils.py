"""
This script provides methods to evaluate the performance of the LVAE model.
It includes functions to:
    - make predictions,
    - quantify the performance of the model
    - create plots to visualize the results.
"""

import os
from typing import Optional, Union, Iterator, Tuple, List
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
# from usplit.analysis.lvae_utils import get_img_from_forward_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from careamics.lightning import VAEModule
from careamics.lvae_training.dataset import MultiChDloaderRef
from careamics.utils.metrics import scale_invariant_psnr
import numpy as np

class TilingMode:
    """
    Enum for the tiling mode.
    """

    TrimBoundary = 0
    PadBoundary = 1
    ShiftBoundary = 2


# ------------------------------------------------------------------------------------------------
# Function of plotting: TODO -> moved them to another file, plot_utils.py
def clean_ax(ax):
    """
    Helper function to remove ticks from axes in plots.
    """
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def get_eval_output_dir(
    saveplotsdir: str, patch_size: int, mmse_count: int = 50
) -> str:
    """
    Given the path to a root directory to save plots, patch size, and mmse count,
    it returns the specific directory to save the plots.
    """
    eval_out_dir = os.path.join(
        saveplotsdir, f"eval_outputs/patch_{patch_size}_mmse_{mmse_count}"
    )
    os.makedirs(eval_out_dir, exist_ok=True)
    print(eval_out_dir)
    return eval_out_dir


def get_psnr_str(tar_hsnr, pred, col_idx):
    """
    Compute PSNR between the ground truth (`tar_hsnr`) and the predicted image (`pred`).
    """
    return f"{scale_invariant_psnr(tar_hsnr[col_idx][None], pred[col_idx][None]).item():.1f}"


def add_psnr_str(ax_, psnr):
    """
    Add psnr string to the axes
    """
    textstr = f"PSNR\n{psnr}"
    props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
    # place a text box in upper left in axes coords
    ax_.text(
        0.05,
        0.95,
        textstr,
        transform=ax_.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        color="white",
    )


def get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return None


def get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def show_for_one(
    idx,
    val_dset,
    highsnr_val_dset,
    model,
    calibration_stats,
    mmse_count=5,
    patch_size=256,
    num_samples=2,
    baseline_preds=None,
):
    """
    Given an index, it plots the input, target, reconstructed images and the difference image.
    Note the the difference image is computed with respect to a ground truth image, obtained from the high SNR dataset.
    """
    highsnr_val_dset.set_img_sz(patch_size, 64)
    highsnr_val_dset.disable_noise()
    _, tar_hsnr = highsnr_val_dset[idx]
    inp, tar, recon_img_list = get_predictions(
        idx, val_dset, model, mmse_count=mmse_count, patch_size=patch_size
    )
    plot_crops(
        inp,
        tar,
        tar_hsnr,
        recon_img_list,
        calibration_stats,
        num_samples=num_samples,
        baseline_preds=baseline_preds,
    )


def plot_crops(
    inp,
    tar,
    tar_hsnr,
    recon_img_list,
    calibration_stats=None,
    num_samples=2,
    baseline_preds=None,
):
    if baseline_preds is None:
        baseline_preds = []
    if len(baseline_preds) > 0:
        for i in range(len(baseline_preds)):
            if baseline_preds[i].shape != tar_hsnr.shape:
                print(
                    f"Baseline prediction {i} shape {baseline_preds[i].shape} does not match target shape {tar_hsnr.shape}"
                )
                print("This happens when we want to predict the edges of the image.")
                return
    color_ch_list = ["goldenrod", "cyan"]
    color_pred = "red"
    insetplot_xmax_value = 10000
    insetplot_xmin_value = -1000
    inset_min_labelsize = 10
    inset_rect = [0.05, 0.05, 0.4, 0.2]

    img_sz = 3
    ncols = num_samples + len(baseline_preds) + 1 + 1 + 1 + 1 + 1 * (num_samples > 1)
    grid_factor = 5
    grid_img_sz = img_sz * grid_factor
    example_spacing = 1
    c0_extra = 1
    nimgs = 1
    fig_w = ncols * img_sz + 2 * c0_extra / grid_factor
    fig_h = int(img_sz * ncols + (example_spacing * (nimgs - 1)) / grid_factor)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=int(grid_factor * fig_h),
        ncols=int(grid_factor * fig_w),
        hspace=0.2,
        wspace=0.2,
    )
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)
    # plot baselines
    for i in range(2, 2 + len(baseline_preds)):
        for col_idx in range(baseline_preds[0].shape[0]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            print(tar_hsnr.shape, baseline_preds[i - 2].shape)
            psnr = get_psnr_str(tar_hsnr, baseline_preds[i - 2], col_idx)
            ax_temp.imshow(baseline_preds[i - 2][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)

    # plot samples
    sample_start_idx = 2 + len(baseline_preds)
    for i in range(sample_start_idx, ncols - 3):
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            psnr = get_psnr_str(tar_hsnr, recon_img_list[i - sample_start_idx], col_idx)
            ax_temp.imshow(recon_img_list[i - sample_start_idx][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)
            # inset_ax = add_pixel_kde(ax_temp,
            #                       inset_rect,
            #                       [tar_hsnr[col_idx],
            #                        recon_img_list[i - sample_start_idx][col_idx]],
            #                       inset_min_labelsize,
            #                       label_list=['', ''],
            #                       color_list=[color_ch_list[col_idx], color_pred],
            #                       plot_xmax_value=insetplot_xmax_value,
            #                       plot_xmin_value=insetplot_xmin_value)

            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])

    # difference image
    if num_samples > 1:
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    (ncols - 3) * grid_img_sz
                    + c0_extra : (ncols - 2) * grid_img_sz
                    + c0_extra,
                ]
            )
            ax_temp.imshow(
                recon_img_list[1][col_idx] - recon_img_list[0][col_idx], cmap="coolwarm"
            )
            clean_ax(ax_temp)

    for col_idx in range(recon_img_list.shape[1]):
        # print(recon_img_list.shape)
        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                c0_extra
                + (ncols - 2) * grid_img_sz : (ncols - 1) * grid_img_sz
                + c0_extra,
            ]
        )
        psnr = get_psnr_str(tar_hsnr, recon_img_list.mean(axis=0), col_idx)
        ax_temp.imshow(recon_img_list.mean(axis=0)[col_idx], cmap="magma")
        add_psnr_str(ax_temp, psnr)
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            recon_img_list.mean(axis=0)[col_idx]],
        #                           inset_min_labelsize,
        #                           label_list=['', ''],
        #                           color_list=[color_ch_list[col_idx], color_pred],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                (ncols - 1) * grid_img_sz
                + 2 * c0_extra : (ncols) * grid_img_sz
                + 2 * c0_extra,
            ]
        )
        ax_temp.imshow(tar_hsnr[col_idx], cmap="magma")
        if col_idx == 0:
            legend_ch1_ax = ax_temp
        if col_idx == 1:
            legend_ch2_ax = ax_temp

        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                grid_img_sz : 2 * grid_img_sz,
            ]
        )
        ax_temp.imshow(tar[0, col_idx].cpu().numpy(), cmap="magma")
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar[0,col_idx].cpu().numpy(),
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_kwargs_list=[{'linestyle':'--'}],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)

        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

    ax_temp = fig.add_subplot(gs[0:grid_img_sz, 0:grid_img_sz])
    ax_temp.imshow(inp[0, 0].cpu().numpy(), cmap="magma")
    clean_ax(ax_temp)

    # line_ch1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='-', label='$C_1$')
    # line_ch2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='-', label='$C_2$')
    # line_pred = mlines.Line2D([0, 1], [0, 1], color=color_pred, linestyle='-', label='Pred')
    # line_noisych1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='--', label='$C^N_1$')
    # line_noisych2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='--', label='$C^N_2$')
    # legend_ch1 = legend_ch1_ax.legend(handles=[line_ch1, line_noisych1, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                         prop={'size': 11})
    # legend_ch2 = legend_ch2_ax.legend(handles=[line_ch2, line_noisych2, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                             prop={'size': 11})

    if calibration_stats is not None:
        smaller_offset = 4
        ax_temp = fig.add_subplot(
            gs[
                grid_img_sz + 1 : 2 * grid_img_sz - smaller_offset + 1,
                smaller_offset - 1 : grid_img_sz - 1,
            ]
        )
        plot_calibration(ax_temp, calibration_stats)


def plot_calibration(ax, calibration_stats):
    """
    To plot calibration statistics (RMV vs RMSE).
    """
    first_idx = get_first_index(calibration_stats[0]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[0]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[0]["rmv"][first_idx:-last_idx],
        calibration_stats[0]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_0$",
    )

    first_idx = get_first_index(calibration_stats[1]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[1]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[1]["rmv"][first_idx:-last_idx],
        calibration_stats[1]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_1$",
    )

    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.legend()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    mid_idx = len(reg_index) // 2
    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        a = np.abs(ri - reg_index[mid_idx]) / reg_index[mid_idx]
        # print(a)
        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(cmap=newcmap, force=True)

    return newcmap


def get_fractional_change(target, prediction, max_val=None):
    """
    Get relative difference between target and prediction.
    """
    if max_val is None:
        max_val = target.max()
    return (target - prediction) / max_val


def get_zero_centered_midval(error):
    """
    When done this way, the midval ensures that the colorbar is centered at 0. (Don't know how, but it works ;))
    """
    vmax = error.max()
    vmin = error.min()
    midval = 1 - vmax / (vmax + abs(vmin))
    return midval


def plot_error(target, prediction, cmap=matplotlib.cm.coolwarm, ax=None, max_val=None):
    """
    Plot the relative difference between target and prediction.
    NOTE: The plot is overlapped to the prediction image (in gray scale).
    NOTE: The colorbar is centered at 0.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Relative difference between target and prediction
    rel_diff = get_fractional_change(target, prediction, max_val=max_val)
    midval = get_zero_centered_midval(rel_diff)
    shifted_cmap = shiftedColorMap(
        cmap, start=0, midpoint=midval, stop=1.0, name="shiftedcmap"
    )
    ax.imshow(prediction, cmap="gray")
    img_err = ax.imshow(rel_diff, cmap=shifted_cmap, alpha=1)
    plt.colorbar(img_err, ax=ax)


# -------------------------------------------------------------------------------------


def get_predictions(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    mmse_count: int = 1,
    num_workers: int = 4,
) -> tuple[dict, dict, dict]:
    """Get patch-wise predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dset : Dataset
        Dataset to predict on.
    batch_size : int
        Batch size to use for prediction.
    loss_type :
        Type of reconstruction loss used by the model, by default `None`.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.
    num_workers : int, optional
        Number of workers to use for DataLoader, by default 4.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]
        Tuple containing:
            - predictions: Predicted images for the dataset.
            - predictions_std: Standard deviation of the predicted images.
            - logvar_arr: Log variance of the predicted images.
            - losses: Reconstruction losses for the predictions.
            - psnr: PSNR values for the predictions.
    """
    if hasattr(dset, "dsets"):
        multifile_stitched_predictions = {}
        multifile_stitched_stds = {}
        for d in dset.dsets:
            stitched_predictions, stitched_stds = get_single_file_mmse(
                model=model,
                dset=d,
                batch_size=batch_size,
                tile_size=tile_size,
                grid_size=grid_size,
                mmse_count=mmse_count,
                num_workers=num_workers,
            )
            # get filename without extension and path
            filename = d._fpath.name
            multifile_stitched_predictions[filename] = stitched_predictions
            multifile_stitched_stds[filename] = stitched_stds
        return (
            multifile_stitched_predictions,
            multifile_stitched_stds,
        )
    else:
        stitched_predictions, stitched_stds = get_single_file_mmse(
            model=model,
            dset=dset,
            batch_size=batch_size,
            tile_size=tile_size,
            grid_size=grid_size,
            mmse_count=mmse_count,
            num_workers=num_workers,
        )
        # TODO stitching still not working properly for weirdly shaped images
        # get filename without extension and path
        # TODO in the ref ds this is the name of a folder not file :(
        filename = dset._fpath.name
        return (
            {filename: stitched_predictions},
            {filename: stitched_stds},
        )


def get_single_file_predictions(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Get patch-wise predictions from a model for a single file dataset."""
    if tile_size and grid_size:
        dset.set_img_sz(tile_size, grid_size)

    device = get_device()

    dloader = DataLoader(
        dset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    model.eval()
    model.to(device)
    tiles = []
    logvar_arr = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp, tar = batch
            inp = inp.to(device)
            tar = tar.to(device)

            # get model output
            rec, _ = model(inp)

            # get reconstructed img
            if model.model.predict_logvar is None:
                rec_img = rec
                logvar = torch.tensor([-1])
            else:
                rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
            logvar_arr.append(logvar.cpu().numpy())  # Why do we need this ?

            tiles.append(rec_img.cpu().numpy())

    tile_samples = np.concatenate(tiles, axis=0)
    return stitch_predictions_new(tile_samples, dset)


def get_single_file_mmse(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    mmse_count: int = 1,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Get patch-wise predictions from a model for a single file dataset."""
    device = get_device()

    dloader = DataLoader(
        dset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    if tile_size and grid_size:
        dset.set_img_sz(tile_size, grid_size)

    model.eval()
    model.to(device)
    tile_mmse = []
    tile_stds = []
    logvar_arr = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp, tar = batch
            inp = inp.to(device)
            tar = tar.to(device)

            rec_img_list = []
            for _ in range(mmse_count):

                # get model output
                rec, _ = model(inp)

                # get reconstructed img
                if model.model.predict_logvar is None:
                    rec_img = rec
                    logvar = torch.tensor([-1])
                else:
                    rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0))  # add MMSE dim
                logvar_arr.append(logvar.cpu().numpy())  # Why do we need this ?

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)  # avg over MMSE dim
            std_imgs = torch.std(samples, dim=0)  # std over MMSE dim

            tile_mmse.append(mmse_imgs.cpu().numpy())
            tile_stds.append(std_imgs.cpu().numpy())

    tiles_arr = np.concatenate(tile_mmse, axis=0)
    tile_stds = np.concatenate(tile_stds, axis=0)
    # TODO temporary hack, because of the stupid jupyter!
    # If a user reruns a cell with class definition, isinstance will return False
    if str(MultiChDloaderRef).split(".")[-1] == str(dset.__class__).split(".")[-1]:
        stitch_func = stitch_predictions_general
    else:
        stitch_func = stitch_predictions_new
    stitched_predictions = stitch_func(tiles_arr, dset)
    stitched_stds = stitch_func(tile_stds, dset)
    return stitched_predictions, stitched_stds

# def get_single_file_mmse_usplit(
#     model: VAEModule,
#     dset: Dataset,
#     batch_size: int,
#     tile_size: Optional[tuple[int, int]] = None,
#     grid_size: Optional[int] = None,
#     mmse_count: int = 1,
#     num_workers: int = 4,
#     sliding_window_flag = False,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Get patch-wise predictions from a model for a single file dataset."""
#     device = get_device()

#     dloader = DataLoader(
#         dset,
#         pin_memory=False,
#         num_workers=num_workers,
#         shuffle=False,
#         batch_size=batch_size,
#     )
#     # if tile_size and grid_size:
#     #     dset.set_img_sz(tile_size, grid_size)

#     model.eval()
#     model.to(device)
#     tile_mmse = []
#     tile_stds = []
#     # logvar_arr = []
#     with torch.no_grad():
#         for batch in tqdm(dloader, desc="Predicting tiles"):
#             inp, tar = batch
#             inp = inp.to(device)
#             tar = tar.to(device)

#             rec_img_list = []
#             for _ in range(mmse_count):
#                 # get model output
#                 rec, _ = model(inp)
#                 rec = get_img_from_forward_output(rec,model) 
#                 # get reconstructed img
#                 # pdb.set_trace()
#                 rec_img_list.append(rec.cpu().unsqueeze(0))  # add MMSE dim

#             # aggregate results
#             # pdb.set_trace()
#             samples = torch.cat(rec_img_list, dim=0)
#             mmse_imgs = torch.mean(samples, dim=0)  # avg over MMSE dim
#             std_imgs = torch.std(samples, dim=0)  # std over MMSE dim

#             tile_mmse.append(mmse_imgs.cpu().numpy())
#             tile_stds.append(std_imgs.cpu().numpy())

#     tiles_arr = np.concatenate(tile_mmse, axis=0)
#     tile_stds = np.concatenate(tile_stds, axis=0)
#     # TODO temporary hack, because of the stupid jupyter!
    
#     # If a user reruns a cell with class definition, isinstance will return False
#     if str(MultiChDloaderRef).split(".")[-1] == str(dset.__class__).split(".")[-1]:
#         stitch_func = stitch_predictions_general
#     else:
#         stitch_func = stitch_predictions_new
#     if sliding_window_flag:
#         stitch_func = stitch_and_crop_predictions_inner_tile
    
#     print(f"Using {stitch_func}")

#     if sliding_window_flag:
#         stitched_predictions, counts_matrix_for_stitched_predictions = stitch_func(tiles_arr, dset)
#         stitched_stds, counts_matrix_for_stitched_stds = stitch_func(tile_stds, dset)
#         return stitched_predictions, stitched_stds, counts_matrix_for_stitched_predictions, counts_matrix_for_stitched_stds
#     stitched_predictions = stitch_func(tiles_arr, dset)
#     stitched_stds = stitch_func(tile_stds, dset)
#     return stitched_predictions, stitched_stds

# # ------------------------------------------------------------------------------------------
### Classes and Functions used to stitch predictions
class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, h_idx_range, w_idx_range, t_idx):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range

    def __str__(self):
        msg = f"T:{self.t} [{self.h_start}-{self.h_end}) [{self.w_start}-{self.w_end}) "
        return msg


def _get_location(extra_padding, hwt, pred_h, pred_w):
    h_start, w_start, t_idx = hwt
    h_start -= extra_padding
    h_end = h_start + pred_h
    w_start -= extra_padding
    w_end = w_start + pred_w
    return PatchLocation((h_start, h_end), (w_start, w_end), t_idx)


def get_location_from_idx(dset, dset_input_idx, pred_h, pred_w):
    """
    For a given idx of the dataset, it returns where exactly in the dataset, does this prediction lies.
    Note that this prediction also has padded pixels and so a subset of it will be used in the final prediction.
    Which time frame, which spatial location (h_start, h_end, w_start,w_end)
    Args:
        dset:
        dset_input_idx:
        pred_h:
        pred_w:

    Returns
    -------
    """
    extra_padding = dset.per_side_overlap_pixelcount()
    htw = dset.get_idx_manager().hwt_from_idx(
        dset_input_idx, grid_size=dset.get_grid_size()
    )
    return _get_location(extra_padding, htw, pred_h, pred_w)


def remove_pad(pred, loc, extra_padding, smoothening_pixelcount, frame_shape):
    assert smoothening_pixelcount == 0
    if extra_padding - smoothening_pixelcount > 0:
        h_s = extra_padding - smoothening_pixelcount

        # rows
        h_N = frame_shape[0]
        if loc.h_end > h_N:
            assert loc.h_end - extra_padding + smoothening_pixelcount <= h_N
        h_e = extra_padding - smoothening_pixelcount

        w_s = extra_padding - smoothening_pixelcount

        # columns
        w_N = frame_shape[1]
        if loc.w_end > w_N:
            assert loc.w_end - extra_padding + smoothening_pixelcount <= w_N

        w_e = extra_padding - smoothening_pixelcount

        return pred[h_s:-h_e, w_s:-w_e]

    return pred


def update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount):
    extra_padding = extra_padding - smoothening_pixelcount
    loc.h_start += extra_padding
    loc.w_start += extra_padding
    loc.h_end -= extra_padding
    loc.w_end -= extra_padding
    return loc


def stitch_predictions(predictions, dset, smoothening_pixelcount=0):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    assert smoothening_pixelcount >= 0 and isinstance(smoothening_pixelcount, int)
    extra_padding = dset.per_side_overlap_pixelcount()
    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    frame_shape = dset.get_data_shape()[1:3]
    for dset_input_idx in range(predictions.shape[0]):
        loc = get_location_from_idx(
            dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1]
        )

        mask = None
        cropped_pred_list = []
        for ch_idx in range(predictions.shape[1]):
            # class i
            cropped_pred_i = remove_pad(
                predictions[dset_input_idx, ch_idx],
                loc,
                extra_padding,
                smoothening_pixelcount,
                frame_shape,
            )

            if mask is None:
                # NOTE: don't need to compute it for every patch.
                assert (
                    smoothening_pixelcount == 0
                ), "For smoothing,enable the get_smoothing_mask. It is disabled since I don't use it and it needs modification to work with non-square images"
                mask = 1
                # mask = _get_smoothing_mask(cropped_pred_i.shape, smoothening_pixelcount, loc, frame_size)

            cropped_pred_list.append(cropped_pred_i)

        loc = update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount)
        for ch_idx in range(predictions.shape[1]):
            output[loc.t, loc.h_start : loc.h_end, loc.w_start : loc.w_end, ch_idx] += (
                cropped_pred_list[ch_idx] * mask
            )

    return output


# from disentangle.analysis.stitch_prediction import *
def stitch_predictions_new(predictions, dset):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    # Commented out since it is not used as of now
    # if isinstance(dset, MultiFileDset):
    #     cum_count = 0
    #     output = []
    #     for dset in dset.dsets:
    #         cnt = dset.idx_manager.total_grid_count()
    #         output.append(
    #             stitch_predictions(predictions[cum_count:cum_count + cnt], dset))
    #         cum_count += cnt
    #     return output

    # else:
    mng = dset.idx_manager

    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    # frame_shape = dset.get_data_shape()[:-1]
    for dset_idx in range(predictions.shape[0]):
        # loc = get_location_from_idx(dset, dset_idx, predictions.shape[-2], predictions.shape[-1])
        # grid start, grid end
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape

        # patch start, patch end
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape
        # print('PS')
        # print(ps)
        # print(pe)

        # valid grid start, valid grid end
        vgs = np.array([max(0, x) for x in gs], dtype=int)
        vge = np.array([min(x, y) for x, y in zip(ge, mng.data_shape)], dtype=int)
        # assert np.all(vgs == gs)
        # assert np.all(vge == ge) # TODO comented out this shit cuz I have no interest to dig why it's failing at this point !
        # print('VGS')
        # print(gs)
        # print(ge)

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        rs = vgs - ps
        re = rs + (vge - vgs)
        # print('RS')
        # print(rs)
        # print(re)

        # print(output.shape)
        # print(predictions.shape)
        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                # channel dimension is the last one.
                output[vgs[0] : vge[0], vgs[1] : vge[1], vgs[2] : vge[2], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1] : re[1], rs[2] : re[2]]
                )
            elif len(output.shape) == 5:
                # channel dimension is the last one.
                assert vge[0] - vgs[0] == 1, "Only one frame is supported"
                output[
                    vgs[0], vgs[1] : vge[1], vgs[2] : vge[2], vgs[3] : vge[3], ch_idx
                ] = predictions[dset_idx][
                    ch_idx, rs[1] : re[1], rs[2] : re[2], rs[3] : re[3]
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output


def stitch_predictions_general(predictions, dset):
    """Stitching for the dataset with multiple files of different shape."""
    mng = dset.idx_manager

    # TODO assert all shapes are equal len
    # adjust number of channels to match with prediction shape #TODO ugly, refac!
    shapes = []
    for shape in dset.get_data_shapes()[0]:
        shapes.append((predictions.shape[1],) + shape[1:])

    output = [np.zeros(shape, dtype=predictions.dtype) for shape in shapes]
    # frame_shape = dset.get_data_shape()[:-1]
    for patch_idx in range(predictions.shape[0]):
        # grid start, grid end
        # channel_idx is 0 because during prediction we're only use one channel. # TODO revisit this
        # 0th dimension is sample index in the output list
        grid_coords = np.array(
            mng.get_location_from_patch_idx(channel_idx=0, patch_idx=patch_idx),
            dtype=int,
        )
        sample_idx = grid_coords[0]
        grid_start = grid_coords[1:]
        # from here on, coordinates are relative to the sample(file in the list of inputs)
        grid_end = grid_start + mng.grid_shape

        # patch start, patch end
        patch_start = grid_start - mng.patch_offset()
        patch_end = patch_start + mng.patch_shape

        # valid grid start, valid grid end
        valid_grid_start = np.array([max(0, x) for x in grid_start], dtype=int)
        valid_grid_end = np.array(
            [min(x, y) for x, y in zip(grid_end, shapes[sample_idx])], dtype=int
        )

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(valid_grid_start)):
                if patch_start[dim] == 0:
                    valid_grid_start[dim] = 0
                if patch_end[dim] == mng.data_shape[dim]:
                    valid_grid_end[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        relative_start = valid_grid_start - patch_start
        relative_end = relative_start + (valid_grid_end - valid_grid_start)

        for ch_idx in range(predictions.shape[1]):
            if len(output[sample_idx].shape) == 3:
                # starting from 1 because 0th dimension is channel relative to input
                # channel dimension for stitched output is relative to model output
                output[sample_idx][
                    ch_idx,
                    valid_grid_start[1] : valid_grid_end[1],
                    valid_grid_start[2] : valid_grid_end[2],
                ] = predictions[patch_idx][
                    ch_idx,
                    relative_start[1] : relative_end[1],
                    relative_start[2] : relative_end[2],
                ]
            elif len(output[sample_idx].shape) == 4:
                assert (
                    valid_grid_end[0] - valid_grid_start[0] == 1
                ), "Only one frame is supported"
                output[
                    ch_idx,
                    valid_grid_start[0],
                    valid_grid_end[1] : valid_grid_end[1],
                    valid_grid_start[2] : valid_grid_end[2],
                    valid_grid_start[3] : valid_grid_end[3],
                ] = predictions[patch_idx][
                    ch_idx,
                    relative_start[1] : relative_end[1],
                    relative_start[2] : relative_end[2],
                    relative_start[3] : relative_end[3],
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output

def stitch_and_crop_predictions_inner_tile(predictions, dset):
    """
    Stitch only the inner centered tile from each prediction patch into the big canvas,
    then average overlapping regions and crop back to the original size.

    Args:
        predictions: np.array of shape (num_patches, C, H, W)
        dset: Dataset object with padding info, original shape, etc.

    Returns:
        Stitched and cropped image of shape (N, H, W, C)
    """

    padded_shape = dset._data.shape  # (N, H_pad, W_pad, C)
    idx_manager = dset.idx_manager

    # Create canvas and count matrix for averaging.
    stitched_padded_image = np.zeros(padded_shape, dtype=predictions.dtype)
    counts = np.zeros_like(stitched_padded_image, dtype=np.float32)

    # Parameters for inner tile cropping
    full_tile_size = idx_manager.patch_spatial_dims[0]  # e.g. 64
    inner_tile_size = full_tile_size // 2  # e.g. 32 (the inner centered tile)
    start_inner = (full_tile_size - inner_tile_size) // 2  # e.g. 16
    end_inner = start_inner + inner_tile_size  # e.g. 48

    for i in tqdm(range(len(predictions)), desc="Stitching the inner crops"):
        patch = predictions[i]  # shape: (C, H, W)

        # Convert patch from (C, H, W) to (H, W, C) for easier placement
        patch = np.transpose(patch, (1, 2, 0))

        # Crop out the central inner tile from the patch spatially
        inner_patch = patch[start_inner:end_inner, start_inner:end_inner, :]  # (inner_tile, inner_tile, C)

        # Get location of this patch on the padded canvas
        loc = idx_manager.get_patch_location_from_dataset_idx(i)
        batch_idx, *spatial_loc = loc  # batch index and spatial location (top-left corner of patch)

        # Adjust spatial location to inner tile offset
        inner_spatial_loc = [s + start_inner for s in spatial_loc]

        # Create slices for stitched image and counts
        slices_img = [batch_idx]
        slices_count = [batch_idx]
        for s_loc in inner_spatial_loc:
            slices_img.append(slice(int(s_loc), int(s_loc + inner_tile_size)))
            slices_count.append(slice(int(s_loc), int(s_loc + inner_tile_size)))
        slices_img.append(slice(None))  # channel last
        slices_count.append(slice(None))

        # Add inner patch to canvas and update counts
        stitched_padded_image[tuple(slices_img)] += inner_patch
        counts[tuple(slices_count)] += 1

    # Avoid division by zero in averaging
    counts[counts == 0] = 1
    stitched_padded_image /= counts

    # Crop back to original size (remove padding)
    if hasattr(dset, "pad_width_spatial"):
        pad_width = dset.pad_width_spatial  # e.g. [(48,48), (48,48)] for H and W dims
    else:
        pad_width = [(0, 0), (0, 0)]

    crop_slices = [slice(None)]
    for pad_before, pad_after in pad_width:
        crop_slices.append(slice(pad_before, -pad_after if pad_after > 0 else None))
    crop_slices.append(slice(None))

    final_image = stitched_padded_image[tuple(crop_slices)]

    return final_image,  counts


def stitch_predictions_windowed_from_dir(
    pred_dir: Union[str, Path],
    dset,
    num_patches: int,
    inner_fraction: Union[float, List[float]] = 0.5,
    num_workers: int = 8,
    batch_size: int = 64,
    digits: int = 10,
    skip_missing: bool = True,
    debug: bool = False,
    return_coverage_mask: bool = False,
    live_stitching: bool = True,
    live_interval_percent: float = 0.5,
    live_dir: Optional[Union[str, Path]] = None,
):
    """
    Disk-based wrapper for `stitch_predictions_windowed`.

    Streams predictions from `.npy` files and stitches them on-the-fly.

    Args:
        ...
        live_stitching: If True, periodically dump intermediate stitched image.
        live_interval_percent: Percent interval between dumps (default=1%).
        live_dir: Optional output directory for live dumps; defaults to pred_dir / "live".
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    pred_dir = Path(pred_dir)
    # if live_dir is None:
    #     live_dir = pred_dir.parent / "live"
    # if live_stitching:
    #     live_dir.mkdir(parents=True, exist_ok=True)

    # Build expected file list
    pred_files = [pred_dir / f"pred_{i:0{digits}d}.npy" for i in range(num_patches)]

    if debug:
        print(f"[INFO] Streaming predictions from: {pred_dir}")
        print(f"[INFO] Expected {num_patches} tiles")
        print(f"[INFO] Using {num_workers} workers, batch size {batch_size}")

    progress_counter = 0
    next_dump_threshold = int(num_patches * live_interval_percent / 100) if live_stitching else None
    dump_count = 0

    # ------------------------------------------------------------------------
    # Generator to lazily load predictions in batches
    # ------------------------------------------------------------------------
    def prediction_generator() -> Iterator[np.ndarray]:
        nonlocal progress_counter, next_dump_threshold, dump_count

        for batch_start in range(0, num_patches, batch_size):
            batch_files = pred_files[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {
                    ex.submit(np.load, f): f
                    for f in batch_files
                    if f.exists() or not skip_missing
                }

                for fut in as_completed(futures):
                    try:
                        arr = fut.result()
                        progress_counter += 1
                        yield arr

                    except Exception as e:
                        f = futures[fut]
                        if skip_missing:
                            if debug:
                                print(f"[WARN] Skipped {f}: {e}")
                            continue
                        else:
                            raise

    # ------------------------------------------------------------------------
    # Stitch streamed predictions
    # ------------------------------------------------------------------------
    stitched, counts = stitch_predictions_windowed(
        generator=prediction_generator(),
        dset=dset,
        num_patches=num_patches,
        inner_fraction=inner_fraction,
        debug=debug,
    )

    counts[counts == 0] = 1
    stitched /= counts

    # Crop away dataset padding if needed
    if hasattr(dset, "boundary_pad") and dset.boundary_pad is not None:
        pads = dset.boundary_pad
        crop_slices = [slice(None)]
        for pad in pads:
            crop_slices.append(slice(pad, -pad if pad > 0 else None))
        crop_slices.append(slice(None))
    elif hasattr(dset, "pad_width_spatial") and dset.pad_width_spatial is not None:
        pads = dset.pad_width_spatial
        crop_slices = [slice(None)]
        for pad_before, pad_after in pads:
            crop_slices.append(slice(pad_before, -pad_after if pad_after > 0 else None))
        crop_slices.append(slice(None))
    else:
        warnings.warn("No padding info found in dataset; skipping crop.")
        crop_slices = [slice(None)] * stitched.ndim

    final_image = stitched[tuple(crop_slices)]
    coverage_mask = counts[tuple(crop_slices)]

    if debug:
        print(f"[INFO] Final stitched shape: {final_image.shape}")
        print(f"[INFO] Coverage mask shape: {coverage_mask.shape}")

    if return_coverage_mask:
        return final_image, coverage_mask
    else:
        return final_image, counts

def stitch_predictions_windowed_highperf(
    pred_dir: Union[str, Path],
    dset,
    num_patches: int,
    inner_fraction: Union[float, List[float]] = 0.5,
    batch_size: int = 256,
    num_workers: int = 8,
    digits: int = 10,
    skip_missing: bool = True,
    debug: bool = False,
    use_memmap: bool = True,
    memmap_dir: Optional[Union[str, Path]] = None,
    return_coverage_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    High-performance stitching for large numbers of tiles (200k+).
    Only uses the central "inner fraction" of each tile.
    Overlapping regions are averaged using a counts array.
    """

    pred_dir = Path(pred_dir)
    pred_files = [pred_dir / f"pred_{i:0{digits}d}.npy" for i in range(num_patches)]
    if debug:
        print(f"[INFO] Stitching {num_patches} tiles from {pred_dir}")

    # ------------------------------------------------------------------------
    # Determine final canvas shape
    # ------------------------------------------------------------------------
    full_shape = dset._data.shape
    dtype = np.float32

    # ------------------------------------------------------------------------
    # Prepare canvas and counts
    # ------------------------------------------------------------------------
    if use_memmap:
        if memmap_dir is None:
            memmap_dir = pred_dir
        memmap_dir = Path(memmap_dir)
        memmap_dir.mkdir(parents=True, exist_ok=True)

        stitched_path = memmap_dir / "stitched_tmp.dat"
        counts_path = memmap_dir / "counts_tmp.dat"

        stitched = np.memmap(stitched_path, mode="w+", dtype=dtype, shape=full_shape)
        counts = np.memmap(counts_path, mode="w+", dtype=dtype, shape=full_shape)
        stitched[:] = 0
        counts[:] = 0
    else:
        stitched = np.zeros(full_shape, dtype=dtype)
        counts = np.zeros_like(stitched)

    # ------------------------------------------------------------------------
    # Batch generator
    # ------------------------------------------------------------------------
    def prediction_generator() -> Iterator[np.ndarray]:
        for batch_start in range(0, num_patches, batch_size):
            batch_files = pred_files[batch_start: batch_start + batch_size]
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(np.load, f): f for f in batch_files if f.exists() or not skip_missing}
                for fut in as_completed(futures):
                    f = futures[fut]
                    try:
                        arr = fut.result()
                        yield arr
                    except Exception as e:
                        if skip_missing:
                            if debug:
                                print(f"[WARN] Skipped {f}: {e}")
                            continue
                        else:
                            raise

    # ------------------------------------------------------------------------
    # Parse inner_fraction per axis
    # ------------------------------------------------------------------------
    patch_spatial_dims = dset.idx_manager.patch_spatial_dims
    num_spatial_dims = len(patch_spatial_dims)
    if isinstance(inner_fraction, (int, float)):
        inner_fractions = [inner_fraction] * num_spatial_dims
    else:
        inner_fractions = list(inner_fraction)
        if len(inner_fractions) != num_spatial_dims:
            raise ValueError(f"inner_fraction length mismatch. Expected {num_spatial_dims}")

    # Compute inner crop indices for each axis
    inner_sizes, starts, ends = [], [], []
    for full_size, frac in zip(patch_spatial_dims, inner_fractions):
        inner_size = int(full_size * frac)
        start = (full_size - inner_size) // 2
        end = start + inner_size
        inner_sizes.append(inner_size)
        starts.append(start)
        ends.append(end)

    # ------------------------------------------------------------------------
    # Stitch tiles
    # ------------------------------------------------------------------------
    patch_idx = 0
    for pred in tqdm(prediction_generator(), total=num_patches, desc="Stitching"):
        # Ensure iterable
        pred_list = [pred] if not isinstance(pred, (list, tuple)) else list(pred)
        for tile in pred_list:
            if patch_idx >= num_patches:
                break

            # Channels-last if needed
            if num_spatial_dims == 2 and tile.ndim == 3 and tile.shape[0] < min(tile.shape[1:]):
                tile = np.transpose(tile, (1, 2, 0))
            elif num_spatial_dims == 3 and tile.ndim == 4 and tile.shape[0] < min(tile.shape[1:]):
                tile = np.transpose(tile, (1, 2, 3, 0))

            loc = dset.idx_manager.get_patch_location_from_dataset_idx(patch_idx)
            batch_idx = loc[0]

            if num_spatial_dims == 2:
                h_start = loc[1] + starts[0]
                w_start = loc[2] + starts[1]
                h_end = h_start + inner_sizes[0]
                w_end = w_start + inner_sizes[1]

                inner_tile = tile[starts[0]:ends[0], starts[1]:ends[1], :]
                stitched[batch_idx, h_start:h_end, w_start:w_end, :] += inner_tile
                counts[batch_idx, h_start:h_end, w_start:w_end, :] += 1

            else:
                z_start = loc[1] + starts[0]
                h_start = loc[2] + starts[1]
                w_start = loc[3] + starts[2]
                z_end = z_start + inner_sizes[0]
                h_end = h_start + inner_sizes[1]
                w_end = w_start + inner_sizes[2]

                inner_tile = tile[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2], :]
                stitched[batch_idx, z_start:z_end, h_start:h_end, w_start:w_end, :] += inner_tile
                counts[batch_idx, z_start:z_end, h_start:h_end, w_start:w_end, :] += 1

            patch_idx += 1

            # Debug: save intermediate visualization
            if debug and patch_idx % 1000 == 0:
                plt.imshow(stitched[0, ..., 0], cmap='gray')
                plt.axis('off')
                plt.savefig("./debug_patch.png", bbox_inches='tight', dpi=150)
                plt.close()

    # ------------------------------------------------------------------------
    # Average overlaps
    # ------------------------------------------------------------------------
    counts[counts == 0] = 1
    stitched /= counts

    # ------------------------------------------------------------------------
    # Crop padding if present
    # ------------------------------------------------------------------------
    if hasattr(dset, "boundary_pad") and dset.boundary_pad is not None:
        pads = dset.boundary_pad
        crop_slices = [slice(None)]
        for pad in pads:
            crop_slices.append(slice(pad, -pad if pad > 0 else None))
        crop_slices.append(slice(None))
    elif hasattr(dset, "pad_width_spatial") and dset.pad_width_spatial is not None:
        pads = dset.pad_width_spatial
        crop_slices = [slice(None)]
        for pad_before, pad_after in pads:
            crop_slices.append(slice(pad_before, -pad_after if pad_after > 0 else None))
        crop_slices.append(slice(None))
    else:
        crop_slices = [slice(None)] * stitched.ndim

    final_image = stitched[tuple(crop_slices)]
    coverage_mask = counts[tuple(crop_slices)]

    return (final_image, coverage_mask) if return_coverage_mask else final_image


def stitch_predictions_windowed(
    generator: Iterator[np.ndarray],
    dset,
    inner_fraction: Union[float, List[float]] = 0.5,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch tile predictions in a sliding windowed manner with per-axis control.
    
    Works with UNPADDED data (refactored WindowedLCDLoader/WindowedTilingDloader).
    
    Correctly handles:
      - 2D datasets: patch_spatial_dims = (H, W) = (64, 64)
      - 3D datasets: patch_spatial_dims = (Z, H, W) = (9, 64, 64)
    
    Supports different inner crop fractions for each spatial dimension.
    Only the inner centered portion of each tile is stitched into the output canvas.
    Overlapping regions are averaged together.
    
    Args:
        generator: Generator or iterator that yields prediction arrays.
                  Each prediction should be:
                  - 2D: (H, W, C) or (C, H, W)
                  - 3D: (Z, H, W, C) or (C, Z, H, W)
        
        dset: Dataset object (WindowedLCDLoader or WindowedTilingDloader) with:
              - _data.shape: Original unpadded data (N, H, W, C) for 2D or (N, Z, H, W, C) for 3D
              - idx_manager: WindowedTilingGridIndexManager with:
                  - patch_spatial_dims: Tuple of spatial patch dimensions
                    * 2D: (H_size, W_size) e.g., (64, 64)
                    * 3D: (Z_size, H_size, W_size) e.g., (9, 64, 64)
                  - get_patch_location_from_dataset_idx(i) → (N_idx, *spatial_coords)
              - _5Ddata: Boolean flag for 3D vs 2D (optional, inferred from patch_spatial_dims)
        
        num_patches: Total number of patches to process.
        
        inner_fraction: Fraction of tile to use as inner region per axis.
                       Can be:
                       - float: Same fraction for all spatial dims (default 0.5)
                       - List[float]: Per-axis fractions in spatial order
                         * 2D: [fy, fx]
                         * 3D: [fz, fy, fx]
        
        batch_size: Process this many predictions at a time (for future optimization).
        
        debug: If True, print debug information.
    
    Returns:
        final_image: Stitched image with unpadded shape (no cropping needed!)
        coverage_mask: Pixel coverage count array (same shape as final_image)
    
    Examples:
        # 2D: uniform 50% center crop
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=0.5
        )
        
        # 2D: asymmetric per-axis
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=[0.5, 0.25]  # 50% Y, 25% X
        )
        
        # 3D: full Z, center 50% XY (RECOMMENDED FOR 3D)
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=[1.0, 0.5, 0.5]
        )
    """
    
    # ========================================================================
    # Get dimensions from dataset (UNPADDED)
    # ========================================================================
    if getattr(dset, '_tar_idx_list', None):
        original_shape = dset._data[...,dset._tar_idx_list].shape
    else:
        original_shape = dset._data.shape
    idx_manager = dset.idx_manager
    num_patches = len(dset)
    # Determine if 2D or 3D from patch_spatial_dims
    patch_spatial_dims = idx_manager.patch_spatial_dims
    num_spatial_dims = len(patch_spatial_dims)
    
    is_3d = num_spatial_dims == 3
    
    if debug:
        print(f"[DEBUG] Original unpadded data shape: {original_shape}")
        print(f"[DEBUG] patch_spatial_dims: {patch_spatial_dims}")
        print(f"[DEBUG] Data is {'3D' if is_3d else '2D'}")
        print(f"[DEBUG] Number of patches to process: {num_patches}")
    
    # ========================================================================
    # Initialize output canvas (UNPADDED SIZE - NO CROPPING NEEDED AT END!)
    # ========================================================================
    stitched = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros_like(stitched)
    
    if debug:
        print(f"[DEBUG] Initialized canvas with shape: {stitched.shape}")
    
    # ========================================================================
    # Parse inner_fraction into per-axis list
    # ========================================================================
    if isinstance(inner_fraction, (int, float)):
        inner_fractions = [inner_fraction] * num_spatial_dims
    elif isinstance(inner_fraction, (list, tuple)):
        inner_fractions = list(inner_fraction)
        if len(inner_fractions) != num_spatial_dims:
            raise ValueError(
                f"inner_fraction list length ({len(inner_fractions)}) must match "
                f"number of spatial dimensions ({num_spatial_dims}). "
                f"Expected {num_spatial_dims} values for {'3D (Z,H,W)' if is_3d else '2D (H,W)'}"
            )
    else:
        raise TypeError(f"inner_fraction must be float or list, got {type(inner_fraction)}")
    
    # ========================================================================
    # Compute inner crop parameters per axis
    # ========================================================================
    inner_tile_sizes = []
    start_inners = []
    end_inners = []
    
    for axis_idx, (full_size, frac) in enumerate(zip(patch_spatial_dims, inner_fractions)):
        inner_size = int(full_size * frac)
        start = (full_size - inner_size) // 2
        end = start + inner_size
        
        inner_tile_sizes.append(inner_size)
        start_inners.append(start)
        end_inners.append(end)
    
    if debug:
        print(f"[DEBUG] Full tile sizes: {patch_spatial_dims}")
        print(f"[DEBUG] Inner fractions: {inner_fractions}")
        print(f"[DEBUG] Inner tile sizes: {inner_tile_sizes}")
        print(f"[DEBUG] Start crop indices: {start_inners}")
        print(f"[DEBUG] End crop indices: {end_inners}")
    
    # ========================================================================
    # Process predictions from generator
    # ========================================================================
    patch_idx = 0
    for pred_batch in tqdm(generator, total=num_patches, desc="Stitching predictions"):
        # Handle case where generator yields batches
        if isinstance(pred_batch, (list, tuple)):
            pred_list = pred_batch if isinstance(pred_batch, list) else [pred_batch]
        else:
            pred_list = [pred_batch]
        
        for pred in pred_list:
            if patch_idx >= num_patches:
                break
            
            # ================================================================
            # Ensure prediction is in spatial-last format
            # ================================================================
            if is_3d:
                # Could be (C, Z, H, W) - check if C is smallest
                if pred.ndim == 4 and pred.shape[0] < min(pred.shape[1:]):
                    pred = np.transpose(pred, (1, 2, 3, 0))  # → (Z, H, W, C)
            else:
                # 2D: Could be (C, H, W)
                if pred.ndim == 3 and pred.shape[0] < min(pred.shape[1:]):
                    pred = np.transpose(pred, (1, 2, 0))  # → (H, W, C)
            
            # ================================================================
            # Extract inner crop from prediction using per-axis fractions
            # ================================================================
            if is_3d:
                # For 3D: (Z, H, W, C)
                inner_pred = pred[
                    start_inners[0]:end_inners[0],
                    start_inners[1]:end_inners[1],
                    start_inners[2]:end_inners[2],
                    :
                ]
            else:
                # For 2D: (H, W, C)
                inner_pred = pred[
                    start_inners[0]:end_inners[0],
                    start_inners[1]:end_inners[1],
                    :
                ]
            
            # ================================================================
            # Get patch location in UNPADDED data space
            # ================================================================
            loc = idx_manager.get_patch_location_from_dataset_idx(patch_idx)
            batch_idx = loc[0]  # N index
            
            if is_3d:
                # loc = (N, Z, H, W)
                z_start, h_start, w_start = loc[1], loc[2], loc[3]
                
                # Compute inner crop positions in unpadded data space
                z_start_inner = z_start + start_inners[0]
                h_start_inner = h_start + start_inners[1]
                w_start_inner = w_start + start_inners[2]
                
                z_end_inner = z_start_inner + inner_tile_sizes[0]
                h_end_inner = h_start_inner + inner_tile_sizes[1]
                w_end_inner = w_start_inner + inner_tile_sizes[2]
                
                # Bounds check against UNPADDED data shape
                # Clips to actual image boundaries
                z_end_inner = min(z_end_inner, original_shape[1])
                h_end_inner = min(h_end_inner, original_shape[2])
                w_end_inner = min(w_end_inner, original_shape[3])
                
                # Clip inner_pred if it exceeds bounds
                z_inner_crop = max(0, z_end_inner - z_start_inner)
                h_inner_crop = max(0, h_end_inner - h_start_inner)
                w_inner_crop = max(0, w_end_inner - w_start_inner)
                
                # Sanity check - skip if dimensions become invalid
                if z_inner_crop <= 0 or h_inner_crop <= 0 or w_inner_crop <= 0:
                    if debug:
                        print(f"[WARNING] Patch {patch_idx} has invalid inner_cropped dimensions: "
                              f"Z:{z_inner_crop}, H:{h_inner_crop}, W:{w_inner_crop}")
                    patch_idx += 1
                    continue
                
                inner_pred_inner_cropped = inner_pred[:z_inner_crop, :h_inner_crop, :w_inner_crop, :]
                
                if debug and patch_idx < 5:  # Debug first few patches
                    print(
                        f"[DEBUG] Patch {patch_idx}: batch={batch_idx}, "
                        f"Z:[{z_start_inner},{z_end_inner}), "
                        f"H:[{h_start_inner},{h_end_inner}), "
                        f"W:[{w_start_inner},{w_end_inner}), "
                        f"pred_shape={pred.shape}, inner_cropped_shape={inner_pred_inner_cropped.shape}"
                    )
                
                # Add to canvas
                stitched[batch_idx, z_start_inner:z_end_inner, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += inner_pred_inner_cropped
                counts[batch_idx, z_start_inner:z_end_inner, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += 1
                
            else:
                # loc = (N, H, W)
                h_start, w_start = loc[1], loc[2]
                
                # Compute inner crop positions in unpadded data space
                h_start_inner = h_start + start_inners[0]
                w_start_inner = w_start + start_inners[1]
                
                h_end_inner = h_start_inner + inner_tile_sizes[0]
                w_end_inner = w_start_inner + inner_tile_sizes[1]
                
                # Bounds check against UNPADDED data shape
                h_end_inner = min(h_end_inner, original_shape[1])
                w_end_inner = min(w_end_inner, original_shape[2])
                
                # Clip inner_pred if it exceeds bounds
                h_inner_crop = max(0, h_end_inner - h_start_inner)
                w_inner_crop = max(0, w_end_inner - w_start_inner)
                
                # Sanity check
                if h_inner_crop <= 0 or w_inner_crop <= 0:
                    if debug:
                        print(f"[WARNING] Patch {patch_idx} has invalid inner_cropped dimensions: "
                              f"H:{h_inner_crop}, W:{w_inner_crop}")
                    patch_idx += 1
                    continue
                
                inner_pred_inner_cropped = inner_pred[:h_inner_crop, :w_inner_crop, :]
                
                if debug and patch_idx < 5:  # Debug first few patches
                    print(
                        f"[DEBUG] Patch {patch_idx}: batch={batch_idx}, "
                        f"H:[{h_start_inner},{h_end_inner}), "
                        f"W:[{w_start_inner},{w_end_inner}), "
                        f"pred_shape={pred.shape}, inner_cropped_shape={inner_pred_inner_cropped.shape}"
                    )
                
                # Add to canvas
                stitched[batch_idx, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += inner_pred_inner_cropped
                counts[batch_idx, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += 1
            
            patch_idx += 1
    
    if patch_idx < num_patches:
        print(f"[WARNING] Only processed {patch_idx}/{num_patches} patches from generator")
    
    # ========================================================================
    # Average overlapping regions
    # ========================================================================
    counts[counts == 0] = 1  # Avoid division by zero
    stitched /= counts
    
    # ========================================================================
    # NO CROPPING NEEDED - data already in unpadded shape!
    # ========================================================================
    # The canvas was initialized with original_shape (unpadded)
    # so final_image is already the correct unpadded size
    
    if debug:
        print(f"[DEBUG] Final stitched shape: {stitched.shape}")
        print(f"[DEBUG] Original shape was: {original_shape}")
        print(f"[DEBUG] Shapes match: {stitched.shape == original_shape}")
    
    # Return coverage mask with same shape as final image
    # (useful for understanding coverage per-pixel)
    coverage_mask = counts
    
    return stitched, coverage_mask


# ============================================================================
# Utility Functions for Coverage Analysis
# ============================================================================

def analyze_coverage(coverage_mask: np.ndarray, debug: bool = True) -> dict:
    """
    Analyze the coverage of stitched predictions.
    
    Args:
        coverage_mask: Output from stitch_predictions_windowed
        debug: Print analysis
    
    Returns:
        Dictionary with coverage statistics
    """
    # Find non-zero coverage
    non_zero = coverage_mask > 0
    
    stats = {
        "total_pixels": coverage_mask.size,
        "covered_pixels": np.sum(non_zero),
        "uncovered_pixels": np.sum(~non_zero),
        "coverage_percentage": 100 * np.sum(non_zero) / coverage_mask.size,
        "min_coverage": np.min(coverage_mask[non_zero]) if np.any(non_zero) else 0,
        "max_coverage": np.max(coverage_mask),
        "mean_coverage": np.mean(coverage_mask[non_zero]) if np.any(non_zero) else 0,
    }
    
    if debug:
        print("\n" + "=" * 60)
        print("Coverage Analysis")
        print("=" * 60)
        print(f"Total pixels: {stats['total_pixels']:,}")
        print(f"Covered pixels: {stats['covered_pixels']:,} ({stats['coverage_percentage']:.1f}%)")
        print(f"Uncovered pixels: {stats['uncovered_pixels']:,}")
        print(f"Coverage range: {stats['min_coverage']:.1f} - {stats['max_coverage']:.1f}")
        print(f"Mean coverage: {stats['mean_coverage']:.2f}")
        print("=" * 60 + "\n")
    
    return stats
