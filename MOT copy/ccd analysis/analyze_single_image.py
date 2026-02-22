# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:45:04 2022

@author: Lothar Maisenbacher/Berkeley/MPQ.

Run beam profile analysis on a single image:
Beam profile moments, ellipse fit, 2D Gaussian fit, D4σ method
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
from pathlib import Path

import beam_caustic

# Default values
# Relative tolerance condition for convergence of D4σ method
d4s_rel_tol = 1e-5
# Absolute tolerance condition for convergence of D4σ method
d4s_abs_tol = 1e-2
# Save plots
save_plots = True

# Input file
dir_data = (
    'C:/Users/Lothar/Unitrap Dropbox/Projects/Collimator for alpha/comparing_three_collimators')
filename_image = '1_far.tif'
# Pixel size of images (mm)
pixel_size = 3.45e-3
d4s_rel_tol = None
d4s_abs_tol = 1

# Scaling factor for ellipse to be used for background subtraction
bkg_ellipse_axes_scaling = 1
# Self-confined width of D4σ method
self_conf_width = 3

filepath = Path(dir_data, filename_image)
image = tifffile.imread(filepath)
# If multiple channels are present, use first channel only
if len(image.shape) == 3:
    image = image[:, :, 0]
Y, X = np.indices(image.shape)

# Create image from 2D Gaussian
# image = np.zeros((500, 800))
# Y, X = np.indices(image.shape)
# xc_ = image.shape[1]/2
# yc_ = image.shape[0]/2
# dx_ = 50
# dy_ = 20
# orientation_ = 1*np.pi/4*0
# gaussian_2d_rot = beam_caustic.get_gaussian_2d(1., xc_, yc_, dx_, dy_, orientation_, 0.)
# image = gaussian_2d_rot(X, Y)

# Plot image
fig_im = plt.figure('Image', clear=True, constrained_layout=True)
ax_im = fig_im.gca()
h = ax_im.imshow(image)
plt.colorbar(h)

fig_hist = plt.figure('Histogram', clear=True, constrained_layout=True)
ax_hist = fig_hist.gca()
h = ax_hist.hist(image.ravel(), bins=100, log=True)
ax_hist.set_title(f'Counts/pixel distribution of image ({filename_image})')
ax_hist.set_xlabel('Counts/pixel')
ax_hist.set_ylabel('Occurence')

# Determine ISO beam parameters from image before correcting for background.
# If there is a background present, this will give erroneous results for the beam width.
xc_iso_wbkg, yc_iso_wbkg, dx_iso_wbkg, dy_iso_wbkg, orientation_iso_wbkg = (
    beam_caustic.determine_iso_beam_parameters(image, fixed_orientation=None))
print(
    'Results from moments of non-background-corrected image (x0, y0, D4σx, D4σy, phi): '
    +f'{xc_iso_wbkg:.2e} px, {yc_iso_wbkg:.2e} px, {dx_iso_wbkg:.2e} px, {dy_iso_wbkg:.2e} px'
    +f', {orientation_iso_wbkg:.2f} deg')

# Find ellipse for background subtraction
xc_ell, yc_ell, minor_saxis_ell, major_saxis_ell, orientation_ell = (
    beam_caustic.find_ellipse(image))
print(
    'Results from ellipse fit (x0, y0, minor semi axis, major semi axis, phi): '
    +f'{xc_ell:.2e} px, {yc_ell:.2e} px, {minor_saxis_ell:.2e} px, {major_saxis_ell:.2e} px'
    +f', {orientation_ell:.2f} deg')

# Show ellipse
image_blank = np.zeros(image.shape)
image_ellipse = cv2.ellipse(
    image_blank,
    (int(xc_ell), int(yc_ell)), (int(minor_saxis_ell), int(major_saxis_ell)),
    orientation_ell+90, 0, 360, 1, 10)
plt.figure('Ellipse', clear=True, constrained_layout=True)
plt.imshow(image_ellipse)

# Find mask matching ellipse, scaled in size by factor `bkg_ellipse_axes_scaling`
ellipse_mask = beam_caustic.get_ellipse_mask(
    xc_ell, yc_ell,
    bkg_ellipse_axes_scaling*minor_saxis_ell, bkg_ellipse_axes_scaling*major_saxis_ell,
    orientation_ell, image.shape)

# Show ellipse mask
image_masked = image.copy()
image_masked[ellipse_mask] = 0
# image_masked[image_masked < 5000] = 0
fig_masked = plt.figure('Masked', clear=True, constrained_layout=True)
ax_masked = fig_masked.gca()
h = ax_masked.imshow(image_masked)
plt.colorbar(h)

fig_hist_bkg = plt.figure('Histogram of background', clear=True, constrained_layout=True)
ax_hist_bkg = fig_hist_bkg.gca()
h = ax_hist_bkg.hist(image_masked[image_masked > 0].ravel(), bins=100, log=True)
ax_hist.set_title(f'Counts/pixel distribution of background ({filename_image})')
ax_hist.set_xlabel('Counts/pixel')
ax_hist.set_ylabel('Occurence')

im_max = image.max()
im_avg = image.mean()
im_std = image.std()
im_err = im_std/np.sqrt(image.size)
print(
    'Image mean, SD, SEM, max (cts/px):'
    +f' {im_avg:.1e}, {im_std:.1e}, {im_err:.1e}, {im_max:.1e}')

# Use area outside scaled ellipse to estimate background
if len(image[~ellipse_mask]) != 0:
    bkg_max = image[~ellipse_mask].max()
    bkg_avg = image[~ellipse_mask].mean()
    bkg_std = image[~ellipse_mask].std()
    bkg_err = bkg_std/np.sqrt(np.sum(~ellipse_mask))
    print(
        'Background mean, SD, SEM, max (cts/px):'
        +f' {bkg_avg:.1e}, {bkg_std:.1e}, {bkg_err:.1e}, {bkg_max:.1e}')
    print(f'Mean background/peak image: {bkg_avg/image.max():.1e}')
else:
    print('No pixels outside scaled ellipse left to estimate background from')

# Subtract average background from image
# image_bkg_corr = image
image_bkg_corr = image-bkg_avg
# image_bkg_corr[image_bkg_corr < 0] = 0.

# Plot background-corrected image
fig_bkg_corr = plt.figure('Background-corrected image', clear=True, constrained_layout=True)
ax_bkg_corr = fig_bkg_corr.gca()
h = ax_bkg_corr.imshow(image_bkg_corr)
plt.colorbar(h)

# Determine ISO beam parameters from background-corrected image
xc_iso, yc_iso, dx_iso, dy_iso, orientation_iso = beam_caustic.determine_iso_beam_parameters(
    image_bkg_corr, fixed_orientation=None)
print(
    'Results from moments of background-corrected image (x0, y0, D4σx, D4σy, phi): '
    +f'{xc_iso:.2e} px, {yc_iso:.2e} px, {dx_iso:.3e} px, {dy_iso:.3e} px'
    +f', {orientation_iso:.2f} deg')

# Fit 2D Gaussian with free orientation
fit_result, fit_dict, gaussian_2d = beam_caustic.fit_gaussian_2d(
    image_bkg_corr,
    pstart=[
        image_bkg_corr.max(),
        xc_ell, yc_ell, minor_saxis_ell, major_saxis_ell, orientation_ell,
        0.])
print(
    'Results of 2D Gaussian fit with free orientation (x0, y0, D4σx, D4σy, phi):\n'
    +f'{fit_dict["x0_Value"]:.2e} px, {fit_dict["y0_Value"]:.2e} px'
    +f', {2*fit_dict["w0x_Value"]:.3e} px, {2*fit_dict["w0y_Value"]:.3e} px'
    +f', {fit_dict["Orientation_Value"]:.2f} deg')
print(
    '1/e^2 radius from 2D Gaussian fit: major/minor/simple average:\n'
    +f'{pixel_size*fit_dict["w0x_Value"]:.3e} mm/{pixel_size*fit_dict["w0y_Value"]:.3e} mm'
    +f'/{pixel_size*np.mean([fit_dict["w0x_Value"], fit_dict["w0y_Value"]]):.3e} mm'
    )
beam_center = [fit_dict["x0_Value"], fit_dict["y0_Value"]]
beam_center_px = np.round(beam_center).astype(int)

# Fit 2D Gaussian with fixed orientation along x-axis
# fit_result, fit_dict, gaussian_2d = beam_caustic.fit_gaussian_2d(
#     image_bkg_corr,
#     pstart=[
#         image_bkg_corr.max(),
#         xc_ell, yc_ell, minor_saxis_ell, major_saxis_ell, 0.],
#     fixed_orientation=0.)
# print(
#     'Results of 2D Gaussian fit oriented along x-axis (x0, y0, D4σx, D4σy, phi): '
#     +f'{fit_dict["x0_Value"]:.2e} px, {fit_dict["y0_Value"]:.2e} px'
#     +f', {2*fit_dict["w0x_Value"]:.2e} px, {2*fit_dict["w0y_Value"]:.2e} px'
#     +f', {fit_dict["Orientation_Value"]:.2f} deg')

# D4σ method
xc_d4s, yc_d4s, dx_d4s, dy_d4s, orientation_d4s, iterations_d4s, converged_d4s = (
    beam_caustic.run_d4sigma_method(
        image_bkg_corr, self_conf_width=self_conf_width, debug=True,
        ignore_crop_error=True,
        rel_tol=d4s_rel_tol, abs_tol=d4s_abs_tol,
        # start_values=[xc_ell, yc_ell, major_saxis_ell, major_saxis_ell],
        ))
print(
    'Results of D4σ method (x0, y0, D4σx, D4σy, phi): '
    +f'{xc_d4s:.2e} px, {yc_d4s:.2e} px, {dx_d4s:.3e} px, {dy_d4s:.3e} px'
    +f', {orientation_d4s:.2f} deg')

#%% Plot results of 2D Gaussian fit

fig_2dgauss, axs_2dgauss = plt.subplots(
    3, 1, num='2D Gaussian fit', figsize=(6, 10), clear=True, constrained_layout=True)
image_fit = gaussian_2d(*fit_result['Popt'])(*np.flipud(np.indices(image_bkg_corr.shape)))

for (ax, im_, im_cmap, label, cmap) in zip(
        axs_2dgauss,
        [image_bkg_corr, image_fit, (image_fit-image_bkg_corr)],
        [image_bkg_corr, image_bkg_corr, None],
        ['Intensity (arb. u.)', 'Intensity (arb. u.)', 'Fit residuals'],
        ['gray', 'gray', 'coolwarm']
        ):

    h_im = ax.imshow(
        im_,
        cmap=cmap,
        vmin=im_cmap.min() if im_cmap is not None else None,
        vmax=im_cmap.max() if im_cmap is not None else None,
        )
    ax.set_ylabel('y (px)')
    h_cbar = plt.colorbar(h_im, ax=ax)
    h_cbar.set_label(label)

axs_2dgauss[1].set_title(
    '2D Gaussian fit:\n'
    +rf'$w_{{0,\mathrm{{major}}}}$ = {fit_dict["w0x_Value"]*pixel_size:.2f} mm'
    +rf', $w_{{0,\mathrm{{minor}}}}$ = {fit_dict["w0y_Value"]*pixel_size:.2f} mm'
    )
axs_2dgauss[2].set_title('Fit residuals')
axs_2dgauss[-1].set_xlabel('x (px)')

if save_plots:
    fig_2dgauss.savefig(str(filepath)+'_2dgauss.png')

#%% Plot 1D cuts

fig_1dcuts, axs_1dcuts = plt.subplots(
    2, 1, num='1D cuts', figsize=(6, 6), clear=True, constrained_layout=True)

# Prepare 1D fit
import pyhs.fit
myFit = pyhs.fit.Fit()
fit_func_id = 'GaussianBeam'
fit_func = myFit.fitFuncs[fit_func_id]

ix = beam_center_px[0]
iy = beam_center_px[1]
# Average over di pixels perpendicular to cut direction
di = 5

x_pixels = np.arange(image_bkg_corr.shape[1])
y_pixels = np.arange(image_bkg_corr.shape[0])

# Cut along x
im_cut_x = np.mean(image_bkg_corr[iy-di:iy+di+1, :], axis=0)
im_fit_cut_x = np.mean(image_fit[iy-di:iy+di+1, :], axis=0)
# Fit with 1D Gaussian
pstart = [fit_dict['x0_Value'], fit_dict['w0x_Value'], np.max(im_cut_x), 0]
fit_gaussian, _ = myFit.do_fit(
    fit_func, x_pixels, im_cut_x, np.ones(len(x_pixels)),
    warning_as_error=False,
    pstart=pstart,
    )
im_fit_1d_cut_x = fit_func['Func'](x_pixels, *fit_gaussian['Popt'])
sr_fit_1d_cut_x = myFit.fit_result_to_series(fit_gaussian, fit_func)
# Cut along y
im_cut_y = np.mean(image_bkg_corr[:, ix-di:ix+di+1], axis=1)
im_fit_cut_y = np.mean(image_fit[:, ix-di:ix+di+1], axis=1)
# Fit with 1D Gaussian
pstart = [fit_dict['y0_Value'], fit_dict['w0y_Value'], np.max(im_cut_y), 0]
fit_gaussian, _ = myFit.do_fit(
    fit_func, y_pixels, im_cut_y, np.ones(len(y_pixels)),
    warning_as_error=False,
    pstart=pstart,
    )
im_fit_1d_cut_y = fit_func['Func'](y_pixels, *fit_gaussian['Popt'])
sr_fit_1d_cut_y = myFit.fit_result_to_series(fit_gaussian, fit_func)

axs_1dcuts[0].plot(
    x_pixels,
    im_cut_x,
    label='Cut through data',
    )
h1_cut_fit = axs_1dcuts[0].plot(
    x_pixels,
    im_fit_cut_x,
    linewidth=2,
    label='Cut through 2D fit'
    )
h1_cut_fit_1d = axs_1dcuts[0].plot(
    x_pixels, im_fit_1d_cut_x,
    linestyle='-', linewidth=2,
    label=(
        '1D fit to cut:\n'
        +rf'$w_{{0,\mathrm{{x}}}}$ = {sr_fit_1d_cut_x["w_Value"]*pixel_size:.2f} mm'),
    )
axs_1dcuts[0].set_xlabel('x (px)')
axs_1dcuts[0].set_ylabel('Intensity (arb. u.)')
axs_1dcuts[0].legend()

axs_1dcuts[1].plot(
    y_pixels,
    im_cut_y,
    label='Cut through data',
    )
h1_cut_fit = axs_1dcuts[1].plot(
    y_pixels,
    im_fit_cut_y,
    linewidth=2,
    label='Cut through 2D fit'
    )
h1_cut_fit_1d = axs_1dcuts[1].plot(
    y_pixels, im_fit_1d_cut_y,
    linestyle='-', linewidth=2,
    label=(
        '1D fit to cut:\n'
        +rf'$w_{{0,\mathrm{{x}}}}$ = {sr_fit_1d_cut_y["w_Value"]*pixel_size:.2f} mm'),
    )
axs_1dcuts[1].set_xlabel('y (px)')
axs_1dcuts[1].set_ylabel('Intensity (arb. u.)')
axs_1dcuts[1].legend()

if save_plots:
    fig_1dcuts.savefig(str(filepath)+'_1dcuts.png')
