# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:46:25 2022

@author: Lothar Maisenbacher/Berkeley/MPQ.

Incorporating work of Vitaly Wirthl/MPQ and Alexander Hertlein/MPQ.

Functions for the analysis of images of the beam profile of laser beams,
such as taken with a beam profiler or camera,
to determine the beam position, widths, and orientation.

Import `cv2` is from package `opencv-python` (install with `pip install opencv-python`),
which wraps cv2 library.
"""

import numpy as np
import scipy.optimize
import cv2

def find_ellipse(image):
    """
    Find an ellipse in the image `image` (2-D array) by first thresholding the
    image using Otsu's method and then fitting an ellipse to the contours.

    Parameters
    ----------
    image : ndarray
        2-D array representing the intensity of a laser beam.

    Returns
    -------
    xc : float
        Center of ellipse along x-axis.
    yc : float
        Center of ellipse along y-axis.
    minor_saxis : float
        Semi minor axis.
    major_saxis : float
        Semi major axis.
    orientation: float
        Angle (degree) of major axis from the x-axis towards to y-axis.
    """
    # Convert to 8-bit image, scale first to use full 8-bit range
    image8 = (image/image.max()*(2**8-1)).astype(np.uint8)
    # blur = cv2.GaussianBlur(image, (5, 5), 0)
    blur = image8
    # Threshold image using Otsu's method
    ret, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    xc, yc, minor_saxis, major_saxis, orientation = 0, 0, 0, 0, 0
    ellipse_found = False
    if len(contours) != 0:
        for cont in contours:
            if len(cont) < 5:
                continue
            (xc_, yc_), (minor_saxis_, major_saxis_), orientation_ = cv2.fitEllipse(cont)
            ellipse_found = True
            if minor_saxis_ > minor_saxis and major_saxis_ > major_saxis:
                (xc, yc), (minor_saxis, major_saxis), orientation = (
                    (xc_, yc_), (minor_saxis_, major_saxis_), orientation_)
    orientation -= 90.

    if not ellipse_found:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        return xc, yc, minor_saxis, major_saxis, orientation

def get_ellipse_mask(xc, yc, minor_saxis, major_saxis, orientation, mask_shape):
    """
    Get array mask in shape of ellipse.

    Parameters
    ----------
    xc : float
        Center of ellipse along x-axis.
    yc : float
        Center of ellipse along y-axis.
    minor_saxis : float
        Semi minor axis.
    major_saxis : float
        Semi major axis.
    orientation: float
        Angle (degree) of major axis from the x-axis towards to y-axis.
    mask_shape: (int, int)
        Shape of mask to return.

    Returns
    -------
    ellipse_mask : 2-D array of shape `mask_shape` of dtype bool
        Mask that is True inside ellipse and False outside.
    """
    orientation_rad = np.deg2rad(orientation)
    Y, X = np.indices(mask_shape)
    xr = -(X-xc)*np.sin(orientation_rad) + (Y-yc)*np.cos(orientation_rad)
    yr = -(X-xc)*np.cos(orientation_rad) - (Y-yc)*np.sin(orientation_rad)
    ellipse_mask = (
        (xr/minor_saxis)**2
        +(yr/major_saxis)**2 < 1)
    return ellipse_mask

def determine_iso_beam_parameters(image, fixed_orientation=None):
    """
    Determine laser beam center, 4-sigma widths, and orientation (tilt) of beam according to
    ISO 11146 standard (https://www.iso.org/obp/ui/#iso:std:iso:11146:-1:ed-2:v1:en)
    for image (intensity) `image` (2-D array), using the first and second moments of the intensity
    distribution.
    As is convention for image files, the first dimension of `data` is taken to be the y-axis,
    and the second dimension is taken to be the x-axis.

    Parameters
    ----------
    image : ndarray
        2-D array representing the intensity of a laser beam.
    fixed_orientation : float, default None
        Fix the orientation of the principal axis from the x-axis to the angle given here
        (in degree). If None is passed instead, the orientation is determined from the image
        (default).

    Returns
    -------
    xc : float
        Center of mass along x-axis.
    yc : float
        Center of mass along y-axis.
    dx : float
        4-sigma beam width along first principal axis.
    dy : float
        4-sigma beam width along second principal axis.
    orientation: float or None, optional
        Angle (degree) of first principal axis from the x-axis towards to y-axis.
    """
    # x- and y-dimensions of data
    y_dim, x_dim = image.shape
    # Prepare x- and y-axis vectors
    X = np.arange(x_dim, dtype=float)
    Y = np.arange(y_dim, dtype=float)
    # Total of all pixels
    total = image.sum()
    # Find first moment (center of weight) along x-direction
    xc = np.einsum('i,ji->', X, image)/total
    # Find first moment (center of weight) along y-direction
    yc = np.einsum('i,ij->', Y, image)/total

    # Find second moments (variances) (see Section 3.2 of ISO 11146-1)
    # Second moment σ_x^2 along x-direction (Eq. (3) of ISO 11146-1)
    xx = np.einsum('i,ji->', (X-xc)**2, image)/total
    # Second moment σ_y^2 along y-direction (Eq. (4) of ISO 11146-1)
    yy = np.einsum('j,ji->', (Y-yc)**2, image)/total
    # Second moment σ_xy^2 (Eq. (5) of ISO 11146-1)
    # Note from ISO 11146-1:
    # "σ_xy^z is a symbolic notation, and not a true square. This quantity can take positive,
    # negative or zero value."
    xy = np.einsum('i,j,ji->', (X-xc), (Y-yc), image)/total

    if fixed_orientation is None:
        # Determine orientation of principal axes (see Eqs. (15-16) of ISO 11146-1)
        if xx == yy:
            orientation = np.sign(xy)*np.pi/4
        else:
            orientation = np.arctan(2*xy/(xx-yy))/2
    else:
        orientation = np.deg2rad(fixed_orientation)
    orientation_deg = np.rad2deg(orientation)

    # 4-sigma beam width (beam width according to ISO 11146) in the direction of the principal axes
    # or direction of fixed orientation (if given),
    # based on Eqs. (18-20) of ISO 11146-1
    dx = 2*np.sqrt(2)*np.sqrt(xx+yy+(xx-yy)*np.sqrt(1+np.tan(2*orientation)**2))
    dy = 2*np.sqrt(2)*np.sqrt(xx+yy-(xx-yy)*np.sqrt(1+np.tan(2*orientation)**2))

    return xc, yc, dx, dy, orientation_deg

def run_d4sigma_method(image_bkg_corr, self_conf_width=3, max_iterations=100, debug=False,
                       ignore_crop_error=False, start_values=None,
                       rel_tol=1e-5, abs_tol=1e-2):
    """
    Use D4σ method to estimate laser beam widths, as described in the ISO 11146 standard.
    For the D4σ method to give reliable results (or even converge), it is extremely important that
    the image is free from background or has been corrected for background.

    Parameters
    ----------
    image_bkg_corr : ndarray
        2-D array representing the background-corrected intensity of a laser beam.
    self_conf_width : float, optional
        Self-confined width of the D4σ method. For each iteration, the size of the area over which
        the moments are determined is set to be 4-sigma beam width times the self-contained width.
        The default is 3, which is the value defined in the ISO 11146 standard.
    max_iterations : int, optional
        Maximum iterations of D4σ method. The default is 100.
    debug : bool, optional
        If set to True, additionial debug messages are printed. The default is False.
    ignore_crop_error : bool, optional
        If set to True, the iterations are not stopped when a crop error is encountered,
        which occurs when the image is not large enough to contain the area over which the moments
        should be determined. The default is False.
    start_values : array_like or None, optional
        Optional start values define the initial area over which the moments are determined:
            `pstart=[xc, yc, dx, dy]`,
        where `xc` and `yc` are the x- and y-center of the area, and `dx` and `dy` are widths of the
        area in the x- and y-direction.
        If set to None (default), the start values are estimated with an ellipse fit
        (see `find_ellipse`) to `image_bkg_corr`.
    rel_tol : float or None, optional
        Relative tolerance condition for convergence of D4σ method:
        Condition is met when relative change from previous iteration for all estimated parameters
        is below `rel_tol`,
        i.e., `p-p_p < rel_tol*p`, where `p_p` is the result from the previous and `p` is the
        result of the current iteration.
        If set to None, this condition is not used.
        Default is 1e-5.
    abs_tol : float or None, optional
        Absolute tolerance condition for convergence of D4σ method:
        Condition is met when absolute change from previous iteration for all estimated parameters
        is below `abs_tol`,
        i.e., `|p-p_p| < abs_tol`, where `p_p` is the result from the previous and `p` is the
        result of the current iteration.
        The units of `abs_tol` are those of the input data `image_bkg_corr`.
        If set to None, this condition is not used.
        Default is 1e-2.

    Returns
    -------
    xc : float
        Center of mass along x-axis.
    yc : float
        Center of mass along y-axis.
    dx : float
        4-sigma beam width along x-axis.
    dy : float
        4-sigma beam width along y-axis.
    orientation : float
        Angle (degree) of major axis from the x-axis towards to y-axis, determined from converged
        cropped area of image.
    iterations: int
        Number of iterations performed. If the D4σ method did not converged, this will be equal
        to maximum number of iterations, `max_iterations`.
    converged: bool
        True if D4σ method converged. If false, the method was stopped after `max_iterations`
        iterations.
    """
    # xc, yc, dx, dy, _ = determine_iso_beam_parameters(image_bkg_corr, fixed_orientation=0.)
    xc_ell, yc_ell, minor_saxis_ell, major_saxis_ell, orientation_ell = find_ellipse(image_bkg_corr)
    if debug:
        print(
            'Results from ellipse fit to determine start values for D4σ method'
            +' (x0, y0, minor semi axis, major semi axis, phi): '
            +f'{xc_ell:.2e} px, {yc_ell:.2e} px, {minor_saxis_ell:.2e} px, {major_saxis_ell:.2e} px'
            +f', {orientation_ell:.2f} deg')
    if start_values is None:
        p = xc_ell, yc_ell, major_saxis_ell, major_saxis_ell
        if debug:
            print('Using ellipse center and major semi axis as start values for D4σ method')
    else:
        p = start_values
        if debug:
            print('Using user-supplied start values for D4σ method')
    p = np.atleast_1d(p)
    # Setting variables for iterative determination of beam center and width using D4σ method
    p_p = np.zeros(4)
    p_p[:] = np.nan
    iterations = 0
    converged = True
    image_cropped = image_bkg_corr

    def check_tol(p, p_p, rel_tol, abs_tol):
        """
        Check if parameter estimation has converged within relative tolerance `rel_tol` (float) and
        absolute tolerance `abs_tol` (float), with the result from the current and previous
        iteration given by `p` and `p_p`, respectively (list-like or float).
        """
        p = np.atleast_1d(p)
        p_p = np.atleast_1d(p_p)
        # Relative tolerance condition:
        # Condition is met when relative change from previous iteration is below `rel_tol`,
        # i.e., `p-p_p < rel_tol*p`, where `p_p` is the result from the previous and `p` is the
        # result of the current iteration.
        # Similar to `FTOL` `LMDIF` routine of `MINPACK`
        # (or, equivalently, `ftol` of the Python wrapper `scipy.optimize.least_squares`).
        if rel_tol is not None:
            rel_tol_met = p-p_p < rel_tol*p
        else:
            rel_tol_met = np.ones(len(p), dtype=bool)
        # Absolute tolerance condition
        # Condition is met when absolute change from previous iteration is below `abs_tol`,
        # i.e., `|p-p_p| < abs_tol`, where `p_p` is the result from the previous and `p` is the
        # result of the current iteration.
        if abs_tol is not None:
            abs_tol_met = np.abs(p-p_p) < abs_tol
        else:
            abs_tol_met = np.ones(len(p), dtype=bool)
        # Both condition must be met
        any_tol_met = np.all([rel_tol_met, abs_tol_met], axis=0)

        # print('Checking convergence')
        # print(p)
        # print(p_p)
        # print((p-p_p)/p)
        # print(rel_tol_met)
        # print(np.abs(p-p_p))
        # print(abs_tol_met)
        # print(any_tol_met)

        return np.all(any_tol_met)

    while not check_tol(p, p_p, rel_tol, abs_tol):
        if debug:
            print(f'Iteration {iterations+1}:')
            # print(np.abs(1-np.array([xc, yc, dx, dy])/np.array([xc_p, yc_p, dx_p, dy_p])))
        # Store values of previous iteration
        p_p = p.copy()
        # Crop image
        if np.any(np.isnan(p)):
            print('Error: Unable to determine beam parameters using moments, stopping')
            converged = False
            break
        x1lim = int(round(p[0]-p[2]/2*self_conf_width, 0))
        x2lim = int(round(p[0]+p[2]/2*self_conf_width, 0))
        y1lim = int(round(p[1]-p[3]/2*self_conf_width, 0))
        y2lim = int(round(p[1]+p[3]/2*self_conf_width, 0))
        if (x1lim < 0) or (x2lim > image_bkg_corr.shape[1]) \
                or (y1lim < 0) or (y2lim > image_bkg_corr.shape[0]):
            msg = (
                'Integration area'
                +f' (x = [{x1lim}, {x2lim}) px, y = [{y2lim}, {y2lim}) px)'
                +' for D4σ method exceeds image size'
                +f' ({image_bkg_corr.shape[1]} px x {image_bkg_corr.shape[0]} px)')
            if ignore_crop_error:
                print('Warning: '+msg+', using image size instead')
            if not ignore_crop_error:
                print(
                    'Error: '+msg+', stopping (set `ignore_crop_error=True` to ignore this error)')
                converged = False
                break
        x1lim = 0 if x1lim < 0 else x1lim
        y1lim = 0 if y1lim < 0 else y1lim
        x2lim = image_bkg_corr.shape[1] if x2lim > image_bkg_corr.shape[1] else x2lim
        y2lim = image_bkg_corr.shape[0] if y2lim > image_bkg_corr.shape[0] else y2lim
        image_cropped = image_bkg_corr[y1lim:y2lim, x1lim:x2lim]
        # Determine moments
        xc, yc, dx, dy, _ = determine_iso_beam_parameters(
            image_cropped, fixed_orientation=0.)
        xc += x1lim
        yc += y1lim
        p = np.array([xc, yc, dx, dy])
        iterations += 1
        if debug:
            # print(y1lim, y2lim, x1lim, x2lim)
            print(p)
        if iterations == max_iterations:
            if debug:
                print(f'Error: D4σ method did not converge within {max_iterations} iterations')
            converged = False
            break

    # Find orientation for cropped image
    _, _, _, _, orientation = determine_iso_beam_parameters(image_cropped)

    return *p, orientation, iterations, converged

def get_gaussian_2d(amplitude, xc, yc, dx, dy, orientation=0., offset=0.):
    """
    Returns a 2D Gaussian function `gaussian_2d_rotated(x, y)` with amplitude `amplitude` (float)
    centered at `xc` (float) and `yc` (float) along the x- and y-direction, respectively,
    and with a 4-sigma width of `dx` (float) and `dy` (float) along the first and second principal
    axis, respectively.
    The first principal axis is at an angle `orientation` (float, degrees) from the x-axis towards
    the y-axis. If `orientation` is not set, the principal axis lies along the x-axis.
    An offset `offset` (float) can be added. If `offset` is not set, no offset is added.
    """
    orientation = -orientation
    orientation = np.deg2rad(orientation)
    xcr = xc*np.cos(orientation)-yc*np.sin(orientation)
    ycr = xc*np.sin(orientation)+yc*np.cos(orientation)
    def gaussian_2d_rotated(x, y):
        xr = x*np.cos(orientation)-y*np.sin(orientation)
        yr = x*np.sin(orientation)+y*np.cos(orientation)
        return (
            amplitude
            * np.exp(-8*(xr-xcr)**2/dx**2-8*(yr-ycr)**2/dy**2)
            + offset)
    return gaussian_2d_rotated

def fit_gaussian_2d(image, pstart, fixed_orientation=None):
    """
    Least squares fit 2D Gaussian to image `image`. `scipy.optimize.leastsq` is used internally.

    Parameters
    ----------
    image : ndarray
        2-D array representing the intensity of a laser beam.
    pstart : array_like
        Start parameters for fit. If `fixed_orientation` is None, there are six parameters:
            `pstart = [amplitude, x0, y0, dx, dy, orientation, offset]`.
        If the orientation is fixed by setting a value to `fixed_orientation`, there are five
        parameters:
            `pstart = [amplitude, x0, y0, dx, dy, offset]`.
    fixed_orientation : float or None, optional
        If set to a float value, the orientation of the 2D Gaussian is assumed to be fixed to this
        value. If set to None, the orientation is a free fit parameter. The default is None.

    Returns
    -------
    fit_result : dict
        Outputs of `scipy.optimize.leastsq`, organized in dict.
    fit_dict : dict
        Determined fit parameters (values and uncertainties), organized in dict.
    gaussian_2d : func
        Mapping of `get_gaussian_2d` with or without orientation as free parameter.
    """
    if fixed_orientation is None:
        gaussian_2d = get_gaussian_2d
    else:
        gaussian_2d = lambda *p: get_gaussian_2d(*p[0:5], fixed_orientation, p[5])

    num_points = len(image.flatten())
    num_params = len(pstart)
    # Merit function
    func = lambda p: (
        np.ravel(
            gaussian_2d(*p)(*np.flipud(np.indices(image.shape)))
            -image))
    y_sigma = np.ones(num_points)
    # Fit
    fit_error = False
    try:
        popt, pcov, infodict, mesg, ier = scipy.optimize.leastsq(
            func, pstart, full_output=True)
        # Check returned pcov
        if np.any(np.diag(pcov) < 0):
            raise scipy.optimize.OptimizeWarning(
                'Covariance matrix has negative diagonal elements')
        perr = np.sqrt(np.diag(pcov))
        chi_sq = np.sum(func(popt)**2/y_sigma**2)
        red_chi_sq = chi_sq/(num_points-len(popt)) if (num_points-len(popt)) > 0 else np.nan
    except (
            ValueError, TypeError, FloatingPointError, RuntimeError,
            RuntimeWarning, scipy.optimize.OptimizeWarning) \
            as err:
        error_msg = f'2D Gaussian fit did not converge: {err}'
        print(error_msg)
        fit_error = True
    if fit_error:
        popt = np.zeros(num_params)
        popt[:] = np.nan
        perr = np.zeros(num_params)
        perr [:] = np.nan
        pcov = np.zeros((num_params, num_params))
        pcov[:] = np.nan
        chi_sq = np.nan
        red_chi_sq = np.nan
    fit_result = {
        'Popt': popt, 'Pcov': pcov, 'Perr': perr,
        'ChiSq': chi_sq, 'RedChiSq': red_chi_sq,
        'Error': fit_error, 'IEr': ier,
        }
    fit_dict = {
        'Amplitude_Value': popt[0], 'Amplitude_Sigma': perr[0],
        'x0_Value': popt[1], 'x0_Sigma': perr[1],
        'y0_Value': popt[2], 'y0_Sigma': perr[2],
        'w0x_Value': popt[3]/2, 'w0x_Sigma': perr[3]/2,
        'w0y_Value': popt[4]/2, 'w0y_Sigma': perr[4]/2,
        'ChiSq': chi_sq, 'RedChiSq': red_chi_sq,
        'Error': fit_error,
        }
    if fixed_orientation is None:
        fit_dict = {
            **fit_dict,
            'Orientation_Value': popt[5], 'Orientation_Sigma': perr[5],
            'Offset_Value': popt[6], 'Offset_Sigma': perr[6],
            }
    else:
        fit_dict = {
            **fit_dict,
            'Orientation_Value': fixed_orientation, 'Orientation_Sigma': np.nan,
            'Offset_Value': popt[5], 'Offset_Sigma': perr[5],
            }
    return fit_result, fit_dict, gaussian_2d
