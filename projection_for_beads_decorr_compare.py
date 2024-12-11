# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:21:16 2024

@author: S233755
"""

"""
Created on Wed Dec  4 12:26:00 2024

@author: S233755
"""

import numpy as np
import tifffile as tiff
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from datetime import datetime
import tifffile as tiff

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.fft import fftn, fftshift, ifftn, ifftshift
#from scipy.signal import general_gaussian
import scipy.signal.windows as ss
from PIL import Image, ImageTk, ImageEnhance
from matplotlib import cm, pyplot as plt


global axis
axis = 'z'

lateral_pixel_size = 147
axial_pixel_size = 147

class ImageDecorr:
    pod_size = 30
    pod_order = 8

    def __init__(self, image, pixel_size=1.0, square_crop=True):
        """Creates an ImageDecorr contrainer class

        Parameters
        ----------
        image: 2D np.ndarray
        pixel_size: float, default 1.0 the physical pixel size (in µm)
        square_crop: bool, default True, crop a square shape with odd number of pixels

        """

        if not image.ndim == 2:
            raise ValueError("This class expects a 2D image")

        self.image = apodise(image, self.pod_size, self.pod_order)
        self.pixel_size = pixel_size
        nx, ny = self.image.shape

        if square_crop:
            # odd number of pixels, square image
            n = min(nx, ny)
            n = n - (1 - n % 2)
            self.image = self.image[:n, :n]
            self.size = n**2
            xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        else:

            nx = nx - (1 - nx % 2)
            ny = ny - (1 - nx % 2)
            self.image = self.image[:nx, :ny]
            self.size = nx * ny
            xx, yy = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))

        self.disk = xx**2 + yy**2
        self.mask0 = self.disk < 1.0

        im_fft0 = _fft(self.image)
        im_fft0 /= np.abs(im_fft0)
        im_fft0[~np.isfinite(im_fft0)] = 0

        self.im_fft0 = im_fft0 * self.mask0  # I in original code
        image_bar = (self.image - self.image.mean()) / self.image.std()
        im_fftk = _fft(image_bar) * self.mask0  # Ik
        self.im_invk = _ifft(im_fftk).real  # imr

        self.im_fftr = _masked_fft(self.im_invk, self.mask0, self.size)  # Ir

        self.snr0, self.kc0 = self.maximize_corcoef(self.im_fftr).values()  # A0, res0
        self.max_width = 2 / self.kc0
        self.kc = None
        self.resolution = None

    def corcoef(self, radius, im_fftr, c1=None):
        """Computes the normed correlation coefficient between
        the two FFTS of eq. 1 in Descloux et al.
        """
        mask = self.disk < radius**2
        f_im_fft = (mask * self.im_fft0).ravel()[: self.size // 2]
        if c1 is None:
            c1 = np.linalg.norm(im_fftr)
        c2 = np.linalg.norm(f_im_fft)

        return (im_fftr * f_im_fft.conjugate()).real.sum() / (c1 * c2)

    def maximize_corcoef(self, im_fftr, r_min=0, r_max=1):
        """Finds the cutoff radius corresponding to the maximum of the correlation coefficient for
        image fft im_fftr (noted r_i in the article)

        Returns
        -------
        result : dict
            the key 'snr' is the value of self.corcoef at the maximum
            the key 'kc' corresponds to the argmax of self.corcoef
        """
        # cost function
        def anti_cor(radius):
            c1 = np.linalg.norm(im_fftr)
            cor = self.corcoef(radius, im_fftr, c1=c1)
            return 1 - cor

        res = minimize_scalar(anti_cor, bounds=(r_min, r_max), method="bounded", options={"xatol": 1e-4})

        if not res.success:
            return {"snr": 0.0, "kc": 0.0}

        if (r_max - res.x) / r_max < 1e-3:
            return {"snr": 0.0, "kc": r_max}

        return {"snr": 1 - res.fun, "kc": res.x}

    def all_corcoefs(self, num_rs, r_min=0, r_max=1, num_ws=0):
        """Computes decorrelation data for num_rs radius and num_ws filter widths

        This allows to produce plots similar to those of the imagej plugin
        or e.g. fig 1b

        Parameters
        ----------
        num_rs : int
            the number of mask radius
        r_min, r_max : floats
            min and max of the mask radii
        num_ws : float
            number of Gaussian blur filters

        Returns
        -------
        data : dict of ndarrays


        """

        radii = np.linspace(r_min, r_max, num_rs)
        c1 = np.linalg.norm(self.im_fftr)
        d0 = np.array([self.corcoef(radius, self.im_fftr, c1=c1) for radius in radii])
        if not num_ws:
            return {"radii": radii, "ds": d0}

        ds = [d0]
        snr, kc = self.maximize_corcoef(self.im_fftr, r_min, r_max).values()
        snrs = [snr]
        kcs = [kc]

        widths = np.concatenate(
            [
                [
                    0,
                ],
                np.logspace(-1, np.log10(self.max_width), num_ws),
            ]
        )
        for width in widths[1:]:
            f_im = self.im_invk - gaussian_filter(self.im_invk, width)
            f_im_fft = _masked_fft(f_im, self.mask0, self.size)
            c1 = np.linalg.norm(f_im_fft)
            d = np.array([self.corcoef(radius, f_im_fft, c1=c1) for radius in radii])
            ds.append(d)
            snr, kc = self.maximize_corcoef(f_im_fft, r_min, r_max).values()
            snrs.append(snr)
            kcs.append(kc)

        data = {
            "radius": np.array(radii),
            "d": np.array(ds),
            "snr": np.array(snrs),
            "kc": np.array(kcs),
            "widths": widths,
        }
        return data

    def filtered_decorr(self, width, returm_gm=True):
        """Computes the decorrelation cutoff for a given
        filter widh

        If return_gm is True, returns 1 minus the geometric means,
        to be used as a cost function, else, returns the snr
        and the cutoff.
        """
        f_im = self.im_invk - gaussian_filter(self.im_invk, width)
        f_im_fft = _masked_fft(f_im, self.mask0, self.size)
        res = self.maximize_corcoef(f_im_fft)

        if returm_gm:
            if (1 - res["kc"]) < 1e-1:
                return 1 + width
            return 1 - (res["kc"] * res["snr"]) ** 0.5
        return res

    def compute_resolution(self):
        """Finds the filter width giving the maximum of the geometric
        mean (kc * snr)**0.5 (eq. 2)


        """

        res = minimize_scalar(
            self.filtered_decorr,
            method="bounded",
            bounds=(0.15, self.max_width),
            options={"xatol": 1e-3},
        )
        width = res.x
        max_cor = self.filtered_decorr(width, returm_gm=False)

        self.kc = max_cor["kc"]
        if self.kc:
            self.resolution = 2 * self.pixel_size / self.kc
        else:
            self.resolution = np.inf
        return res, max_cor


class StackImDecorr:
    """Apply the image decorrelation algorithm to a n-dimentional stack

    The estimation is passed on each stack plane, where the plane is
    assumed to be stored in the last two dimensions of the array.

    """

    def __init__(self, stack, pixel_size=1.0, axes="TCZYX", square_crop=True):
        """Creates the container class

        Parameters
        ----------

        stack: np.ndarray
        axes: str, one letter per dimension, default 'TCZYX'

        """
        self.stack = stack
        self.pixel_size = pixel_size
        self.axes = axes

        self.ndim = len(axes)

        self.measures = {
            "SNR": np.zeros(stack.shape[:-2]),
            "resolution": np.zeros(stack.shape[:-2]),
        }

        grid = np.meshgrid(*(np.arange(size) for size in stack.shape[:-2]))
        self.grid = [uu.ravel() for uu in grid]
        for c, uu in zip(self.axes, self.grid):
            self.measures[c] = uu

    def _measure_plane(self, coords):

        plane = self.stack[coords]
        imdecorr = ImageDecorr(np.asarray(plane), self.pixel_size)
        imdecorr.compute_resolution()
        self.measures["SNR"][coords] = imdecorr.snr0
        self.measures["resolution"][coords] = imdecorr.resolution

    def measure(self, parallel=True, n_jobs=8):
        """Performs the resolution estimation on all the stack planes
        If parallel is True, uses joblib to
        """
        if parallel:
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._measure_plane)(coords) for coords in zip(*self.grid)
            )

        else:
            for coords in zip(*self.grid):
                self._measure_plane(coords)

    def to_csv(self, filename):
        """Saves computed data to a csv file"""
        from . import __version__

        header = f"""
        Generated from pyimdecorr
        software_version: {__version__}
        software_source: https://gitlab.in2p3.fr/fbi-data/pyImDecorr

        date: {datetime.now().isoformat()}
        {",".join(list(self.axes[: -2]) + ["SNR", "resolution"])}

        """
        tidy = np.stack([c for c in self.grid] + [self.measures["SNR"].ravel(), self.measures["resolution"].ravel()]).T

        with open(filename, "w") as fh:
            np.savetxt(fh, tidy, header=header, delimiter=",")


def measure(image, metadata, **kwargs):
    """Estimates SNR and resolution of an image based on the Image Resolution Estimation
    algorithm by A. Descloux et al.


    Descloux, A., K. S. Grußmayer, et A. Radenovic. _Parameter-Free Image
    Resolution Estimation Based on Decorrelation Analysis_. Nature Methods
    16, nᵒ 9 (septembre 2019) 918‑24. https://doi.org/10.1038/s41592-019-0515-7.

    Parameters
    ----------
    image : np.ndarray
        the nD image to be evaluated
    metadata : dict
        image metadata (the key physicalSizeX will be use as pixel size)

    Returns
    -------
    measured_data : dict
        the evaluated SNR and resolution

    """
    if image.ndim == 2:
        pixel_size = metadata.get("physicalSizeX", 1.0)
        imdecor = ImageDecorr(image, pixel_size)
        imdecor.compute_resolution()
        return {"SNR": imdecor.snr0, "resolution": imdecor.resolution}

    elif image.ndim > 2:
        pixel_size = metadata.get("physicalSizeX", 1.0)
        axes = metadata.get("DimensionOrder", "ZCYX")
        sid = StackImDecorr(image, axes=axes, pixel_size=pixel_size)
        sid.measure(**kwargs)

        return sid.measures



def _fft(image):
    """shifted fft 2D"""
    return fftshift(fftn(fftshift(image)))


def _ifft(im_fft):
    """shifted ifft 2D"""
    return ifftshift(ifftn(ifftshift(im_fft)))


def _masked_fft(im, mask, size):
    """fft of an image multiplied by the mask"""
    return (mask * _fft(im)).ravel()[: size // 2]


# apodImRect.m
def apodise(image, border, order=8):
    """
    Parameters
    ----------

    image: np.ndarray
    border: int, the size of the boreder in pixels

    Note
    ----
    The image is assumed to be of float datatype, no datatype management
    is performed.

    This is different from the original apodistation method,
    which multiplied the image borders by a quater of a sine.
    """
    # stackoverflow.com/questions/46211487/apodization-mask-for-fast-fourier-transforms-in-python
    nx, ny = image.shape
    # Define a general Gaussian in 2D as outer product of the function with itself
    window = np.outer(
        ss.general_gaussian(nx, order, nx // 2 - border),
        ss.general_gaussian(ny, order, ny // 2 - border),
    )
    ap_image = window * image

    return ap_image


def fft_dist(nx, ny):

    uu2, vv2 = np.meshgrid(np.fft.fftfreq(ny) ** 2, np.fft.fftfreq(nx) ** 2)
    dist = (uu2 + vv2) ** 0.5
    return dist  # / dist.sum()


# Function to load TIFF files using Tkinter
def load_tiff_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_paths = filedialog.askopenfilenames(
        title="Select TIFF files",
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )
    if not file_paths:
        print("No files selected.")
        return None

    stack = [tiff.imread(file) for file in sorted(file_paths)]
    return np.array(stack), file_paths

# Projection functions
def project_along_axis(data, axis):
    if axis == 'y':
        return data.max(axis=1)  # Maximum projection along x-axis
    elif axis == 'x':
        return data.max(axis=2)  # Maximum projection along y-axis
    else:
        return data.max(axis=0)

# Visualize projection
def visualize_projection(projection, axis):
    plt.figure(figsize=(8, 6))
    plt.imshow(projection, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.title(f"Projection along {axis}-axis")
    if axis == 'x':
        plt.xlabel("y")
        plt.ylabel("z")
    elif axis == 'y':
        plt.xlabel("x")
        plt.ylabel("z")
    else:
        plt.xlabel("x")
        plt.ylabel("y")
    plt.show()

def res_snr_analysis(axis):
    projected_images = []
    resolutions = []
    signal_to_noise_s = []
    print("Select TIFF files to load...")
    tiff_stack, file_paths = load_tiff_files()
    if tiff_stack is None:
        print("No files loaded. Exiting.")
    else:
        print(f"Loaded stack shape: {tiff_stack.shape}")
        for i, file_path in enumerate(file_paths):
            # Choose axis for projection
            projection = project_along_axis(tiff_stack[i], axis)
            projected_images.append(projection)
            # Automatically save the projection to the same folder
            folder, original_file_name = os.path.split(file_path)
            base_name, ext = os.path.splitext(original_file_name)
            projection_file_name = f"{base_name}_projection_{axis}{ext}"
            save_path = os.path.join(folder, projection_file_name)
            tiff.imwrite(save_path, projection.astype(np.float32))
            print(f"Projection saved to {save_path}")
            # Visualize the projection
            visualize_projection(projection, axis)
    
    for i in np.arange(len(projected_images)):
        m = measure(projected_images[i], {})
        resolutions.append(m['resolution'])
        signal_to_noise_s.append(m['SNR'])
        
    resolutions = np.array(resolutions)*lateral_pixel_size
    signal_to_noise_s = np.array(signal_to_noise_s)
    ind_ = np.arange(len(resolutions))  


    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ind_, resolutions, label='Resolution')
    plt.title("Resolution")
    plt.xlabel("Time")
    plt.ylabel("Resolution (nm)")

    plt.subplot(3, 1, 2)
    plt.plot(ind_, signal_to_noise_s, label='Resolution')
    plt.title("SNR")
    plt.xlabel("Time")
    plt.ylabel("SNR")

    plt.tight_layout()
    plt.show()
    
    return resolutions, signal_to_noise_s

# Main function
if __name__ == "__main__":
    with_auto_focus_res, with_auto_focus_SNR = res_snr_analysis(axis)    
    without_auto_focus_res, without_auto_focus_SNR = res_snr_analysis(axis)

    ind_ = np.arange(len(with_auto_focus_res))
    
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(ind_, with_auto_focus_res, label='With Autofocus Resolution', color ='r')
    # plt.plot(ind_[0:20], without_auto_focus_res[0:20], label='With Autofocus Resolution', color ='k')
    # plt.title("Resolution")
    # plt.xlabel("Time")
    # plt.ylabel("Resolution (nm)")

    # plt.subplot(3, 1, 2)
    # plt.plot(ind_, with_auto_focus_SNR, label='With Autofocus Resolution', color ='r')
    # plt.plot(ind_, without_auto_focus_SNR, label='With Autofocus Resolution', color ='k')
    # plt.title("SNR")
    # plt.xlabel("Time")
    # plt.ylabel("SNR")

    # plt.tight_layout()
    # plt.show()
        
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ind_, with_auto_focus_res, label='With Autofocus Resolution', color ='r')
    plt.plot(ind_, without_auto_focus_res, label='With Autofocus Resolution', color ='k')
    plt.title("Resolution")
    plt.xlabel("Time")
    plt.ylabel("Resolution (nm)")
    
    plt.subplot(3, 1, 2)
    plt.plot(ind_, with_auto_focus_SNR, label='With Autofocus Resolution', color ='r')
    plt.plot(ind_, without_auto_focus_SNR, label='With Autofocus Resolution', color ='k')
    plt.title("SNR")
    plt.xlabel("Time")
    plt.ylabel("SNR")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure()

    plt.plot(ind_, with_auto_focus_res, label='With Autofocus Resolution', color ='r')
    plt.plot(ind_, without_auto_focus_res, label='With Autofocus Resolution', color ='k')
    plt.title("Resolution")
    plt.xlabel("Time")
    plt.ylabel("Resolution (nm)")
    

    plt.tight_layout()
    plt.show()
    
    
    