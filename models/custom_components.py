import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import torchvision
from functools import cmp_to_key
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from models.yolo import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync
import torch.nn as nn
from utils.loss import ComputeLoss

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Keypoint():
    def __init__(self, pt=(None, None), size=None, angle=None, resp=None, oct=None):
        self.pt = pt
        self.size = size
        self.angle = angle
        self.response = resp
        self.octave = oct
        return


class HessianKernelGood(nn.Module):
    """
    """

    def __init__(self, sigma=0.707107, kernel_size=21, scale=2, max_keys=2000):
        super().__init__()

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.float_tolerance = 1e-7
        self.image_scale = scale
        self.key_descriptors = {}
        self.max_num_keys = max_keys
        scale_factor = np.sqrt(2)
        sigma = (sigma, sigma*scale_factor, sigma*(scale_factor*2), sigma*(scale_factor*3), sigma*(scale_factor*4))

        whole_kernel = torch.zeros((5, kernel_size, kernel_size))
        kernel_size = [kernel_size] * 2
        self.sigma = [sigma] * 2
        self.gauss_scale_dict = {}
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for idx, sigma in enumerate(self.sigma[0]):
            self.gauss_scale_dict[idx] = sigma
            sigma = [sigma] * 2
            kernel = 1
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            # Make sure sum of values in gaussian kernel equals 1.
            gauss_kernel = kernel / torch.sum(kernel)
            whole_kernel[idx] = gauss_kernel

        # Make sure sum of values in gaussian kernel equals 1.
        # gauss_kernel = kernel / torch.sum(kernel)

        # Make 3-D to apply to all three channels at once
        gauss_kernel = whole_kernel[0].view(1, 1, *kernel_size).float()
        gauss_kernel2 = whole_kernel[1].view(1, 1, *kernel_size).float()
        gauss_kernel3 = whole_kernel[2].view(1, 1, *kernel_size).float()
        gauss_kernel4 = whole_kernel[3].view(1, 1, *kernel_size).float()
        gauss_kernel5 = whole_kernel[4].view(1, 1, *kernel_size).float()

        # gauss_kernel = gauss_kernel.repeat(3, *[1] * (gauss_kernel.dim() - 1)).float()

        """plt.imshow(gauss_kernel[0][0], cmap='gray')
        plt.title('filter 0')
        plt.show()

        plt.imshow(gauss_kernel[1][0], cmap='gray')
        plt.title('filter 1')
        plt.show()

        plt.imshow(gauss_kernel[2][0], cmap='gray')
        plt.title('filter 2')
        plt.show()"""

        # Det(Hassian) kernels below
        det_x_kernel = torch.tensor(np.array([[0, 0, 0],
                                              [1, -2, 1],
                                              [0, 0, 0]])).float()
        det_x_kernel = det_x_kernel.view(1, 1, *det_x_kernel.size()).float()
        det_y_kernel = torch.tensor(np.array([[0, 1, 0],
                                              [0, -2, 0],
                                              [0, 1, 0]])).float()
        det_y_kernel = det_y_kernel.view(1, 1, *det_y_kernel.size()).float()

        det_xy_kernel = 0.5 * torch.tensor(np.array([[1, -1, 0],
                                                     [-1, 2, -1],
                                                     [0, -1, 1]])).float()
        det_xy_kernel = det_xy_kernel.view(1, 1, *det_xy_kernel.size()).float()

        # Laplacian Kernel Below
        lap_kernel = torch.tensor(np.array([[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]])).float()
        lap_kernel = lap_kernel * self.sigma[0][0]  # Scale normalize laplacian kernel
        lap_kernel = lap_kernel.view(1, 1, *lap_kernel.size()).float()
        # Shape [out channels, in channels/groups, kernel size[0], kernel size[1]]

        # Output shape of 3 x img size - 3 gaussian scales for computing maxima/minima later on
        self.gauss_blur1 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur1.weight = nn.Parameter(gauss_kernel, requires_grad=False)
        # self.gauss_blur1 = self.gauss_blur1.to(dev)

        self.gauss_blur2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur2.weight = nn.Parameter(gauss_kernel2, requires_grad=False)
        # self.gauss_blur2 = self.gauss_blur2.to(dev)

        self.gauss_blur3 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur3.weight = nn.Parameter(gauss_kernel3, requires_grad=False)
        # self.gauss_blur3 = self.gauss_blur3.to(dev)

        self.gauss_blur4 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur4.weight = nn.Parameter(gauss_kernel4, requires_grad=False)
        # self.gauss_blur4 = self.gauss_blur4.to(dev)

        self.gauss_blur5 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur5.weight = nn.Parameter(gauss_kernel5, requires_grad=False)
        # self.gauss_blur5 = self.gauss_blur5.to(dev)

        # Compute Laplacian/
        # TODO Do we want to pad when it's an edge detector?
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, stride=1,  bias=False, groups=1, padding_mode='reflect',
                                   padding=1)
        self.laplacian.weight = nn.Parameter(lap_kernel, requires_grad=False)

        # Determinent of Hessian Convolutions
        self.det_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                               padding=1)
        self.det_x.weight = nn.Parameter(det_x_kernel, requires_grad=False)
        # self.det_x = self.det_x.to(dev)

        self.det_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                               padding=1)
        self.det_y.weight = nn.Parameter(det_y_kernel, requires_grad=False)
        # self.det_y = self.det_y.to(dev)

        self.det_xy = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                padding=1)
        self.det_xy.weight = nn.Parameter(det_xy_kernel, requires_grad=False)
        # self.det_xy = self.det_xy.to(dev)

        # For Maxima/Minima extraction
        self.max_3d = nn.MaxPool3d(kernel_size=3, stride=(3, 1, 1), padding=(1, 1, 1), return_indices=True)

        self.init_localization_kernels()

    def forward(self, images):
        with torch.no_grad():
            self.img_dev = images.device

            to_gray = torchvision.transforms.Grayscale()
            images = to_gray(images)

            # TODO Initially blur image?
            sigma_diff = np.sqrt(max((1.6 ** 2) - ((2 * 0.5) ** 2), 0.01))
            init_blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=sigma_diff)
            blur_img = init_blur(images)

            num_octaves = int(round(np.log(min(blur_img.size()[2:4])) / np.log(2) - 1))

            # TODO Create scale-space with octaves?
            kernels = self.generate_gaussian_scales(1.6, 3)

            gauss_imgs = self.generate_gaussian_imgs(blur_img, num_octaves, kernels)

            # images = images * 255
            gauss_images = self.gauss_blur1(images)
            gauss_images2 = self.gauss_blur2(images)
            gauss_images3 = self.gauss_blur3(images)
            gauss_images4 = self.gauss_blur4(images)
            gauss_images5 = self.gauss_blur5(images)
            # gauss_comb = torch.stack((gauss_images, gauss_images2, gauss_images3, gauss_images4, gauss_images5), dim=2)
            gauss_comb = torch.cat((gauss_images, gauss_images2, gauss_images3, gauss_images4, gauss_images5), dim=0)
            del gauss_images
            del gauss_images2
            del gauss_images3
            del gauss_images4
            del gauss_images5
            # image2 = gauss_images.numpy()[0][0]
            # image22 = gauss_images.numpy()[0][1]
            # plt.imshow(image2.reshape(image2.shape[0], image2.shape[1]), cmap='gray')
            # plt.title('filter 0')
            # plt.show()
            # plt.imshow(image22.reshape(image22.shape[0], image22.shape[1]), cmap='gray')
            # plt.title('filter 1')
            # plt.show()

            # image3 = det_hess.numpy()[0][0]
            # plt.imshow(image3.reshape(image3.shape[0], image3.shape[1]), cmap='gray')
            # plt.title('filter 0')
            # plt.show()

            # Extract Maxima/Minima (And Saddles for Det of Hessian?)
            # These max min are for a specific scale (middle of self.sigma)
            # Returns with shape of input, with all values 0 except for maxima or minima

            # self.orientated_keys = self.find_keys(gauss_comb)  # localized keypoints (no orientations)

            orientated_keys = self.find_keys(gauss_imgs)
            filt_keys = self.remove_duplicates(orientated_keys)
            conv_keys = self.convert_keys_to_img_size(filt_keys)

            described_keys = self.generate_descriptors(conv_keys, gauss_imgs)

        return described_keys

    def remove_duplicates(self, keys):
        if len(keys) < 2:
            return keys

        keys.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keys = [keys[0]]

        for next_key in keys[1:]:
            last_unique_key = unique_keys[-1]
            if last_unique_key.pt[0] != next_key.pt[0] or \
                last_unique_key.pt[1] != next_key.pt[1] or \
                last_unique_key.size != next_key.size or \
                last_unique_key.angle != next_key.angle:
                unique_keys.append(next_key)
        return unique_keys

    def compare_keypoints(self, keypoint1, keypoint2):
        """Return True if keypoint1 is less than keypoint2
        """
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        return keypoint2.octave - keypoint1.octave

    def convert_keys_to_img_size(self, keys):
        convert_keys = []
        for key in keys:
            key.pt = tuple(0.5 * torch.tensor(key.pt))
            key.size *= 0.5
            key.octave = (key.octave & ~255) | ((key.octave - 1) & 255)
            convert_keys.append(key)
        return convert_keys

    def generate_descriptors(self, keypoints, gauss_imgs, window_width=4, num_bins=8, scale_mult=3, desc_max_value=0.2):
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gauss_img = gauss_imgs[octave + 1][int(layer)][0]
            num_rows, num_cols = gauss_img.size()
            point = (scale * torch.tensor(keypoint.pt[0], dtype=torch.int),
                     scale * torch.tensor(keypoint.pt[1], dtype=torch.int))
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = torch.cos(torch.rad2deg(angle))
            sin_angle = torch.sin(torch.rad2deg(angle))
            weight_mult = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            mag_list = []
            orient_bin_list = []
            hist_tensor = torch.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_mult * 0.5 * scale * keypoint.size
            half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            half_width = int(np.minimum(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = point[1] + row
                        window_col = point[0] + col
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gauss_img[window_row, window_col + 1] - gauss_img[window_row, window_col - 1]
                            dy = gauss_img[window_row - 1, window_col] - gauss_img[window_row + 1, window_col]
                            gradient_magnitude = torch.sqrt(dx * dx + dy * dy)
                            gradient_orientation = torch.rad2deg(torch.atan2(dy, dx)) % 360
                            weight = torch.exp(weight_mult * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            mag_list.append(weight * gradient_magnitude)
                            orient_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, mag_list,
                                                                    orient_bin_list):
                # Smoothing via trilinear interpolation
                # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(
                    int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                hist_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                hist_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                hist_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                hist_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                hist_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                hist_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                hist_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                hist_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = hist_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = torch.norm(descriptor_vector) * desc_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(torch.norm(descriptor_vector), self.float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = torch.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            # whole_tensor = torch.cat((torch.tensor((*keypoint.pt, keypoint.angle, scale)), descriptor_vector))
            whole_tensor = [keypoint.pt[0].item(), keypoint.pt[1].item(), keypoint.angle.item(), scale, *descriptor_vector.tolist()]
            descriptors.append(whole_tensor)

        return descriptors
        # return torch.stack(descriptors)

    def unpack_octave(self, key):
        octave = key.octave & 255
        layer = (key.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, scale, layer

    def generate_gaussian_scales(self, sigma, num_intervals):
        num_imgs_per_oct = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = torch.zeros(num_imgs_per_oct)
        gaussian_kernels[0] = sigma

        for image_idx in range(1, num_imgs_per_oct):
            sigma_prev = (k ** (image_idx - 1)) * sigma
            sigma_total = k * sigma_prev
            gaussian_kernels[image_idx] = np.sqrt(sigma_total ** 2 - sigma_prev ** 2)

        return gaussian_kernels

    def generate_gaussian_imgs(self, img, num_octaves, kernels):
        gaussian_imgs = []

        for octave_idx in range(num_octaves):
            gauss_img_in_oct = []
            gauss_img_in_oct.append(img)
            for gauss_kernel in kernels[1:]:
                blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=np.float32(gauss_kernel))
                img = blur(img)
                gauss_img_in_oct.append(img)
            gaussian_imgs.append(torch.cat(gauss_img_in_oct))
            octave_base = gauss_img_in_oct[-3]
            sizer = torchvision.transforms.Resize((int(octave_base.size()[3] / 2), int(octave_base.size()[2] / 2)),
                                                  interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            img = sizer(octave_base)

        return gaussian_imgs

    def find_keys(self, gauss_images, border_width=3):
        keypoints = []

        for octave_idx, imgs_in_oct in enumerate(gauss_images):
            det_x = self.det_x(imgs_in_oct)
            dey_y = self.det_y(imgs_in_oct)
            det_xy = self.det_xy(imgs_in_oct)
            det_hess = (det_x * dey_y) - (det_xy * det_xy)
            # det_hess = self.laplacian(gauss_images)
            del det_x
            del dey_y
            del det_xy

            for image_idx, (first_img, second_img, third_img) in enumerate(zip(det_hess, det_hess[1:], det_hess[2:])):

                for i in range(border_width, first_img.size()[1] - border_width):
                    for j in range(border_width, first_img.size()[2] - border_width):
                        if self.is_pixel_extrema(first_img[0, i-1:i+2, j-1:j+2], second_img[0, i-1:i+2, j-1:j+2],
                                                 third_img[0, i-1:i+2, j-1:j+2], threshold=0.04):
                            localized_key = self.localize_point(i, j, image_idx+1, det_hess, border_width,
                                                                octave_idx=octave_idx, sigma=1.6, num_intervals=3)

                            if localized_key is not None:
                                keypoint, localized_img_idx = localized_key
                                keys_with_orientation = self.compute_orientation(keypoint, octave_index=octave_idx, gaussian_image=gauss_images[octave_idx][localized_img_idx])
                                for key in keys_with_orientation:
                                    keypoints.append(key)

            return keypoints

    def is_pixel_extrema(self, first_sub, second_sub, third_sub, threshold=0.3):
        center_pix_value = second_sub[1, 1]
        if abs(center_pix_value) > threshold:
            if center_pix_value > 0:
                return torch.all(center_pix_value >= first_sub) and torch.all(center_pix_value >= third_sub) and \
                       torch.all(center_pix_value >= second_sub[0, :]) and \
                       torch.all(center_pix_value >= second_sub[2, :]) and \
                       center_pix_value >= second_sub[1, 0] and center_pix_value >= second_sub[1, 2]
            elif center_pix_value < 0:
                return torch.all(center_pix_value <= first_sub) and torch.all(center_pix_value <= third_sub) and \
                       torch.all(center_pix_value <= second_sub[0, :]) and \
                       torch.all(center_pix_value <= second_sub[2, :]) and \
                       center_pix_value <= second_sub[1, 0] and center_pix_value <= second_sub[1, 2]
        return False

    def localize_point(self, x, y, image_idx, images, image_border, num_intervals, octave_idx, sigma, num_attempts=5, contrast_thresh=10,
                       eigenvalue_ratio=10):
        extremum_outside_image = False
        image_shape = images[0].size()

        for attempt_idx in range(num_attempts):
            first_img, second_img, third_img = images[image_idx-1:image_idx+2]
            cube = torch.stack([first_img[0, x - 1:x + 2, y - 1:y + 2],
                                second_img[0, x - 1:x + 2, y - 1:y + 2],
                                third_img[0, x - 1: x + 2, y - 1:y + 2]], dim=0)
            cube_grad = self.calc_gradient(cube)
            cube_hess = self.calc_hess(cube)

            extremum_update = -torch.linalg.lstsq(cube_hess.float(), cube_grad.float(), rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(
                    extremum_update[2]) < 0.5:
                break  # Break convergance attempt loop

            x += int(torch.round(extremum_update[0]))
            y += int(torch.round(extremum_update[1]))
            image_idx += int(torch.round(extremum_update[2]))

            if x < image_border or x >= image_shape[1] - image_border or \
                    y < image_border or y >= image_shape[2] - image_border or \
                    image_idx < 1 or image_idx > num_intervals:  # TODO Confirm this last line
                extremum_outside_image = True
                break

        if extremum_outside_image:
            return None
        if attempt_idx >= num_attempts - 1:
            return None
        value_at_updated_extrema = cube[1, 1, 1] + 0.5 * torch.dot(cube_grad, extremum_update)
        if abs(value_at_updated_extrema) * num_intervals >= contrast_thresh:
            xy_hessian = cube_hess[:2, :2]
            xy_hessian_trace = torch.trace(xy_hessian)
            xy_hessian_det = torch.det(xy_hessian)
            if xy_hessian_det > 0 and \
                    eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Passes contrast check - therefore good keypoint
                keypoint = Keypoint()
                keypoint.pt = ((x + extremum_update[0]) * (2 ** octave_idx), (y + extremum_update[1]) * (2 ** octave_idx))
                keypoint.octave = octave_idx + image_idx * (2 ** 8) + int(torch.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_idx + extremum_update[2]) / float(num_intervals))) * (2 ** (octave_idx + 1))
                keypoint.response = abs(value_at_updated_extrema)
                return keypoint, image_idx

        return None

    def compute_orientation(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8,
                            scale_factor=1.5, float_tolerance=0):
        keypoints_with_orientation = []
        image_shape = gaussian_image.size()[1:]
        gauss_image = gaussian_image[0]

        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(torch.round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_hist = torch.zeros(num_bins)
        smooth_hist = torch.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(torch.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(torch.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gauss_image[region_y, region_x + 1] - gauss_image[region_y, region_x - 1]
                        dy = gauss_image[region_y - 1, region_x] = gauss_image[region_y + 1, region_x]
                        gradient_mag = torch.sqrt(dx * dx + dy * dy)
                        gradient_orientation = torch.rad2deg(torch.atan2(dy, dx))
                        weight = torch.exp(weight_factor * (i ** 2 + j ** 2))
                        histo_idx = int(torch.round(gradient_orientation * num_bins / 360.))
                        raw_hist[histo_idx % num_bins] += weight * gradient_mag

        for n in range(num_bins):
            smooth_hist[n] = (6 * raw_hist[n] + 4 * (raw_hist[n - 1] + raw_hist[(n + 1) % num_bins]) + raw_hist[n - 2] + raw_hist[(n + 2) % num_bins]) / 16
            orientation_max = max(smooth_hist)
            orientation_peaks = torch.where(torch.logical_and(smooth_hist > torch.roll(smooth_hist, 1), smooth_hist > torch.roll(smooth_hist, -1)))[0]
            for peak_idx in orientation_peaks:
                peak_value = smooth_hist[peak_idx]
                if peak_value >= peak_ratio * orientation_max:
                    # Quadratic peak interpolation
                    left_value = smooth_hist[(peak_idx - 1) % num_bins]
                    right_value = smooth_hist[(peak_idx + 1) % num_bins]
                    interp_peak_idx = (peak_idx + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value)) % num_bins
                    orientation = 360. - interp_peak_idx * 360. / num_bins
                    if abs(orientation - 360.) < float_tolerance:
                        orientation = 0
                    new_key = Keypoint(pt=keypoint.pt, size=keypoint.size, angle=orientation,
                                       resp=keypoint.response, oct=keypoint.octave)
                    keypoints_with_orientation.append(new_key)

        return keypoints_with_orientation

    def localize(self, max_min_points, images):
        localized_dict = {}

        # sx_all = self.gradients_large(self.gradientx_large(images))
        # sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
        # sy_all = self.gradients_large(self.gradienty_large(images))
        # sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
        ss_all = self.gradients_large(self.gradients_large(images))
        # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])

        # Max min points has shape (num scale, batch, num_non_zero)
        for scale in max_min_points.keys():

            if scale == 0:
                continue
            elif scale == 4:
                continue

            batched_dicts = {}  # Batch: Keys
            local_points = []
            all_points = []
            # Need to calculate gradient and hessian and solve resulting 3x3 system to localize points
            # Then check if offset is less than 0.5

            gradx = self.gradientx(images[:, :, scale])  # scale instead of 1
            grady = self.gradienty(images[:, :, scale])
            grads = self.gradients(images)
            grads = grads.view(grads.size()[0], 1, *grads.size()[-2:])  # Reduce dimensions by 1

            # Testing
            xx = self.gradientx(gradx)
            xy = self.gradientx(grady)
            xs = self.gradientx(grads)
            yx = self.gradienty(gradx)
            yy = self.gradienty(grady)
            ys = self.gradienty(grads)
            sx = self.gradients(self.gradientx_large(images))
            sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
            sy = self.gradients(self.gradienty_large(images))
            sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
            # ss = self.gradients_large(self.gradients_large(images))
            # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])
            # sx = sx_all[:, :, scale]
            # sy = sy_all[:, :, scale]
            ss = ss_all[:, :, scale]

            # TODO Faster maybe?
            first_img, second_img, third_img = images[:, :, scale-1:scale+2][0, 0]

            # xx = self.xx_conv(images[:, :, scale])
            # xy = self.xy_conv(images[:, :, scale])
            # xs = self.xs_conv(images)
            # xs = xs.view(xs.size()[0], 1, *xs.size()[-2:])
            # yx = self.yx_conv(images[:, :, scale])
            # yy = self.yy_conv(images[:, :, scale])
            # ys = self.ys_conv(images)
            # ys = ys.view(ys.size()[0], 1, *ys.size()[-2:])
            # sx = self.sx_conv(images)
            # sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
            # sy = self.sy_conv(images)
            # sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
            # ss = self.ss_conv(images)
            # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])

            # TODO Add image batching
            for batch, image in enumerate(images):
                for point in max_min_points[scale][batch]:
                    j = 0
                    i = 0
                    x, y = point[-2:]

                    if x == 0:
                        continue
                    elif y == 0:
                        continue

                    for _ in range(5):
                        # Compute gradient of center pixel from each 3x3x3 window

                        # Catches if i or j is massive and causes overflow error
                        if i > image[0, scale].size()[1]:
                            i = 10
                        if j > image[0, scale].size()[0]:
                            j = 10

                        if x+j >= image[0, scale].size()[0]:
                            j = (image[0, scale].size()[0] - x) - 1
                        if y+i >= image[0, scale].size()[1]:
                            i = (image[0, scale].size()[1] - y) - 1
                        if x+j < 0:
                            j = -int(x)
                        if y+i < 0:
                            i = -int(y)

                        local_grad = torch.stack((gradx[batch, :, x+j, y+i], grady[batch, :, x+j, y+i],
                                                  grads[batch, :, x+j, y+i]))
                        hess_x = torch.stack((xx[batch, :, x+j, y+i], xy[batch, :, x+j, y+i], xs[batch, :, x+j, y+i]))
                        hess_y = torch.stack((yx[batch, :, x+j, y+i], yy[batch, :, x+j, y+i], ys[batch, :, x+j, y+i]))
                        hess_s = torch.stack((sx[batch, :, x+j, y+i], sy[batch, :, x+j, y+i], ss[batch, :, x+j, y+i]))
                        local_hess = torch.stack((hess_x, hess_y, hess_s), dim=0).squeeze()  # TODO Confirm dimension

                        # TODO Confirm works for batched images
                        extremum_update = -torch.linalg.lstsq(local_hess.float(), local_grad.float(), rcond=None)[0]

                        # Check change is less than 0.5 in any direction
                        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                            # single_descriptor = torch.zeros(129)
                            # Threshold
                            single_point = image[:, scale, x, y] + 0.5 * torch.matmul(local_grad.squeeze(), extremum_update.view(3))
                            if single_point > 0.03:
                                edge_hess = torch.stack((torch.stack((xx[batch, 0, x, y], xy[batch, 0, x, y])),
                                                         torch.stack((yx[batch, 0, x, y], yy[batch, 0, x, y]))))
                                eigv, _ = torch.linalg.eig(edge_hess.float())
                                eigv = torch.real(eigv)
                                ratio = eigv[0] / eigv[1]

                                if ratio < 10:
                                    # This is the actual localized maxima/minima
                                    local_points.append(torch.tensor((batch, scale, point[1] + j, point[2] + i)))

                            else:
                                pass
                            break

                        # TODO Need better way to handle this
                        if torch.isinf(extremum_update[0]):
                            extremum_update[0] = j - 1
                            break
                        if torch.isinf(extremum_update[1]):
                            extremum_update[1] = i - 1
                            break

                        # TODO Need better way to handle this
                        if torch.isnan(extremum_update[0]):
                            extremum_update[0] = 1
                            break
                        if torch.isnan(extremum_update[1]):
                            extremum_update[1] = 1
                            break

                        j += int(torch.round(extremum_update[0]))
                        i += int(torch.round(extremum_update[1]))
                        # TODO This is also important
                        # image_idx += int(round(extremum_update[2]))

                if len(local_points) > 0:
                    batched_dicts[batch] = torch.stack(local_points, dim=0)
                else:
                    batched_dicts[batch] = torch.tensor([])
            # local_points = torch.stack(local_points, dim=0)
            localized_dict[scale] = batched_dicts

        return localized_dict  # Scale: {batch: keys}

    def calc_gradient(self, points):
        # Gradient at center pixel [1,1,1] of 3x3x3 array using central difference formula
        dx = 0.5 * (points[1, 1, 2] - points[1, 1, 0])
        dy = 0.5 * (points[1, 2, 1] - points[1, 0, 1])
        ds = 0.5 * (points[2, 1, 1] - points[0, 1, 1])
        gradient = torch.stack([dx, dy, ds])

        return gradient

    def calc_hess(self, points):
        # Hessian at center pixel [1,1,1] of 3x3x3 array using central difference formula
        center_pix_value = points[1, 1, 1]
        dxx = points[1, 1, 2] - 2 * center_pix_value + points[1, 1, 0]
        dyy = points[1, 2, 1] - 2 * center_pix_value + points[1, 0, 1]
        dss = points[2, 1, 1] - 2 * center_pix_value + points[0, 1, 1]
        dxy = 0.25 * (points[1, 2, 2] - points[1, 2, 0] - points[1, 0, 2] + points[1, 0, 0])
        dxs = 0.25 * (points[2, 1, 2] - points[2, 1, 0] - points[0, 1, 2] + points[0, 1, 0])
        dys = 0.25 * (points[2, 2, 1] - points[2, 0, 1] - points[0, 2, 1] + points[0, 0, 1])
        x = torch.stack([dxx, dxy, dxs])
        y = torch.stack([dxy, dyy, dys])
        s = torch.stack([dxs, dys, dss])
        hessian = torch.stack([x, y, s], dim=1)

        return hessian

    def orientate(self, keys, images):

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        scale_descriptors = {}
        for scale in keys.keys():

            scale_sigma = self.gauss_scale_dict[scale]
            kernel_size = [16, 16]
            meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
            kernel_sigma = [scale_sigma * 1.5] * 2
            kernel = 1
            for size, std, mgrid in zip(kernel_size, kernel_sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            # Make sure sum of values in gaussian kernel equals 1.
            gauss_kernel = (kernel / torch.sum(kernel)).to(dev)

            gradx = self.gradientx(images[:, :, scale]) + 1e-6
            grady = self.gradienty(images[:, :, scale]) + 1e-6

            mag = torch.sqrt(torch.square(gradx) + torch.square(grady))
            orientation = torch.atan(grady / gradx)

            for batch in keys[scale]:
                batch_descriptors = []
                for point in keys[scale][batch]:
                    x, y = point[-2:]
                    if x < 9:
                        continue
                    elif y < 9:
                        continue
                    elif x > mag.size()[2] - 16:
                        continue
                    elif y > mag.size()[3] - 16:
                        continue
                    ind_mag = mag[:, :, x-8:x+8, y-8:y+8]
                    ind_ori = orientation[:, :, x-8:x+8, y-8:y+8]

                    # Account for negative values
                    ind_ori_mask = ind_ori < 0.
                    ind_ori[ind_ori_mask] = ind_ori[ind_ori_mask] + (2*np.pi)

                    # Create Histogram
                    orientation_prime = ind_ori / 10  # TODO Check this?
                    # orientation_prime = ind_ori
                    mag_weight = ind_mag * gauss_kernel

                    histo, bin_edges = torch.histogram(orientation_prime.to('cpu').float(), bins=36,
                                                       range=(0., 2*np.pi), weight=mag_weight.to('cpu'), density=False)
                    histo = histo.to(dev)

                    max_value = histo.max()
                    max_mask = histo >= max_value * 0.8

                    for idx in max_mask.nonzero():
                        # peak_value = histo.max()
                        single_descriptor = torch.zeros(132)
                        single_descriptor[0] = x
                        single_descriptor[1] = y
                        single_descriptor[2] = scale
                        single_descriptor[3] = histo[idx]

                        # Calculate Descriptor
                        # Compute histograms for 4x4 regions of the 16x16 window
                        desc = torch.zeros((4, 4, 8))
                        for x_count, x_idx in enumerate(range(0, ind_mag.size()[2], 4)):
                            for y_count, y_idx in enumerate(range(0, ind_mag.size()[3], 4)):
                                indiv_mag = mag[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                                indiv_ori = ind_ori[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                                indiv_mag_weight = indiv_mag * gauss_kernel[6:10, 6:10]

                                # TODO Need to calc Orientation prime?
                                # TODO Better off computing on my own using GPU?
                                eight_histo, eight_bin_edges = torch.histogram(indiv_ori.to('cpu').float(), bins=8,
                                                                               range=(0., 2 * np.pi),
                                                                               weight=indiv_mag_weight.to('cpu'), density=False)
                                eight_histo = eight_histo.to(dev)
                                desc[x_count, y_count, :] = eight_histo

                        # Flatten 4x4 array of histos
                        flat_desc = torch.flatten(desc)
                        single_descriptor[4:] = flat_desc

                    # TODO Confirm below
                    batch_descriptors.append(single_descriptor)
                if len(batch_descriptors) == 0:
                    batch_descriptors.append(torch.zeros(132))
                scale_descriptors[batch] = torch.stack(batch_descriptors).to(self.img_dev)

            self.key_descriptors[scale] = scale_descriptors

        return

    def single_orientate(self, scale, batch, x, y, images):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        scale_descriptors = {}

        # For a single scale below (scale.keys())
        scale_sigma = self.gauss_scale_dict[scale]
        kernel_size = [16, 16]
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernel_sigma = [scale_sigma * 1.5] * 2
        kernel = 1
        for size, std, mgrid in zip(kernel_size, kernel_sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        gauss_kernel = (kernel / torch.sum(kernel)).to(dev)

        gradx = self.gradientx(images[:, :, scale]) + 1e-6
        grady = self.gradienty(images[:, :, scale]) + 1e-6
        mag = torch.sqrt(torch.square(gradx) + torch.square(grady))
        orientation = torch.atan(grady / gradx)

        # For a single batch key below
        batch_descriptors = []
        if x < 9:
            return
        elif y < 9:
            return
        elif x > mag.size()[2] - 16:
            return
        elif y > mag.size()[3] - 16:
            return
        ind_mag = mag[:, :, x - 8:x + 8, y - 8:y + 8]
        ind_ori = orientation[:, :, x - 8:x + 8, y - 8:y + 8]

        # Account for negative values
        ind_ori_mask = ind_ori < 0.
        ind_ori[ind_ori_mask] = ind_ori[ind_ori_mask] + (2 * np.pi)

        # Create Histogram
        orientation_prime = ind_ori / 10  # TODO Check this?
        # orientation_prime = ind_ori
        mag_weight = ind_mag * gauss_kernel

        histo, bin_edges = torch.histogram(orientation_prime.to('cpu').float(), bins=36,
                                           range=(0., 2 * np.pi), weight=mag_weight.to('cpu'), density=False)
        histo = histo.to(dev)

        max_value = histo.max()
        max_mask = histo >= max_value * 0.8

        for idx in max_mask.nonzero():
            # peak_value = histo.max()
            single_descriptor = torch.zeros(132)
            single_descriptor[0] = x
            single_descriptor[1] = y
            single_descriptor[2] = scale
            single_descriptor[3] = histo[idx]

            # Calculate Descriptor
            # Compute histograms for 4x4 regions of the 16x16 window
            desc = torch.zeros((4, 4, 8))
            for x_count, x_idx in enumerate(range(0, ind_mag.size()[2], 4)):
                for y_count, y_idx in enumerate(range(0, ind_mag.size()[3], 4)):
                    indiv_mag = mag[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                    indiv_ori = ind_ori[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                    indiv_mag_weight = indiv_mag * gauss_kernel[6:10, 6:10]

                    # TODO Need to calc Orientation prime?
                    # TODO Better off computing on my own using GPU?
                    eight_histo, eight_bin_edges = torch.histogram(indiv_ori.to('cpu').float(), bins=8,
                                                                   range=(0., 2 * np.pi),
                                                                   weight=indiv_mag_weight.to('cpu'), density=False)
                    eight_histo = eight_histo.to(dev)
                    desc[x_count, y_count, :] = eight_histo

            # Flatten 4x4 array of histos
            flat_desc = torch.flatten(desc)
            single_descriptor[4:] = flat_desc

        self.key_descriptors[scale][batch] = scale_descriptors

        return

    def init_localization_kernels(self):

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Gradient for point localization
        zero_kernel = torch.tensor(np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]])).float()
        grad_kernel_x = torch.tensor(np.array([[0, 0, 0],
                                               [-1, 0, 1],
                                               [0, 0, 0]])).float() * 0.5
        grad_kernel_y = torch.tensor(np.array([[0, -1, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]])).float() * 0.5
        grad_kernel_s1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s2 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s = torch.stack((grad_kernel_s1, grad_kernel_s2, grad_kernel_s3))

        grad_kernel_x = grad_kernel_x.view(1, 1, *grad_kernel_x.size()).float()
        grad_kernel_y = grad_kernel_y.view(1, 1, *grad_kernel_y.size()).float()
        grad_kernel_s = grad_kernel_s.view(1, 1, *grad_kernel_s.size()).float()

        self.gradientx = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='zeros',
                                   padding=1)
        self.gradientx.weight = nn.Parameter(grad_kernel_x, requires_grad=False)
        # self.gradientx = self.gradientx.to(dev)

        self.gradienty = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=1)
        self.gradienty.weight = nn.Parameter(grad_kernel_y, requires_grad=False)
        # self.gradienty = self.gradienty.to(dev)

        self.gradients = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.gradients.weight = nn.Parameter(grad_kernel_s, requires_grad=False)
        # self.gradients = self.gradients.to(dev)

        grad_kernel_x1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x2 = torch.tensor(np.array([[0, 0, 0],
                                                [-1, 0, 1],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x = torch.stack((grad_kernel_x1, grad_kernel_x2, grad_kernel_x3))

        grad_kernel_x = grad_kernel_x.view(1, 1, *grad_kernel_x.size()).float()
        self.gradientx_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradientx_large.weight = nn.Parameter(grad_kernel_x, requires_grad=False)
        # self.gradientx_large = self.gradientx_large.to(self.img_dev)

        grad_kernel_y1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_y2 = torch.tensor(np.array([[0, -1, 0],
                                                [0, 0, 0],
                                                [0, 1, 0]])).float() * 0.5
        grad_kernel_y3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_y = torch.stack((grad_kernel_y1, grad_kernel_y2, grad_kernel_y3))

        grad_kernel_y = grad_kernel_y.view(1, 1, *grad_kernel_y.size()).float()
        self.gradienty_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradienty_large.weight = nn.Parameter(grad_kernel_y, requires_grad=False)
        # self.gradienty_large = self.gradienty_large.to(self.img_dev)

        grad_kernel_s1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s2 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s = torch.stack((grad_kernel_s1, grad_kernel_s2, grad_kernel_s3))
        grad_kernel_s = grad_kernel_s.view(1, 1, *grad_kernel_s.size()).float()

        self.gradients_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradients_large.weight = nn.Parameter(grad_kernel_s, requires_grad=False)
        # self.gradients_large = self.gradients_large.to(self.img_dev)


        # Localization Hessian Convolutions
        '''
                    |dxx dxy dxs|
        Hessian =   |dyx dyy dys|
                    |dsx dsy dss|
        Calculate each element of Hessian individually using convolution
        Then only loop through each max/min point individually once
        '''
        '''
        xx = torch.tensor(np.array([[0, 0, 0],
                                    [1, -2, 1],
                                    [0, 0, 0]])).float()
        xx = xx.view(1, 1, *xx.size()).float()
        self.xx_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.xx_conv.weight = nn.Parameter(xx, requires_grad=False)
        # self.xx_conv = self.xx_conv.to(dev)

        xy = torch.tensor(np.array([[0.5, -0.5, 0],
                                    [-0.5, 1, -0.5],
                                    [0, -0.5, 0.5]])).float()
        xy = xy.view(1, 1, *xy.size()).float()
        self.xy_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.xy_conv.weight = nn.Parameter(xy, requires_grad=False)
        # self.xy_conv = self.xy_conv.to(dev)

        xs_1 = torch.tensor(np.array([[0, 0, 0],
                                      [-1, -1, 0],
                                      [0, 0, 0]])).float() * 0.5
        xs_2 = torch.tensor(np.array([[0, 0, 0],
                                      [-1, 2, -1],
                                      [0, 0, 0]])).float() * 0.5
        xs_3 = torch.tensor(np.array([[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]])).float() * 0.5
        xs = torch.stack((xs_1, xs_2, xs_3))
        xs = xs.view(1, 1, *xs.size()).float()
        self.xs_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.xs_conv.weight = nn.Parameter(xs, requires_grad=False)
        # self.xs_conv = self.xs_conv.to(dev)

        yx = torch.tensor(np.array([[0.5, -0.5, 0],
                                    [-0.5, 1, -0.5],
                                    [0, -0.5, 0.5]])).float()
        yx = yx.view(1, 1, *yx.size()).float()
        self.yx_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.yx_conv.weight = nn.Parameter(yx, requires_grad=False)
        # self.yx_conv = self.yx_conv.to(dev)

        yy = torch.tensor(np.array([[0, 1, 0],
                                    [0, -2, 0],
                                    [0, 1, 0]])).float()
        yy = yy.view(1, 1, *yy.size()).float()
        self.yy_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.yy_conv.weight = nn.Parameter(yy, requires_grad=False)
        # self.yy_conv = self.yy_conv.to(dev)

        ys1 = torch.tensor(np.array([[0, 1, 0],
                                     [0, -1, 0],
                                     [0, 0, 0]])).float() * 0.5
        ys2 = torch.tensor(np.array([[0, -1, 0],
                                     [0, 2, 0],
                                     [0, -1, 0]])).float() * 0.5
        ys3 = torch.tensor(np.array([[0, 0, 0],
                                     [0, -1, 0],
                                     [0, 1, 0]])).float() * 0.5
        ys = torch.stack((ys1, ys2, ys3))
        # ys = ys.repeat(3, 1, 1, 1)
        ys = ys.view(1, 1, *ys.size()).float()
        self.ys_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.ys_conv.weight = nn.Parameter(ys, requires_grad=False)
        # self.ys_conv = self.ys_conv.to(dev)

        sx = xs  # Same kernel
        self.sx_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.sx_conv.weight = nn.Parameter(sx, requires_grad=False)
        # self.sx_conv = self.sx_conv.to(dev)

        sy = ys  # Same kernel
        self.sy_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.sy_conv.weight = nn.Parameter(sy, requires_grad=False)
        # self.sy_conv = self.sy_conv.to(dev)

        ss_1 = torch.tensor(np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])).float()
        ss_2 = torch.tensor(np.array([[0, 0, 0],
                                      [0, -2, 0],
                                      [0, 0, 0]])).float()
        ss_3 = torch.tensor(np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])).float()
        ss = torch.stack((ss_1, ss_2, ss_3))
        # ss = ss.repeat(3, 1, 1, 1)
        ss = ss.view(1, 1, *ss.size()).float()

        self.ss_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.ss_conv.weight = nn.Parameter(ss, requires_grad=False)
        # self.ss_conv = self.ss_conv.to(dev)
        '''
        return




# TODO Add this
class HessianKernel(nn.Module):
    """

    """

    def __init__(self, sigma=0.707107, kernel_size=21, scale=2, max_keys=2000):
        super().__init__()

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_scale = scale
        self.key_descriptors = {}
        self.max_num_keys = max_keys
        scale_factor = np.sqrt(2)
        sigma = (sigma, sigma*scale_factor, sigma*(scale_factor*2), sigma*(scale_factor*3), sigma*(scale_factor*4))

        whole_kernel = torch.zeros((5, kernel_size, kernel_size))
        kernel_size = [kernel_size] * 2
        self.sigma = [sigma] * 2
        self.gauss_scale_dict = {}
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for idx, sigma in enumerate(self.sigma[0]):
            self.gauss_scale_dict[idx] = sigma
            sigma = [sigma] * 2
            kernel = 1
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            # Make sure sum of values in gaussian kernel equals 1.
            gauss_kernel = kernel / torch.sum(kernel)
            whole_kernel[idx] = gauss_kernel

        # Make sure sum of values in gaussian kernel equals 1.
        # gauss_kernel = kernel / torch.sum(kernel)

        # Make 3-D to apply to all three channels at once
        gauss_kernel = whole_kernel[0].view(1, 1, *kernel_size).float()
        gauss_kernel2 = whole_kernel[1].view(1, 1, *kernel_size).float()
        gauss_kernel3 = whole_kernel[2].view(1, 1, *kernel_size).float()
        gauss_kernel4 = whole_kernel[3].view(1, 1, *kernel_size).float()
        gauss_kernel5 = whole_kernel[4].view(1, 1, *kernel_size).float()

        # gauss_kernel = gauss_kernel.repeat(3, *[1] * (gauss_kernel.dim() - 1)).float()

        """plt.imshow(gauss_kernel[0][0], cmap='gray')
        plt.title('filter 0')
        plt.show()

        plt.imshow(gauss_kernel[1][0], cmap='gray')
        plt.title('filter 1')
        plt.show()

        plt.imshow(gauss_kernel[2][0], cmap='gray')
        plt.title('filter 2')
        plt.show()"""

        # Det(Hassian) kernels below
        det_x_kernel = torch.tensor(np.array([[0, 0, 0],
                                              [1, -2, 1],
                                              [0, 0, 0]])).float()
        det_x_kernel = det_x_kernel.view(1, 1, *det_x_kernel.size()).float()
        det_y_kernel = torch.tensor(np.array([[0, 1, 0],
                                              [0, -2, 0],
                                              [0, 1, 0]])).float()
        det_y_kernel = det_y_kernel.view(1, 1, *det_y_kernel.size()).float()

        det_xy_kernel = 0.5 * torch.tensor(np.array([[1, -1, 0],
                                                     [-1, 2, -1],
                                                     [0, -1, 1]])).float()
        det_xy_kernel = det_xy_kernel.view(1, 1, *det_xy_kernel.size()).float()

        # Laplacian Kernel Below
        lap_kernel = torch.tensor(np.array([[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]])).float()
        lap_kernel = lap_kernel * self.sigma[0][0]  # Scale normalize laplacian kernel
        lap_kernel = lap_kernel.view(1, 1, *lap_kernel.size()).float()
        # Shape [out channels, in channels/groups, kernel size[0], kernel size[1]]

        # Output shape of 3 x img size - 3 gaussian scales for computing maxima/minima later on
        self.gauss_blur1 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur1.weight = nn.Parameter(gauss_kernel, requires_grad=False)
        # self.gauss_blur1 = self.gauss_blur1.to(dev)

        self.gauss_blur2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur2.weight = nn.Parameter(gauss_kernel2, requires_grad=False)
        # self.gauss_blur2 = self.gauss_blur2.to(dev)

        self.gauss_blur3 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur3.weight = nn.Parameter(gauss_kernel3, requires_grad=False)
        # self.gauss_blur3 = self.gauss_blur3.to(dev)

        self.gauss_blur4 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur4.weight = nn.Parameter(gauss_kernel4, requires_grad=False)
        # self.gauss_blur4 = self.gauss_blur4.to(dev)

        self.gauss_blur5 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, groups=1,
                                     padding_mode='reflect', padding=10)
        self.gauss_blur5.weight = nn.Parameter(gauss_kernel5, requires_grad=False)
        # self.gauss_blur5 = self.gauss_blur5.to(dev)

        # Compute Laplacian/
        # TODO Do we want to pad when it's an edge detector?
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, stride=1,  bias=False, groups=1, padding_mode='reflect',
                                   padding=1)
        self.laplacian.weight = nn.Parameter(lap_kernel, requires_grad=False)

        # Determinent of Hessian Convolutions
        self.det_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                               padding=1)
        self.det_x.weight = nn.Parameter(det_x_kernel, requires_grad=False)
        # self.det_x = self.det_x.to(dev)

        self.det_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                               padding=1)
        self.det_y.weight = nn.Parameter(det_y_kernel, requires_grad=False)
        # self.det_y = self.det_y.to(dev)

        self.det_xy = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                padding=1)
        self.det_xy.weight = nn.Parameter(det_xy_kernel, requires_grad=False)
        # self.det_xy = self.det_xy.to(dev)

        # For Maxima/Minima extraction
        self.max_3d = nn.MaxPool3d(kernel_size=3, stride=(3, 1, 1), padding=(1, 1, 1), return_indices=True)

        self.init_localization_kernels()

    def forward(self, images):
        with torch.no_grad():
            self.img_dev = images.device

            to_gray = torchvision.transforms.Grayscale()
            images = to_gray(images)
            # images = images * 255
            gauss_images = self.gauss_blur1(images)
            gauss_images2 = self.gauss_blur2(images)
            gauss_images3 = self.gauss_blur3(images)
            gauss_images4 = self.gauss_blur4(images)
            gauss_images5 = self.gauss_blur5(images)
            gauss_comb = torch.stack((gauss_images, gauss_images2, gauss_images3, gauss_images4, gauss_images5), dim=2)
            del gauss_images
            del gauss_images2
            del gauss_images3
            del gauss_images4
            del gauss_images5
            # image2 = gauss_images.numpy()[0][0]
            # image22 = gauss_images.numpy()[0][1]
            # plt.imshow(image2.reshape(image2.shape[0], image2.shape[1]), cmap='gray')
            # plt.title('filter 0')
            # plt.show()
            # plt.imshow(image22.reshape(image22.shape[0], image22.shape[1]), cmap='gray')
            # plt.title('filter 1')
            # plt.show()

            # image3 = det_hess.numpy()[0][0]
            # plt.imshow(image3.reshape(image3.shape[0], image3.shape[1]), cmap='gray')
            # plt.title('filter 0')
            # plt.show()

            # Extract Maxima/Minima (And Saddles for Det of Hessian?)
            # These max min are for a specific scale (middle of self.sigma)
            # Returns with shape of input, with all values 0 except for maxima or minima
            max_min = self.extract_maxmin(gauss_comb, threshold=2)

            maxmin_batch_dict = {}  # Shape: (num scale, batch, num_non_zero)
            for scale in max_min.keys():
                maxmin_id_dict = {}
                for batch in range(gauss_comb.size()[0]):  # TODO Confirm this is the right batch
                    max_min_idx = torch.nonzero(max_min[scale][batch])
                    maxmin_id_dict[batch] = max_min_idx
                maxmin_batch_dict[scale] = maxmin_id_dict

            # img1 = max_min[0][0]
            # plt.imshow(img1)
            # plt.title('Max/Min')
            # plt.show()

            # plt.imshow(images[0][0], cmap='gray')
            # count = 0
            # for point in max_min_idx:
            #     plt.scatter(point[-1], point[-2])
            #     count += 1
            #     if count == 500:
            #         break
            # plt.show()
            local_keys = self.localize(maxmin_batch_dict, gauss_comb)

            self.orientate(local_keys, gauss_comb)

        return self.key_descriptors

    def extract_maxmin(self, gauss_images, threshold):
        maxmin_dict = {}
        # TODO Loop through all 5? gaussian images
        for i in range(0, 5):

            det_x = self.det_x(gauss_images[:, :, 3])
            dey_y = self.det_y(gauss_images[:, :, 3])
            det_xy = self.det_xy(gauss_images[:, :, 3])
            det_hess = (det_x * dey_y) - (det_xy * det_xy)
            # det_hess = self.laplacian(gauss_images)
            del det_x
            del dey_y
            del det_xy

            maxes, max_idx = self.max_3d(det_hess)
            mins, min_idx = self.max_3d(-det_hess)

            # Apply thresholding
            max_thresh = maxes > threshold
            min_thresh = -mins < -threshold

            # Combine
            max_min = (maxes * max_thresh) + (mins * min_thresh)
            del maxes
            del mins
            maxmin_dict[i] = max_min  # Dict includes scale for each set of maxmins

        return maxmin_dict  # Shape (Batch, 1, 1024, 1024)

    def localize(self, max_min_points, images):
        localized_dict = {}

        # sx_all = self.gradients_large(self.gradientx_large(images))
        # sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
        # sy_all = self.gradients_large(self.gradienty_large(images))
        # sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
        ss_all = self.gradients_large(self.gradients_large(images))
        # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])

        # Max min points has shape (num scale, batch, num_non_zero)
        for scale in max_min_points.keys():

            if scale == 0:
                continue
            elif scale == 4:
                continue

            batched_dicts = {}  # Batch: Keys
            local_points = []
            all_points = []
            # Need to calculate gradient and hessian and solve resulting 3x3 system to localize points
            # Then check if offset is less than 0.5

            gradx = self.gradientx(images[:, :, scale])  # scale instead of 1
            grady = self.gradienty(images[:, :, scale])
            grads = self.gradients(images)
            grads = grads.view(grads.size()[0], 1, *grads.size()[-2:])  # Reduce dimensions by 1

            # Testing
            xx = self.gradientx(gradx)
            xy = self.gradientx(grady)
            xs = self.gradientx(grads)
            yx = self.gradienty(gradx)
            yy = self.gradienty(grady)
            ys = self.gradienty(grads)
            sx = self.gradients(self.gradientx_large(images))
            sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
            sy = self.gradients(self.gradienty_large(images))
            sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
            # ss = self.gradients_large(self.gradients_large(images))
            # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])
            # sx = sx_all[:, :, scale]
            # sy = sy_all[:, :, scale]
            ss = ss_all[:, :, scale]

            # TODO Faster maybe?
            first_img, second_img, third_img = images[:, :, scale-1:scale+2][0, 0]

            # xx = self.xx_conv(images[:, :, scale])
            # xy = self.xy_conv(images[:, :, scale])
            # xs = self.xs_conv(images)
            # xs = xs.view(xs.size()[0], 1, *xs.size()[-2:])
            # yx = self.yx_conv(images[:, :, scale])
            # yy = self.yy_conv(images[:, :, scale])
            # ys = self.ys_conv(images)
            # ys = ys.view(ys.size()[0], 1, *ys.size()[-2:])
            # sx = self.sx_conv(images)
            # sx = sx.view(sx.size()[0], 1, *sx.size()[-2:])
            # sy = self.sy_conv(images)
            # sy = sy.view(sy.size()[0], 1, *sy.size()[-2:])
            # ss = self.ss_conv(images)
            # ss = ss.view(ss.size()[0], 1, *ss.size()[-2:])

            # TODO Add image batching
            for batch, image in enumerate(images):
                for point in max_min_points[scale][batch]:
                    j = 0
                    i = 0
                    x, y = point[-2:]

                    if x == 0:
                        continue
                    elif y == 0:
                        continue

                    # TODO Testing
                    cube = torch.stack([first_img[x-1:x+2, y-1:y+2],
                                        second_img[x-1:x+2, y-1:y+2],
                                        third_img[x-1: x+2, y-1:y+2]])
                    cube_grad = self.calc_gradient(cube)
                    cube_hess = self.calc_hess(cube)

                    for _ in range(5):
                        # Compute gradient of center pixel from each 3x3x3 window

                        # Catches if i or j is massive and causes overflow error
                        if i > image[0, scale].size()[1]:
                            i = 10
                        if j > image[0, scale].size()[0]:
                            j = 10

                        if x+j >= image[0, scale].size()[0]:
                            j = (image[0, scale].size()[0] - x) - 1
                        if y+i >= image[0, scale].size()[1]:
                            i = (image[0, scale].size()[1] - y) - 1
                        if x+j < 0:
                            j = -int(x)
                        if y+i < 0:
                            i = -int(y)

                        local_grad = torch.stack((gradx[batch, :, x+j, y+i], grady[batch, :, x+j, y+i],
                                                  grads[batch, :, x+j, y+i]))
                        hess_x = torch.stack((xx[batch, :, x+j, y+i], xy[batch, :, x+j, y+i], xs[batch, :, x+j, y+i]))
                        hess_y = torch.stack((yx[batch, :, x+j, y+i], yy[batch, :, x+j, y+i], ys[batch, :, x+j, y+i]))
                        hess_s = torch.stack((sx[batch, :, x+j, y+i], sy[batch, :, x+j, y+i], ss[batch, :, x+j, y+i]))
                        local_hess = torch.stack((hess_x, hess_y, hess_s), dim=0).squeeze()  # TODO Confirm dimension

                        # TODO Confirm works for batched images
                        extremum_update = -torch.linalg.lstsq(local_hess.float(), local_grad.float(), rcond=None)[0]

                        # TODO Testing maybe faster or better?
                        extremum_update_cube = -torch.linalg.lstsq(cube_hess.float(), cube_grad.float(), rcond=None)[0]
                        if abs(extremum_update_cube[0]) < 0.5 and abs(extremum_update_cube[1]) < 0.5 and abs(
                                extremum_update_cube[2]) < 0.5:
                            print('This worked!')

                        # Check change is less than 0.5 in any direction
                        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                            # single_descriptor = torch.zeros(129)
                            # Threshold
                            single_point = image[:, scale, x, y] + 0.5 * torch.matmul(local_grad.squeeze(), extremum_update.view(3))
                            if single_point > 0.03:
                                edge_hess = torch.stack((torch.stack((xx[batch, 0, x, y], xy[batch, 0, x, y])),
                                                         torch.stack((yx[batch, 0, x, y], yy[batch, 0, x, y]))))
                                eigv, _ = torch.linalg.eig(edge_hess.float())
                                eigv = torch.real(eigv)
                                ratio = eigv[0] / eigv[1]

                                if ratio < 10:
                                    # This is the actual localized maxima/minima
                                    local_points.append(torch.tensor((batch, scale, point[1] + j, point[2] + i)))

                            else:
                                pass
                            break

                        # TODO Need better way to handle this
                        if torch.isinf(extremum_update[0]):
                            extremum_update[0] = j - 1
                            break
                        if torch.isinf(extremum_update[1]):
                            extremum_update[1] = i - 1
                            break

                        # TODO Need better way to handle this
                        if torch.isnan(extremum_update[0]):
                            extremum_update[0] = 1
                            break
                        if torch.isnan(extremum_update[1]):
                            extremum_update[1] = 1
                            break

                        j += int(torch.round(extremum_update[0]))
                        i += int(torch.round(extremum_update[1]))

                if len(local_points) > 0:
                    batched_dicts[batch] = torch.stack(local_points, dim=0)
                else:
                    batched_dicts[batch] = torch.tensor([])
            # local_points = torch.stack(local_points, dim=0)
            localized_dict[scale] = batched_dicts

        return localized_dict  # Scale: {batch: keys}

    def calc_gradient(self, points):
        # Gradient at center pixel [1,1,1] of 3x3x3 array using central difference formula
        dx = 0.5 * (points[1, 1, 2] - points[1, 1, 0])
        dy = 0.5 * (points[1, 2, 1] - points[1, 0, 1])
        ds = 0.5 * (points[2, 1, 1] - points[0, 1, 1])
        gradient = torch.stack([dx, dy, ds])

        return gradient

    def calc_hess(self, points):
        # Hessian at center pixel [1,1,1] of 3x3x3 array using central difference formula
        center_pix_value = points[1, 1, 1]
        dxx = points[1, 1, 2] - 2 * center_pix_value + points[1, 1, 0]
        dyy = points[1, 2, 1] - 2 * center_pix_value + points[1, 0, 1]
        dss = points[2, 1, 1] - 2 * center_pix_value + points[0, 1, 1]
        dxy = 0.25 * (points[1, 2, 2] - points[1, 2, 0] - points[1, 0, 2] + points[1, 0, 0])
        dxs = 0.25 * (points[2, 1, 2] - points[2, 1, 0] - points[0, 1, 2] + points[0, 1, 0])
        dys = 0.25 * (points[2, 2, 1] - points[2, 0, 1] - points[0, 2, 1] + points[0, 0, 1])
        x = torch.stack([dxx, dxy, dxs])
        y = torch.stack([dxy, dyy, dys])
        s = torch.stack([dxs, dys, dss])
        hessian = torch.stack([x, y, s], dim=1)

        return hessian

    def orientate(self, keys, images):

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        scale_descriptors = {}
        for scale in keys.keys():

            scale_sigma = self.gauss_scale_dict[scale]
            kernel_size = [16, 16]
            meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
            kernel_sigma = [scale_sigma * 1.5] * 2
            kernel = 1
            for size, std, mgrid in zip(kernel_size, kernel_sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            # Make sure sum of values in gaussian kernel equals 1.
            gauss_kernel = (kernel / torch.sum(kernel)).to(dev)

            gradx = self.gradientx(images[:, :, scale]) + 1e-6
            grady = self.gradienty(images[:, :, scale]) + 1e-6

            mag = torch.sqrt(torch.square(gradx) + torch.square(grady))
            orientation = torch.atan(grady / gradx)

            for batch in keys[scale]:
                batch_descriptors = []
                for point in keys[scale][batch]:
                    x, y = point[-2:]
                    if x < 9:
                        continue
                    elif y < 9:
                        continue
                    elif x > mag.size()[2] - 16:
                        continue
                    elif y > mag.size()[3] - 16:
                        continue
                    ind_mag = mag[:, :, x-8:x+8, y-8:y+8]
                    ind_ori = orientation[:, :, x-8:x+8, y-8:y+8]

                    # Account for negative values
                    ind_ori_mask = ind_ori < 0.
                    ind_ori[ind_ori_mask] = ind_ori[ind_ori_mask] + (2*np.pi)

                    # Create Histogram
                    orientation_prime = ind_ori / 10  # TODO Check this?
                    # orientation_prime = ind_ori
                    mag_weight = ind_mag * gauss_kernel

                    histo, bin_edges = torch.histogram(orientation_prime.to('cpu').float(), bins=36,
                                                       range=(0., 2*np.pi), weight=mag_weight.to('cpu'), density=False)
                    histo = histo.to(dev)

                    max_value = histo.max()
                    max_mask = histo >= max_value * 0.8

                    for idx in max_mask.nonzero():
                        # peak_value = histo.max()
                        single_descriptor = torch.zeros(132)
                        single_descriptor[0] = x
                        single_descriptor[1] = y
                        single_descriptor[2] = scale
                        single_descriptor[3] = histo[idx]

                        # Calculate Descriptor
                        # Compute histograms for 4x4 regions of the 16x16 window
                        desc = torch.zeros((4, 4, 8))
                        for x_count, x_idx in enumerate(range(0, ind_mag.size()[2], 4)):
                            for y_count, y_idx in enumerate(range(0, ind_mag.size()[3], 4)):
                                indiv_mag = mag[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                                indiv_ori = ind_ori[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                                indiv_mag_weight = indiv_mag * gauss_kernel[6:10, 6:10]

                                # TODO Need to calc Orientation prime?
                                # TODO Better off computing on my own using GPU?
                                eight_histo, eight_bin_edges = torch.histogram(indiv_ori.to('cpu').float(), bins=8,
                                                                               range=(0., 2 * np.pi),
                                                                               weight=indiv_mag_weight.to('cpu'), density=False)
                                eight_histo = eight_histo.to(dev)
                                desc[x_count, y_count, :] = eight_histo

                        # Flatten 4x4 array of histos
                        flat_desc = torch.flatten(desc)
                        single_descriptor[4:] = flat_desc

                    # TODO Confirm below
                    batch_descriptors.append(single_descriptor)
                if len(batch_descriptors) == 0:
                    batch_descriptors.append(torch.zeros(132))
                scale_descriptors[batch] = torch.stack(batch_descriptors).to(self.img_dev)

            self.key_descriptors[scale] = scale_descriptors

        return

    def single_orientate(self, scale, batch, x, y, images):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        scale_descriptors = {}

        # For a single scale below (scale.keys())
        scale_sigma = self.gauss_scale_dict[scale]
        kernel_size = [16, 16]
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernel_sigma = [scale_sigma * 1.5] * 2
        kernel = 1
        for size, std, mgrid in zip(kernel_size, kernel_sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        gauss_kernel = (kernel / torch.sum(kernel)).to(dev)

        gradx = self.gradientx(images[:, :, scale]) + 1e-6
        grady = self.gradienty(images[:, :, scale]) + 1e-6
        mag = torch.sqrt(torch.square(gradx) + torch.square(grady))
        orientation = torch.atan(grady / gradx)

        # For a single batch key below
        batch_descriptors = []
        if x < 9:
            return
        elif y < 9:
            return
        elif x > mag.size()[2] - 16:
            return
        elif y > mag.size()[3] - 16:
            return
        ind_mag = mag[:, :, x - 8:x + 8, y - 8:y + 8]
        ind_ori = orientation[:, :, x - 8:x + 8, y - 8:y + 8]

        # Account for negative values
        ind_ori_mask = ind_ori < 0.
        ind_ori[ind_ori_mask] = ind_ori[ind_ori_mask] + (2 * np.pi)

        # Create Histogram
        orientation_prime = ind_ori / 10  # TODO Check this?
        # orientation_prime = ind_ori
        mag_weight = ind_mag * gauss_kernel

        histo, bin_edges = torch.histogram(orientation_prime.to('cpu').float(), bins=36,
                                           range=(0., 2 * np.pi), weight=mag_weight.to('cpu'), density=False)
        histo = histo.to(dev)

        max_value = histo.max()
        max_mask = histo >= max_value * 0.8

        for idx in max_mask.nonzero():
            # peak_value = histo.max()
            single_descriptor = torch.zeros(132)
            single_descriptor[0] = x
            single_descriptor[1] = y
            single_descriptor[2] = scale
            single_descriptor[3] = histo[idx]

            # Calculate Descriptor
            # Compute histograms for 4x4 regions of the 16x16 window
            desc = torch.zeros((4, 4, 8))
            for x_count, x_idx in enumerate(range(0, ind_mag.size()[2], 4)):
                for y_count, y_idx in enumerate(range(0, ind_mag.size()[3], 4)):
                    indiv_mag = mag[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                    indiv_ori = ind_ori[:, :, x_idx:x_idx + 4, y_idx:y_idx + 4]
                    indiv_mag_weight = indiv_mag * gauss_kernel[6:10, 6:10]

                    # TODO Need to calc Orientation prime?
                    # TODO Better off computing on my own using GPU?
                    eight_histo, eight_bin_edges = torch.histogram(indiv_ori.to('cpu').float(), bins=8,
                                                                   range=(0., 2 * np.pi),
                                                                   weight=indiv_mag_weight.to('cpu'), density=False)
                    eight_histo = eight_histo.to(dev)
                    desc[x_count, y_count, :] = eight_histo

            # Flatten 4x4 array of histos
            flat_desc = torch.flatten(desc)
            single_descriptor[4:] = flat_desc

        self.key_descriptors[scale][batch] = scale_descriptors

        return

    def init_localization_kernels(self):

        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Gradient for point localization
        zero_kernel = torch.tensor(np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]])).float()
        grad_kernel_x = torch.tensor(np.array([[0, 0, 0],
                                               [-1, 0, 1],
                                               [0, 0, 0]])).float() * 0.5
        grad_kernel_y = torch.tensor(np.array([[0, -1, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]])).float() * 0.5
        grad_kernel_s1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s2 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s = torch.stack((grad_kernel_s1, grad_kernel_s2, grad_kernel_s3))

        grad_kernel_x = grad_kernel_x.view(1, 1, *grad_kernel_x.size()).float()
        grad_kernel_y = grad_kernel_y.view(1, 1, *grad_kernel_y.size()).float()
        grad_kernel_s = grad_kernel_s.view(1, 1, *grad_kernel_s.size()).float()

        self.gradientx = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='zeros',
                                   padding=1)
        self.gradientx.weight = nn.Parameter(grad_kernel_x, requires_grad=False)
        # self.gradientx = self.gradientx.to(dev)

        self.gradienty = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=1)
        self.gradienty.weight = nn.Parameter(grad_kernel_y, requires_grad=False)
        # self.gradienty = self.gradienty.to(dev)

        self.gradients = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.gradients.weight = nn.Parameter(grad_kernel_s, requires_grad=False)
        # self.gradients = self.gradients.to(dev)

        grad_kernel_x1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x2 = torch.tensor(np.array([[0, 0, 0],
                                                [-1, 0, 1],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_x = torch.stack((grad_kernel_x1, grad_kernel_x2, grad_kernel_x3))

        grad_kernel_x = grad_kernel_x.view(1, 1, *grad_kernel_x.size()).float()
        self.gradientx_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradientx_large.weight = nn.Parameter(grad_kernel_x, requires_grad=False)
        # self.gradientx_large = self.gradientx_large.to(self.img_dev)

        grad_kernel_y1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_y2 = torch.tensor(np.array([[0, -1, 0],
                                                [0, 0, 0],
                                                [0, 1, 0]])).float() * 0.5
        grad_kernel_y3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_y = torch.stack((grad_kernel_y1, grad_kernel_y2, grad_kernel_y3))

        grad_kernel_y = grad_kernel_y.view(1, 1, *grad_kernel_y.size()).float()
        self.gradienty_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradienty_large.weight = nn.Parameter(grad_kernel_y, requires_grad=False)
        # self.gradienty_large = self.gradienty_large.to(self.img_dev)

        grad_kernel_s1 = torch.tensor(np.array([[0, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s2 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s3 = torch.tensor(np.array([[0, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 0]])).float() * 0.5
        grad_kernel_s = torch.stack((grad_kernel_s1, grad_kernel_s2, grad_kernel_s3))
        grad_kernel_s = grad_kernel_s.view(1, 1, *grad_kernel_s.size()).float()

        self.gradients_large = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, groups=1,
                                         padding_mode='zeros',
                                         padding=(1, 1, 1))
        self.gradients_large.weight = nn.Parameter(grad_kernel_s, requires_grad=False)
        # self.gradients_large = self.gradients_large.to(self.img_dev)


        # Localization Hessian Convolutions
        '''
                    |dxx dxy dxs|
        Hessian =   |dyx dyy dys|
                    |dsx dsy dss|
        Calculate each element of Hessian individually using convolution
        Then only loop through each max/min point individually once
        '''
        '''
        xx = torch.tensor(np.array([[0, 0, 0],
                                    [1, -2, 1],
                                    [0, 0, 0]])).float()
        xx = xx.view(1, 1, *xx.size()).float()
        self.xx_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.xx_conv.weight = nn.Parameter(xx, requires_grad=False)
        # self.xx_conv = self.xx_conv.to(dev)

        xy = torch.tensor(np.array([[0.5, -0.5, 0],
                                    [-0.5, 1, -0.5],
                                    [0, -0.5, 0.5]])).float()
        xy = xy.view(1, 1, *xy.size()).float()
        self.xy_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.xy_conv.weight = nn.Parameter(xy, requires_grad=False)
        # self.xy_conv = self.xy_conv.to(dev)

        xs_1 = torch.tensor(np.array([[0, 0, 0],
                                      [-1, -1, 0],
                                      [0, 0, 0]])).float() * 0.5
        xs_2 = torch.tensor(np.array([[0, 0, 0],
                                      [-1, 2, -1],
                                      [0, 0, 0]])).float() * 0.5
        xs_3 = torch.tensor(np.array([[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]])).float() * 0.5
        xs = torch.stack((xs_1, xs_2, xs_3))
        xs = xs.view(1, 1, *xs.size()).float()
        self.xs_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.xs_conv.weight = nn.Parameter(xs, requires_grad=False)
        # self.xs_conv = self.xs_conv.to(dev)

        yx = torch.tensor(np.array([[0.5, -0.5, 0],
                                    [-0.5, 1, -0.5],
                                    [0, -0.5, 0.5]])).float()
        yx = yx.view(1, 1, *yx.size()).float()
        self.yx_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.yx_conv.weight = nn.Parameter(yx, requires_grad=False)
        # self.yx_conv = self.yx_conv.to(dev)

        yy = torch.tensor(np.array([[0, 1, 0],
                                    [0, -2, 0],
                                    [0, 1, 0]])).float()
        yy = yy.view(1, 1, *yy.size()).float()
        self.yy_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, groups=1, padding_mode='reflect',
                                 padding=1)
        self.yy_conv.weight = nn.Parameter(yy, requires_grad=False)
        # self.yy_conv = self.yy_conv.to(dev)

        ys1 = torch.tensor(np.array([[0, 1, 0],
                                     [0, -1, 0],
                                     [0, 0, 0]])).float() * 0.5
        ys2 = torch.tensor(np.array([[0, -1, 0],
                                     [0, 2, 0],
                                     [0, -1, 0]])).float() * 0.5
        ys3 = torch.tensor(np.array([[0, 0, 0],
                                     [0, -1, 0],
                                     [0, 1, 0]])).float() * 0.5
        ys = torch.stack((ys1, ys2, ys3))
        # ys = ys.repeat(3, 1, 1, 1)
        ys = ys.view(1, 1, *ys.size()).float()
        self.ys_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.ys_conv.weight = nn.Parameter(ys, requires_grad=False)
        # self.ys_conv = self.ys_conv.to(dev)

        sx = xs  # Same kernel
        self.sx_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.sx_conv.weight = nn.Parameter(sx, requires_grad=False)
        # self.sx_conv = self.sx_conv.to(dev)

        sy = ys  # Same kernel
        self.sy_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.sy_conv.weight = nn.Parameter(sy, requires_grad=False)
        # self.sy_conv = self.sy_conv.to(dev)

        ss_1 = torch.tensor(np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])).float()
        ss_2 = torch.tensor(np.array([[0, 0, 0],
                                      [0, -2, 0],
                                      [0, 0, 0]])).float()
        ss_3 = torch.tensor(np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])).float()
        ss = torch.stack((ss_1, ss_2, ss_3))
        # ss = ss.repeat(3, 1, 1, 1)
        ss = ss.view(1, 1, *ss.size()).float()

        self.ss_conv = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(3, 1, 1), bias=False, groups=1,
                                   padding_mode='zeros',
                                   padding=(0, 1, 1))
        self.ss_conv.weight = nn.Parameter(ss, requires_grad=False)
        # self.ss_conv = self.ss_conv.to(dev)
        '''
        return


class Fuse(nn.Module):
    """
    Class for learning and fusing the keypoints and relevant feature map before detection.
    First 2 layers passed are keypoints. The third is the feature map
    """
    def __init__(self, num_keys):
        super().__init__()
        self.num_keys = num_keys
        return

    def forward(self, x):
        key1, key2 = x  # Break apart
        batch_keys = []
        batch_max = 0
        # Each Key: {gauss scale: {batch: keys}}
        for batch in key1[0].keys():
            b1, b2 = [], []
            keys = []
            for scale1, scale2 in zip(key1, key2):
                b1.append(key1[scale1][batch])
                # b2.append(key2[scale2][batch])
                device = key1[scale1][batch].device

                keys.append(key1[scale1][batch])
                # keys.append(key2[scale2][batch])

            # if torch.cat(keys).size()[0] < self.num_keys:
            #     diff = self.num_keys - torch.cat(keys).size()[0]
            #     zeros = torch.zeros((diff, 132)).to(device)
            #     keys.append(zeros)

            batch_keys.append(torch.unsqueeze(torch.cat(keys), dim=0))

        for item in batch_keys:
            if item.size()[1] > batch_max:
                batch_max = item.size()[1]

        for id, item in enumerate(batch_keys):
            if item.size()[1] < batch_max:
                diff = batch_max - item.size()[1]
                zs = torch.zeros((1, diff, 132))
                batch_keys[id] = torch.cat((item, zs))

        total_keys = torch.cat(batch_keys)
        total_keys = torch.unsqueeze(total_keys, dim=1)


        # Finish with size (batch, 1, num_keys, 132)
        # total_keys = total_keys.to(device)

        return total_keys


class AdaptPool(nn.Module):
    def __init__(self, out1, out2):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool2d((out1, out2))

    def forward(self, x):
        x = self.pooler(x)
        return x


class Add(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        fm, keys = x

        if fm.size()[-1] == 32:
            keys = keys[:, 0, 0:32, 0:32]
        elif fm.size()[-1] == 16:
            keys = keys[:, 0, 0:16, 0:16]
        elif fm.size()[-1] == 8:
            keys = keys[:, 0, 0:8, 0:8]
        # elif fm.size()[-1] != keys.size()[-1]:
        #     keys = keys[:, 0, 0:fm.size()[-2], 0:fm.size()[-1]]

        added_x = torch.add(fm, keys, alpha=1)
        return added_x


class Linear(nn.Module):
    def __init__(self, filt, in1, dim1, dim2):
        super().__init__()
        self.linear = nn.Linear(filt*in1*2, dim1*dim2)
        self.in1 = in1
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        # x = x[0]
        # TODO Maybe reshape -> Linear -> Reshape
        x = torch.reshape(x, (x.size()[0], 1, 1, -1))  # 200 * 100 * num_filters
        x = self.linear(x)
        x = torch.reshape(x, (x.size()[0], 1, self.dim1, self.dim2))
        return x


class StrideConv(nn.Module):
    def __init__(self, c1, c2, k1=1, k2=1, s1=1, s2=1, p=0, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, (k1, k2), (s1, s2), padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv1D(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride)

    def forward(self, x):
        x = self.conv(x)
        return x


class KeyDetect(Detect):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(KeyDetect, self).__init__(nc, anchors, ch, inplace)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):  # -1 to account for added Key layer?
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


class KeyModel(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_custom_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        self.input_img = None

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        # x is the input image initially
        self.input_img = x
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f == -2:
                x = self.input_img
            elif m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_custom_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is KeyDetect:  # TODO May need adjustments
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # TODO Add Keypoint layer
        elif m is HessianKernelGood:
            print('test')  # TODO Confirm what args, f, etc are
        # TODO Add Fuse layer
        elif m is Fuse:
            c2 = 1
        elif m is Add:
            # TODO Add c2 line here Confirm
            c2 = ch[f[0]]
        elif m is Linear:
            c2 = 1
        elif m is AdaptPool:
            c2 = 1
        elif m is StrideConv:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is Conv1D:
            c1, c2 = ch[f], args[0]
            args = [c1, *args]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class ComputeLossIoU(ComputeLoss):
    def __init__(self, model, autobalance=False):
        super(ComputeLossIoU, self).__init__(model, autobalance)
        return

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch