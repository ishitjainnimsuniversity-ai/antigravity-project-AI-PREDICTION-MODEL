"""
DDR Dataset & Fundus Multi-Disease Preprocessing Module
======================================================
Based on DDR-Dataset (Diabetic Retinopathy Classification, Lesion Segmentation)
and linchundan88/fundus_multiple_diseases_web.

Provides:
- del_black_or_white: Auto-detects and trims extreme black/white borders with adaptive padding.
- detect_xyr: Circular Field of View (FOV) detection via cv2.HoughCircles with middle-line fallback.
- my_crop_xyr: True geometric bounding crop around detected retinal circle.
- add_black_margin: Preserves boundary margins for rotation/data augmentation.
- my_preprocess: Complete fundus normalization pipeline.
"""

import os
import cv2
import numpy as np

DEL_PADDING_RATIO = 0.02   # Used for del_black_or_white
CROP_PADDING_RATIO = 0.02  # Used for my_crop_xyr

# Margin thresholds
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles radius ratios relative to minimum dimension
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.60


def del_black_or_white(img1: np.ndarray) -> np.ndarray:
    """Deletes non-informative black or overexposed white borders from fundus image."""
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    width, height = (img1.shape[1], img1.shape[0])
    left, bottom = 0, 0
    right, top = img1.shape[1], img1.shape[0]

    padding = int(min(width, height) * DEL_PADDING_RATIO)

    # Left boundary
    for i in range(width):
        col = img1[:, i, :]
        if np.sum(col) > THRETHOLD_LOW * col.shape[0] * col.shape[1] and \
           np.sum(col) < THRETHOLD_HIGH * col.shape[0] * col.shape[1]:
            left = i
            break
    left = max(0, left - padding)

    # Right boundary
    for i in range(width - 1, -1, -1):
        col = img1[:, i, :]
        if np.sum(col) > THRETHOLD_LOW * col.shape[0] * col.shape[1] and \
           np.sum(col) < THRETHOLD_HIGH * col.shape[0] * col.shape[1]:
            right = i
            break
    right = min(width, right + padding)

    # Bottom boundary
    for i in range(height):
        row = img1[i, :, :]
        if np.sum(row) > THRETHOLD_LOW * row.shape[0] * row.shape[1] and \
           np.sum(row) < THRETHOLD_HIGH * row.shape[0] * row.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    # Top boundary
    for i in range(height - 1, -1, -1):
        row = img1[i, :, :]
        if np.sum(row) > THRETHOLD_LOW * row.shape[0] * row.shape[1] and \
           np.sum(row) < THRETHOLD_HIGH * row.shape[0] * row.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    if bottom >= top or left >= right:
        return img1

    img2 = img1[bottom:top, left:right, :]
    return img2


def detect_xyr(img_source):
    """
    Detects retinal circle center (x, y) and radius r using HoughCircles.
    Falls back to middle-line intensity distribution if circle is partially occluded.
    """
    if isinstance(img_source, str):
        if not os.path.exists(img_source):
            raise FileNotFoundError(f"Image file not found: {img_source}")
        img = cv2.imread(img_source)
        if img is None:
            raise ValueError(f"Image file error or corrupt: {img_source}")
    else:
        img = img_source

    width = img.shape[1]
    height = img.shape[0]
    my_min_dim = min(width, height)

    my_min_radius = round(my_min_dim * MIN_REDIUS_RATIO)
    my_max_radius = round(my_min_dim * MAX_REDIUS_RATIO)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, minDist=max(100, int(my_min_dim * 0.4)),
        param1=120, param2=32,
        minRadius=my_min_radius,
        maxRadius=my_max_radius
    )

    found_circle = False
    x, y, r = 0, 0, 0

    if circles is not None:
        circles_arr = np.round(circles[0, :]).astype("int")
        if len(circles_arr) >= 1:
            x1, y1, r1 = circles_arr[0]
            # Ensure detected circle center is plausibly near the optical axis
            if (2 / 5 * width < x1 < 3 / 5 * width) and (2 / 5 * height < y1 < 3 / 5 * height):
                x, y, r = int(x1), int(y1), int(r1)
                found_circle = True

    if not found_circle:
        # Fallback: Image center & median line intensity distribution
        x = width // 2
        y = height // 2
        mid_row = img[int(height / 2), :, :] if img.ndim == 3 else img[int(height / 2), :]
        temp_x = mid_row.sum(axis=-1) if img.ndim == 3 else mid_row
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)
        r = max(my_min_radius, min(r, my_max_radius))

    return found_circle, x, y, r


def my_crop_xyr(img_source, x, y, r, crop_size=None):
    """Crops the fundus image strictly around the detected FOV circle."""
    if isinstance(img_source, str):
        image1 = cv2.imread(img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise ValueError("Image input is None in my_crop_xyr")

    original_height, original_width = image1.shape[:2]
    img_padding = int(min(original_width, original_height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, original_width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, original_height - 1))

    if original_width >= original_height:
        if original_height >= 2 * (r + img_padding):
            image1 = image1[image_bottom:image_top, image_left:image_right]
        else:
            image1 = image1[:, image_left:image_right]
    else:
        if original_width >= 2 * (r + img_padding):
            image1 = image1[image_bottom:image_top, image_left:image_right]
        else:
            image1 = image1[image_bottom:image_top, :]

    if crop_size is not None and image1.size > 0:
        image1 = cv2.resize(image1, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    return image1


def add_black_margin(img_source, add_black_pixel_ratio=0.05):
    """Adds symmetric black margins to protect boundary vessels during spatial augmentation."""
    if isinstance(img_source, str):
        image1 = cv2.imread(img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise ValueError("Image input is None in add_black_margin")

    height, width = image1.shape[:2]
    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)
    channels = 3 if image1.ndim == 3 else 1

    if channels == 3:
        img_h = np.zeros((add_black_pixel, width, 3), dtype=image1.dtype)
        img_v = np.zeros((height + add_black_pixel * 2, add_black_pixel, 3), dtype=image1.dtype)
    else:
        img_h = np.zeros((add_black_pixel, width), dtype=image1.dtype)
        img_v = np.zeros((height + add_black_pixel * 2, add_black_pixel), dtype=image1.dtype)

    image1 = np.concatenate((img_h, image1, img_h), axis=0)
    image1 = np.concatenate((img_v, image1, img_v), axis=1)
    return image1


def my_preprocess(img_source, crop_size=512, train_or_valid='valid', img_file_dest=None):
    """Complete DDR / Fundus Preprocessing Pipeline."""
    if isinstance(img_source, str):
        image1 = cv2.imread(img_source)
    else:
        image1 = img_source.copy()

    if image1 is None:
        raise ValueError("Invalid image provided to my_preprocess")

    image1 = del_black_or_white(image1)
    min_dim = min(image1.shape[0], image1.shape[1])
    if min_dim < 100:
        return None

    # Multi-scale downscale if excessively large
    image_size_before_hough = crop_size * 2
    if min_dim > image_size_before_hough:
        crop_ratio = image_size_before_hough / min_dim
        image1 = cv2.resize(image1, None, fx=crop_ratio, fy=crop_ratio, interpolation=cv2.INTER_AREA)

    found_circle, x, y, r = detect_xyr(image1)

    if train_or_valid == 'train':
        image1 = my_crop_xyr(image1, x, y, r)
        image1 = add_black_margin(image1, add_black_pixel_ratio=0.07)
        image1 = cv2.resize(image1, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    else:
        image1 = my_crop_xyr(image1, x, y, r, crop_size)

    if img_file_dest is not None:
        os.makedirs(os.path.dirname(os.path.abspath(img_file_dest)), exist_ok=True)
        cv2.imwrite(img_file_dest, image1)

    return image1
