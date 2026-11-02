"""
TreadVision: Image Validation, Resizing, Grayscale Conversion and Noise Reduction
"""
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def validate_image(image_input) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate that the input is a valid image (path, numpy array, or PIL Image/bytes)
    with acceptable dimensions and channels.
    """
    if image_input is None:
        return False, "Input image is null or empty.", None

    img_bgr = None

    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                return False, f"Could not read image file from path: {image_input}", None
        else:
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return False, "Could not decode byte buffer into an image.", None
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input)
        if len(img_rgb.shape) == 2:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
        elif img_rgb.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif len(image_input.shape) == 3:
            img_bgr = image_input.copy()
        else:
            return False, f"Unexpected array dimensions: {image_input.shape}", None
    else:
        return False, f"Unsupported image input type: {type(image_input)}", None

    h, w = img_bgr.shape[:2]
    if h < 64 or w < 64:
        return False, f"Image dimensions too small ({w}x{h}). Minimum required is 64x64.", None

    return True, "Image is valid.", img_bgr


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512), keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Normalize image dimensions for consistent computer vision processing.
    If keep_aspect_ratio is True, letterboxes the image with zero-padding.
    """
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    return padded


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB or BGR image to single-channel 8-bit grayscale.
    """
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray_image: np.ndarray, method: str = "bilateral", **kwargs) -> np.ndarray:
    """
    Noise reduction using Gaussian, Median, or Bilateral filtering.
    Bilateral filtering preserves sharp tread groove boundaries while smoothing grain.
    """
    method = method.lower()
    if method == "gaussian":
        ksize = kwargs.get("ksize", (5, 5))
        sigma_x = kwargs.get("sigma_x", 1.0)
        return cv2.GaussianBlur(gray_image, ksize, sigma_x)
    elif method == "median":
        ksize = kwargs.get("ksize", 5)
        if isinstance(ksize, tuple):
            ksize = ksize[0]
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(gray_image, ksize)
    elif method == "bilateral":
        d = kwargs.get("d", 9)
        sigma_color = kwargs.get("sigma_color", 75)
        sigma_space = kwargs.get("sigma_space", 75)
        return cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
    else:
        # Default fallback to bilateral
        return cv2.bilateralFilter(gray_image, 9, 75, 75)
