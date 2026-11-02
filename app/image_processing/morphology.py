"""
TreadVision: Morphological Operations for Noise Removal and Tread Cleanup
"""
from typing import Tuple
import cv2
import numpy as np


def get_morph_kernel(shape: str = "rect", size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Generate morphological structuring element.
    Shapes supported: 'rect', 'ellipse', 'cross'.
    """
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    cv_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv_shape, size)


def apply_morphology(
    image: np.ndarray,
    operation: str = "closing",
    kernel_shape: str = "rect",
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1
) -> np.ndarray:
    """
    Apply OpenCV morphological operations to clean tread contours and grooves.
    - 'erosion': shrinks bright regions, widens dark grooves.
    - 'dilation': expands bright regions, closes thin dark gaps.
    - 'opening': erosion followed by dilation; removes isolated noise specks.
    - 'closing': dilation followed by erosion; bridges small interruptions in continuous grooves.
    """
    kernel = get_morph_kernel(kernel_shape, kernel_size)
    op = operation.lower()

    if op == "erosion":
        return cv2.erode(image, kernel, iterations=iterations)
    elif op == "dilation":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif op == "opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op == "gradient":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op == "tophat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op == "blackhat":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    return image.copy()
