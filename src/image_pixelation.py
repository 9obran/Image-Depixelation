import numpy as np
from typing import Tuple


def prepare_pixelated_image(
        image: np.ndarray,
        start_x: int,
        start_y: int,
        pixelated_width: int,
        pixelated_height: int,
        pixel_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a pixelated version of the input image, a known array, and a target array.

    Args:
        image (np.ndarray): Input image array with shape (channels, height, width).
        start_x (int): Horizontal starting coordinate of the pixelated area.
        start_y (int): Vertical starting coordinate of the pixelated area.
        pixelated_width (int): Width of the pixelated area.
        pixelated_height (int): Height of the pixelated area.
        pixel_size (int): Size of the individual pixels in the pixelated area.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the pixelated image,
                                                  known array, and target array.

    Raises:
        ValueError: If the input image shape is not valid or the pixelated area exceeds the input image dimensions.
    """
    if not (len(image.shape) == 3 and image.shape[1] > 0 and image.shape[2] > 0):
        raise ValueError("Invalid input image shape. Expected shape: (channels, height, width)")

    if pixelated_width < 2 or pixelated_height < 2 or pixel_size < 2:
        raise ValueError("Width, height, and pixel_size must be greater than or equal to 2")

    if start_x < 0 or start_x + pixelated_width > image.shape[2] or start_y < 0 or start_y + pixelated_height > image.shape[1]:
        raise ValueError("Pixelated area exceeds input image dimensions")

    pixelated_image = np.copy(image)
    known_array = np.ones(image.shape, dtype=bool)
    target_array = np.zeros(image.shape, dtype=image.dtype)

    for i in range(start_x, start_x + pixelated_width, pixel_size):
        for j in range(start_y, start_y + pixelated_height, pixel_size):
            block_width = min(pixel_size, start_x + pixelated_width - i)
            block_height = min(pixel_size, start_y + pixelated_height - j)
            known_array[:, j:j + block_height, i:i + block_width] = False
            target_array[:, j:j + block_height, i:i + block_width] = image[:, j:j + block_height, i:i + block_width]
            pixelated_image[:, j:j + block_height, i:i + block_width] = np.mean(image[:, j:j + block_height, i:i + block_width], axis=(1, 2))[:, None, None]

    return pixelated_image, known_array, target_array
