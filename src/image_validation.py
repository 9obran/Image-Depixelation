from pathlib import Path
import glob
from PIL import Image
import numpy as np
import logging
from typing import List, Tuple


class InvalidFileExtensionError(ValueError):
    pass


class InvalidFileSizeError(ValueError):
    pass


class LowImageVarianceError(ValueError):
    pass


def get_image_files(input_dir: Path) -> List[Path]:
    """Get a list of image file paths in the input directory."""
    return sorted(input_dir.glob("**/*"))


def validate_file_extension(file: Path) -> bool:
    """Check if the file has a valid image extension."""
    return file.suffix.lower() in (".jpg", ".jpeg")


def validate_file_size(file: Path) -> bool:
    """Check if the file size is within the allowed range."""
    return file.stat().st_size <= 250000


def validate_image_variance(image: Image.Image, threshold: float) -> bool:
    """Check if the image variance is above the given threshold."""
    grayscale_image = image.convert("L")
    variance = np.var(np.array(grayscale_image))
    return variance >= threshold


def process_image(file: Path, threshold: float) -> Tuple[bool, str]:
    """Validate the image and return a tuple (is_valid, error_message)."""
    try:
        if not validate_file_extension(file):
            raise InvalidFileExtensionError("Invalid file extension")
        if not validate_file_size(file):
            raise InvalidFileSizeError("Invalid file size")

        with Image.open(file) as image:
            image.verify()
            if not validate_image_variance(image, threshold):
                raise LowImageVarianceError("Low variance")

            image_hash = hash(image.tobytes())

        return True, "", image_hash
    except (InvalidFileExtensionError, InvalidFileSizeError, LowImageVarianceError, IOError) as e:
        return False, str(e), None


def validate_images(input_dir: Path, output_dir: Path, log_file: Path, threshold: float):
    """Validate images in the input directory and log any errors."""
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist")

    image_files = get_image_files(input_dir)
    if not log_file.exists():
        log_file.touch()


    logging.basicConfig(filename=str(log_file), level=logging.ERROR, format="%(asctime)s - %(message)s")

    valid_files = 0
    hash_list = []
    for file in image_files:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        is_valid, error_message, image_hash = process_image(file, threshold)
        if not is_valid:
            logging.error(f"{file.name} - {error_message}")
        else:
            valid_files += 1
            hash_list.append(image_hash)
