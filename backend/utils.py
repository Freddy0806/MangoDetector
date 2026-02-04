"""Utility helpers for image I/O used by server code."""
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def read_image_bytes(file) -> Image.Image:
    """Read image bytes from a file-like or raw bytes and return a PIL RGB image.

    Args:
        file: UploadFile-like object (has .read) or raw bytes

    Raises:
        RuntimeError: if image cannot be decoded
    """
    try:
        if hasattr(file, 'read'):
            data = file.read()
        else:
            data = file
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e:
        logger.exception('Error leyendo imagen desde bytes: %s', e)
        raise RuntimeError(f'No se pudo decodificar la imagen: {e}')
