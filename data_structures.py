from PIL import Image
from datetime import datetime

class ImageObject:
    """
    A minimal data container for storing image, its captured date/time, and file path.
    """
    def __init__(self, image: Image.Image, date: datetime, file_path: str):
        self._image = image
        self._date = date
        self._file_path = file_path

    def get_image(self) -> Image.Image:
        return self._image

    def get_date(self) -> datetime:
        return self._date

    def get_file_path(self) -> str:
        return self._file_path

    def set_image(self, image: Image.Image):
        self._image = image

    def set_date(self, date: datetime):
        self._date = date

    def set_file_path(self, file_path: str):
        self._file_path = file_path
