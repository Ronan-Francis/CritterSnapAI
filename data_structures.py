from PIL import Image
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ImageObject:
    """
    Data container for storing an image, its capture date/time, and the file path.
    """
    image: Image.Image
    date: datetime
    file_path: str

    def get_image(self) -> Image.Image:
        return self.image

    def get_date(self) -> datetime:
        return self.date

    def get_file_path(self) -> str:
        return self.file_path
