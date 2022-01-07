from abc import ABC, abstractmethod


class DnnModel(ABC):
    @abstractmethod
    def detect_single_image(self, img_path):
        pass
