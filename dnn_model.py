from abc import ABC, abstractmethod


class DnnModel(ABC):
    @abstractmethod
    def detect_single_image(self, img_path):
        pass

    @abstractmethod
    def detect_single_frame(self, frame, a_name):
        pass

    @abstractmethod
    def detect_for_colab(self, frame):
        pass
