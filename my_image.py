"""
Κλάσση MyImage για την οργάνωση, αποθήκευση και 
διαχείρηση εικόνων και πληροφοριών αυτών στο πρόγραμμα
"""

import numpy


class MyImage:
    def __init__(self, name: str, img: numpy.ndarray):
        self.name = str(name).strip()
        self.image = img
        self.keypoints = []
        self.descriptors = []
        self.candidate_imgs = {}
        self.group = -1

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, img: numpy.ndarray):
        self.__image = img

    @property
    def keypoints(self):
        return self.__keypoints

    @keypoints.setter
    def keypoints(self, kps: numpy.ndarray):
        self.__keypoints = kps

    @property
    def descriptors(self):
        return self.__descriptors

    @descriptors.setter
    def descriptors(self, descs: numpy.ndarray):
        self.__descriptors = descs

    @property
    def candidate_imgs(self):
        return self.__candidate_imgs

    @candidate_imgs.setter
    def candidate_imgs(self, c_imgs: []):
        self.__candidate_imgs = c_imgs
