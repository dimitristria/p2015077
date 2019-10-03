"""
Περιέχει όλα τα πιθανά σφάλματα της εφαρμογής καθώς και βοηθητικές συναρτήσεις.
"""

class NotEnoughImages(Exception):
    def __init__(self, msg: str):
        self.msg = "{}".format(msg)


class NotEnoughMatchesError(Exception):
    def __init__(self, num_of_matches: int):
        self.msg = "{} - not enough matches!".format(num_of_matches)


class HomographyMatrixNotFoundError(Exception):
    def __init__(self):
        self.msg = "Homography matrix not found!"


class StitcherError(Exception):
    def __init__(self, msg=None):
        self.msg = "{}".format(msg)


class IntervalsNotFoundError(Exception):
    pass


class OutOfMemoryError(Exception):
    pass


CV2_ERRORS = {
    "cv::OutOfMemoryError": OutOfMemoryError()
}


def is_cv2_error(msg: str):
    """Αναγώριση error της βιβλιοθήκης OpenCV από το μύνημα τους"""
    for error_msg in CV2_ERRORS.keys():
        if error_msg in msg:
            return True
    return False
