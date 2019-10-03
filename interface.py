"""
Περιέχει συναρτήσεις με γενικού σκοπού λειτουργικότητα.
"""

import os
import hashlib
import numpy
import cv2
import my_image


# Κενή εικόνα
blank_image = numpy.ndarray((100, 100, 1), numpy.uint8)


def my_print(msg: str, msg_type_code=0):
    """
    Τροποποιημένη συναρτηση print για debugging σκοπούς
    Διαθέσημα μυνήματα: message (0), warning (1), error (2)
    """
    all_msgs_types = ["message", "-warning", "--error"]
    if msg_type_code not in range(len(all_msgs_types)):
        msg_type_code = 0
    print(r"{}: {}".format(all_msgs_types[msg_type_code], msg))


def my_range(start, end, step):
    """
    Χρήση: προσδιορισμός ορίων σε for loops.\n
    Αντί για: range(start, end+1, step).
    """
    while start <= end:
        yield start
        start += step


def sort_dict_by_value_len(dct: dict, rev: bool):
    """
    Ταξινόμηση ενός dictionary βάση του μεγέθους (μήκος λίστας) των τιμών του
    """
    dict_len = {key: len(value) for key, value in dct.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=rev)
    sorted_dict = {item[0]: dct[item[0]] for item in sorted_key_list}
    return sorted_dict


def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_file_name(filename):
    """
    Ανάκτηση ονόματος αρχείου χωρίς την κατάληξη του
    """
    return filename.rsplit('.', 1)[0]


def get_file_format(filename):
    """
    Ανάκτηση ονόματος κατάληξης αρχείου
    """
    return filename.rsplit('.', 1)[1].lower()


def path_leaf(path):
    """
    Ανάκτηση τελευταίου στοιχείου βάση μια διαδρομής
    """
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


def my_imshow(imgs: list):
    """
    Προβολή παραθύρου με την εικόνα εισόδου
    """
    if not imgs:
        return
    for c, img in enumerate(imgs):
        if img is None:
            continue
        elif isinstance(img, my_image.MyImage):
            cv2.imshow(img.name, img.image)
        else:
            cv2.imshow("frame" + str(c), img)
    cv2.waitKey()
    cv2.destroyAllWindows()
