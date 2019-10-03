"""
Φίλτρα τα οποία μπορούν να χρησιμοοπιηθούν σε εικόνες τύπου numpy.ndarray.
"""

import numpy
import cv2
import my_image
import statistics
import error


def is_zero_pixel(pixel: list):
    """
    Έλεγχος αν το τρέxον εικονοστοιχείο είναι μηδενικής έντασης φωτεινότητας
    """
    flag = 0
    for ch in range(len(pixel)):
        if pixel[ch] == 0:
            flag = flag + 1
    if flag == len(pixel):
        return True
    return False


def num_of_zero_pixels_in_column(img, r1, r2, step):
    counter = 0
    for c in range(r1, r2, 1):
        if is_zero_pixel(img[step, c, :]):
            counter = counter + 1
    return counter


def num_of_zero_pixels_in_row(img, r1, r2, step):
    counter = 0
    for c in range(r1, r2, 1):
        if is_zero_pixel(img[c, step, :]):
            counter = counter + 1
    return counter


def find_best_interval(img: numpy.ndarray, interval1: tuple, interval2: tuple, orientation: str, point: str, max_error: int):
    # Ακραίες περιπτώσεις (interval1[1] - interval1[0] == 0)
    # Υπόλοιπες περιπτώσεις (interval1[1] - interval1[0] == 1)

    step_counter = 0
    step = interval1[0]
    prev_step = step
    prev_step_error = interval1[1]
    x = list()
    while True:
        if step > interval1[1]:
            step = interval1[1]

        """Υπολόγησε το step_error για τη γραμμή/στήλη του τρέχον step"""
        step_error = 0
        if orientation == "horizontal":
            step_error = num_of_zero_pixels_in_column(img, interval2[0], interval2[1]+1, step)
        elif orientation == "vertically":
            step_error = num_of_zero_pixels_in_row(img, interval2[0], interval2[1]+1, step)

        """Σύγριση του τρέχοντος error με το προηγούμενο ανάλογα με το σημείο αναζήτησης"""
        # Για τις συντεταγμένες του πρώτου σημείου (a)
        if point == "a":
            if step_error <= max_error < prev_step_error:
                x = [prev_step, step]
                break
        # Για τις συντεταγμένες του δεύτερου σημείου (b)
        elif point == "b":
            if prev_step_error <= max_error < step_error:
                x = [prev_step, step]
                break

        # Διακοπή αναζήτησης όταν το step είναι ίσο με το rows ή columns
        if step == interval1[1]:
            break

        # Ανανέωσε τις μεταβλητές
        prev_step_error = step_error
        prev_step = step
        step = pow(2, step_counter) + interval1[0]
        step_counter += 1
    
    if not x:
        return interval1[1]
    elif (x[1] - x[0]) in (0, 1):
        return x[1]
    
    return find_best_interval(img, x, interval2, orientation, point, max_error)


def num_of_zero_pixels(img: numpy.ndarray):
    """
    Μέτρηση των εικονοστοιχείων που έχουν μηδενικές τιμές 
    σε όλα τα κανάλια τους και επιστρέφει το τελικό αποτέλεσμα
    """
    rows, columns = img.shape[0:2]
    all_pixels = rows*columns
    non_zero_pixels = cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return all_pixels - non_zero_pixels


def automatic_cropping(img: numpy.ndarray):
    """
    Αυτόματη περικοπή κενών περιθωρίων εικόνας εισόδου.
    """
    rows, columns = img.shape[0:2]
    or_zero_pixels = num_of_zero_pixels(img)
    # ax, ay = 0, rows-1
    # bx, by = 0, columns-1
    c = 10
    stop_counter = 0
    
    while True:
        # Ποσοστό
        p = 1/c
        stop_counter += 1

        # Εύρεση χρήσιμου εύρους εικόνας οριζόντια
        max_error_rows = int(rows * p)
        ax = find_best_interval(img, (0, rows-1), (0, columns-1), "horizontal", "a", max_error_rows)
        bx = find_best_interval(img, (ax, rows-1), (0, columns-1), "horizontal", "b", max_error_rows)

        # Εύρεση χρήσιμου εύρους εικόνας κάθετα
        max_error_columns = int(columns * p)
        ay = find_best_interval(img, (0, columns-1), (ax, bx), "vertically", "a", max_error_columns)
        by = find_best_interval(img, (ay, columns-1), (ax, bx), "vertically", "b", max_error_columns)

        zero_pixels = num_of_zero_pixels(img[ax:bx, ay:by])

        if zero_pixels < or_zero_pixels / p * 100:
            # Επέστρεψε την εικόνα αποκομένη βάση των προηγούμενων σημείων
            return img[ax:bx, ay:by]

            # Draw clipping window
            # cv2.rectangle(img,(ay,ax),(by, bx),(0,255,0),3)
            # return img
        elif stop_counter:
            break
        else:
            c += 10
    return img


def create_histogram(img: numpy.ndarray, channels=None):
    """
    Δημιουργία ιστογράμματος για την δοθείσα εικόνα τύπου numpy.ndarray.\n
    Αν channels==None, τότε λαμβάνονται υπ' όψη όλα τα κανάλια της.\n
    Αν channels==[1,2], τότε θα ληφθούν ενέργειες μόνο για αυτά
    """

    # Εαν δεν δωθεί είσοδος στη μεταβλητή channels
    # τότε χρησιμοποιούνται όλα τα αρχικά κανάλια χρωμάτων
    if len(img.shape) != 3:
        channels = ["gray"]
    elif not valid_input_channels(img, channels):
        channels = list(ch for ch in range(img.shape[2]))

    rows, columns = img.shape[:2]
    min_inten = numpy.iinfo(img.dtype).min      # ελάχιστης του τύπου δεδομένων
    max_inten = numpy.iinfo(img.dtype).max      # μέγιστης του τύπου δεδομένων

    # Αρχικοποίηση του tuple h που περιέχει dictionaries
    # για τα ιστογράμματα κάθε καναλιού χρωμάτων
    h = tuple(dict() for _ in range(len(channels)))
    for d in h:                                     # Για κάθε κανάλι χρωμάτων
        for g in range(min_inten, max_inten+1, 1):  # Για κάθε ένταση pixel
            d[g] = 0

    # Μέτρηση εμφανίσεων κάθε φωτεινότητας σε κάθε pixel για κάθε κανάλι
    for r in range(rows):
        for c in range(columns):
            for ch in range(len(channels)):
                if len(img.shape) != 3:
                    h[ch][img[r, c]] += 1                   # gray
                else:
                    h[ch][img[r, c, channels[ch]]] += 1     # bgr
    return h


def valid_input_channels(img: numpy.ndarray, channels: list):
    if channels is None:
        return False
    for val in channels:
        if not isinstance(val, int):                # Έλεγχος τύπου δεδομένων
            return False
        if len(img.shape) == 3:                     # bgr
            if val not in range(0, img.shape[2]):   # Έλεγχος αρίθμησης
                return False
    return True


def detect_defective_channels(h: list, channels: list):
    """
    Ανίχνευση ελατωματικών καναλίων με τη χρήση του μέτρου κύρτωσης b2.
    """

    data = tuple([h[c][k] for k in h[c]] for c in range(len(h)))
    target_channels = tuple()
    threshold = 3
    for ch in range(len(channels)):
        if 0.0 < statistics.b2(data[ch]) > threshold:
            target_channels += (channels[ch],)
    if not target_channels:
        return None
    return target_channels


def equalize_histograms(img: numpy.ndarray, automatic=False, channels=None):
    # Έλεγχος εισόδου: channels
    if len(img.shape) != 3:
        channels = [0]
    elif automatic or not valid_input_channels(img, channels):
        channels = tuple(ch for ch in range(img.shape[2]))

    # Πληροφορίες για τον τύπο δεδομένων που χρησιμοποιεί η εικόνα
    dtype_info = numpy.iinfo(img.dtype)
    # Εύρεση της ελάχιστης του τύπου δεδομένων της εικόνας εισόδου
    min_inten = dtype_info.min
    # Εύρεση της μέγιστης του τύπου δεδομένων της εικόνας εισόδου
    max_inten = dtype_info.max

    # Αρχικοποίηση πίνακα με τις σχετικές αθροιστηκές ποιθανότητες
    # για κάθε ένταση φωτεινότητας σε κάθε κανάλι χρωμάτων της εικόνας
    P = tuple(dict() for _ in range(len(channels)))
    # Για κάθε κανάλι χρωμάτων
    for d in P:
        # Για κάθε ένταση εικονοστοιχείων
        for g in range(min_inten, max_inten+1, 1):
            d[g] = 0

    # εξαγωγή ιστογράμματος για κάθε κανάλι της εικόνας
    h = create_histogram(img, channels)

    # πρόσθετος κώδικας της αυτόματης λειτουργίας
    # Εύρεση πιθανόν ελλατωματικών
    if automatic:
        channels = detect_defective_channels(h, channels)
        if channels is None:
            return img

    # Διαστάσεις εικόνας εισόδου
    rows, columns = img.shape[:2]

    # Υπολόγισμός αθροιστικής πιθανότητας
    # Για κάθε κανάλι χρωμάτων
    for ch in range(len(channels)):
        # Για κάθε ένταση εικονοστοιχείων
        for g1 in range(min_inten, max_inten+1, 1):
            S = 0
            # Για κάθε ένταση από min_inten έως g1
            for g2 in range(min_inten, g1, 1):
                S += h[ch][g2]
            P[ch][g1] = int((S * (max_inten+1)) / (rows * columns))

    # Εφαρμογή νέων τιμών έντασης σε αντίγραφο της αρχικής εικόνας
    new_img = numpy.ndarray(img.shape, numpy.uint8)
    for r in range(rows):
        for c in range(columns):
            for ch in range(len(channels)):
                if len(img.shape) != 3:
                    new_img.itemset((r, c), P[ch][img[r, c]])
                else:
                    color = channels[ch]
                    new_img.itemset((r, c, color), P[ch][img[r, c, color]])
    return new_img


# def fix_aspect_ratio(img: numpy.ndarray, ideal_height=768, ideal_width=1366):
#     height, width = img.shape[:2]
#     aspect = width / float(height)

#     ideal_aspect = ideal_width / float(ideal_height)

#     new_width = width
#     new_height = height

#     if aspect > ideal_aspect:
#         new_width = int(ideal_aspect * height)
#     else:
#         new_height = int(ideal_aspect * width)
    
#     # Αν παρατηρηθεί διαφορά τότε κάνε την αλλαγή στην κλίμακα
#     if new_height != height or new_width != width:
#         return cv2.resize(img, (new_height, new_width))
    
#     return img


def negative(img: numpy.ndarray):
    tmp = img.ravel()
    a = min(tmp)
    b = max(tmp)
    new_img = (a+b) - img
    return new_img


def filter_image(img: numpy.ndarray, autocrop=False, eqhist=False, fixaspratio=False):
    """
    Εφαρμογή φίλτρων για τη δοθείσα εικόνα.
    """

    # Επιλογή χρήσης συνάρτησης για την αυτόματη αφαίρεση περιττών ορίων (κόψιμo) της εικόνας
    if autocrop:
        img = automatic_cropping(img)
    
    # Επιλογή χρήσης συνάρτησης για την αυτόματη εξισορόπηση ιστογράμματος
    if eqhist:
        # cv2.equalizeHist
        # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        # img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        # img[:,:,2] = cv2.equalizeHist(img[:,:,2])

        # filters.equalize_histograms
        img = equalize_histograms(img)

    # if fixaspratio:
    #     img = fix_aspect_ratio(img)
    
    # Επιστροφή επεξεργασμένης εικόνας
    return img
