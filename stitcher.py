"""
Stitcher
"""

import os
import os.path
import numpy
import cv2
import my_image
import filters
import interface
import error


AUTOMATICALLY = "automatically"
MANUALLY = "manually"
ALLOWED_METHODS = (AUTOMATICALLY, MANUALLY)


class Stitcher:
    """
    Βασικές λειτουργίες: automatically() και manually().
    """

    # Δημιουργία αντικειμένου KAZE για την εξαγωγή χαρακτηριστικών
    kaze = cv2.KAZE_create()

    # Παράμετροι και δημιουργία αντικειμένου cv2.FlannBasedMatcher
    # για το ταίριασμα μεταξύ των εικόνων
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    MIN_MATCH_COUNT = 10

    def __init__(self):
        self.imgs = list()
        self.CANDIDATE_IMAGES_LIMIT = -1
        self.NUM_OF_GROUPS = -1
        self.grouped_imgs = dict()

    def read_imgs(self, paths: tuple):
        """
        Δημιουργία αντικειμένων MyImage.\n
        paths: είναι μια λίστα που μπορεί να περιέχει συμβολοσειρές ονομάτων αρχεία και φακέλους αυτών.
        """
        imgs = list()
        for f in paths:
            if os.path.isfile(f):
                img = cv2.imread(f)
                imgs.append(my_image.MyImage(f, img))
            elif os.path.isdir(f):
                for f_file in os.listdir(f):
                    img = cv2.imread(os.path.join(f, f_file))
                    imgs.append(my_image.MyImage(f_file, img))
            else:
                interface.my_print("{} --> skipped".format(f), 1)
        return imgs
    
    @staticmethod
    def check_given_images_instances(imgs: list):
        for img in imgs:
            if not isinstance(img, my_image.MyImage):
                return True
        return False
    
    def check_given_images(self, imgs: list):
        if len(imgs) <= 1 or self.check_given_images_instances(imgs):
            raise error.StitcherError("Invalid input images or their types")

    @staticmethod
    def check_given_method(method: str):
        if method not in ALLOWED_METHODS:
            raise error.StitcherError("invalid method")

    def extract_features(self, imgs: list):
        """Εξαγωγή χαρακτηριστικών σημείων και περιγραφών αυτών για τις εικόνες εισόδου"""
        for img in imgs:
            if not img.keypoints and not img.descriptors:
                gray_image = cv2.cvtColor(img.image, cv2.COLOR_BGR2GRAY)
                try:
                    kps, descs = self.kaze.detectAndCompute(gray_image, None)
                except cv2.error as e:
                    raise error.StitcherError("CV2_ERROR")
                img.keypoints, img.descriptors = kps, descs
    
    def match_features(self, desc1: numpy.ndarray, desc2: numpy.ndarray):
        """
        Επιστρέφει μια λίστα με μέγεθος το οποίο προκύπτει από τα match που προέκυψαν βάση των δυο descriptors.\n
        Η λίστα περιέχει λίστες μέγιστου μεγέθους όσο με το 'k' (τα matches στη γειτονιά του σημείου).
        """
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        good_matches = tuple()

        if matches:
            # Διάλεξε τo καλύτερο γειτονικό σημείο από τα k υποψήφια
            # κάθε υπολίστας στη λίστα matches βάση κάποιου φίλτρου
            # γνωστό ως και το "Lowe's ratio test".
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches += (m,)

        return good_matches
    
    def find_m_candidate_images_for_each_image(self, targets: list, candidates: list):
        """
        Για κάθε εικόνα, εκτός από τον εαυτό της, βρίσκονται κοινά χαρακτηριστικά με τις υπόλοιπες.
        """

        for img_i in targets:
            if img_i.candidate_imgs:
                continue
            
            cand_imgs_dct = dict()
            for img_j in candidates:
                if img_j != img_i:
                    desc1, desc2 = img_i.descriptors, img_j.descriptors
                    matches = self.match_features(desc1, desc2)
                    if not matches or len(matches) < self.MIN_MATCH_COUNT:
                        continue
                    cand_imgs_dct[img_j] = matches

            if not cand_imgs_dct:
                continue

            # Ταξινόμηση του λεξικού cur_matches_dct βάση
            # το πλήθος των ταιριασμάτων κατά αύξουσα σειρά
            cand_imgs_dct = interface.sort_dict_by_value_len(cand_imgs_dct, True)

            # Περιορισμός υποψήφιων εικόνων βάση ενός κατωφλιού
            # Αποθήκευση των υποψήφιων εικόνων στην ανάλογη
            # λίστα της τρέχουσας εικόνας
            if len(cand_imgs_dct) > self.CANDIDATE_IMAGES_LIMIT:
                for k in list(cand_imgs_dct)[0:self.CANDIDATE_IMAGES_LIMIT]:
                    img_i.candidate_imgs[k] = cand_imgs_dct[k]
            else:
                img_i.candidate_imgs = cand_imgs_dct
        
        if len(targets) <= 1:
            return targets[0].candidate_imgs
        return None
    
    @staticmethod
    def remove_noise_or_unused_imgs(imgs: list):
        """
        Οι εικόνες που δεν έχουν άλλες για να ταιριάξουν απορίπτονται και δεν συμμετέχουν στις ακόλουθες διαδικασίες.
        """
        for img in imgs:
            if not img.candidate_imgs:
                imgs.remove(img)
        return True

    @staticmethod
    def assign_group_on_imgs(imgs: list):
        """
        Ομαδοποίηση των εικόνων που πέρασαν σε αυτή τη φάση βάση αυτών που ταιριάζουν.
        """
        groups_counter = 0
        for img_i in imgs:
            if img_i.group == -1:
                groups_counter += 1
                img_i.group = groups_counter
            for img_j in img_i.candidate_imgs.keys():
                img_j.group = img_i.group
        return groups_counter

    def create_grouped_imgs_dct(self, imgs):
        """
        Διαχωρισμός των εικόνων βάση των ομάδων που ανοίκουν.
        """
        self.grouped_imgs = {k: list() for k in range(1, self.NUM_OF_GROUPS+1, 1)}
        for img in imgs:
            self.grouped_imgs[img.group].append(img)
        return True

    def get_homography(self, kp1: numpy.ndarray, kp2: numpy.ndarray, matches: numpy.ndarray):
        """
        Εύρεση ομοιογραφίας μεταξύ δύο εικόνων βάση των χαρακτηριστικών τους σημείων.
        """

        # έλεγχος για ύπαρξη επαρκών matches βάση κάποιου κατωφλιού
        if len(matches) < self.MIN_MATCH_COUNT:
            raise error.NotEnoughMatchesError(len(matches))

        # Αποθηκεύουμε τις συντεταγμένες των σημείων της λίστας matches
        # src_pts: περιέχει τα σημεία της πρώτης εικόνας
        # dst_pts: περιέχει τα σημεία της δεύτερης εικόνας
        src_pts, dst_pts = tuple(), tuple()
        for m in matches:
            try:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
            except IndexError:
                continue
            else:
                src_pts += (pt1,)
                dst_pts += (pt2,)
        src_pts = numpy.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = numpy.float32(dst_pts).reshape(-1, 1, 2)

        # Εύρεση του καλύτερου μετασχηματισμού μεταξύ δύο δειγμάτων σημείων.
        # Για καλύτερη αξιοπιστία των αποτελεσμάτων γίνεται χρήση του αλγορίθμου
        # RANSAC για την εξάλλειψη εκτρεπώμενων τιμών στα παραπάνω δεγματα.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        if M is None or mask is None:
            raise error.HomographyMatrixNotFoundError()
        
        # matchesMask = mask.ravel().tolist()
        # num_of_inliers, num_of_outliers = 0, 0
        # for c, val in enumarate(matchesMask):
        #     print(c, val)
            # if val == 0:
            #     num_of_inliers += 1
            # elif val == 1:
            #     num_of_outliers += 1

        return M

    @staticmethod
    def get_extreme_points(img: numpy.ndarray, dtype: numpy.ndarray.dtype):
        rows, columns = img.shape[0:2]
        extreme_points = numpy.array([[0, 0], [0, rows], [columns, rows], [columns, 0]])
        if dtype:
            return extreme_points.astype(dtype)
        return extreme_points

    def connect_images(self, img1: numpy.ndarray, img2: numpy.ndarray, M: numpy.ndarray):
        """
        Σύνδεση των δύο εικόνων εισόδου με τη χρήση του πίνακα μετασχηματισμού M.
        """

        # Εύρεση ακραίων σημείων των εικόνων εισόδου
        # (μετατροπή σε "float" λόγω ακόλουθων μετασχηματισμών)
        pts1_temp = self.get_extreme_points(img1, float).reshape(-1, 1, 2)
        pts2 = self.get_extreme_points(img2, float).reshape(-1, 1, 2)

        # Αλλαγή των ακραίων σημείων της δεύτερης εικόνας βάση του πίνακα
        # προοπτικής προβολής M που προέκυψε από την εύρεση της ομοιογραφίας.
        # Χρήση: στον υπολογισμό του μεγέθους της σύνθετης εικόνας
        pts1 = cv2.perspectiveTransform(pts1_temp, M)
        all_pts = numpy.concatenate((pts1, pts2), axis=0)
        ax, ay = numpy.int32(all_pts.min(axis=0).ravel() - 0.5)
        bx, by = numpy.int32(all_pts.max(axis=0).ravel() + 0.5)
        
        # Υπολογιμός πίνακα μεταφοράς της για την τελική εικόνα
        tx, ty = -ax, -ay
        translate_array = numpy.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        new_img_shape = (bx - ax, by - ay)
        new_img = cv2.warpPerspective(img1, translate_array.dot(M), new_img_shape)

        # Προσθήκη της δεύτερης εικόνας στην νέα που ποέκυψε προηγουμένως
        rows2, columns2 = img2.shape[0:2]
        for r in range(ty, (ty + rows2), 1):
            for c in range(tx, (tx + columns2), 1):
                new_img[r, c] = img2[r - ty, c - tx]
        
        return new_img
    
    def stitching(self, imgs: list):
        """
        Συρραφή εικόνων που ανοίκουν στην ίδια ομάδα.
        """

        # Έλεγχος εισόδου εικόνων
        try:
            self.check_given_images(self.imgs)
        except error.StitcherError as e:
            interface.my_print(e.msg, 2)
            return tuple()

        # Αποθήκευση ονομάτων εικόνων
        src_imgs_filenames = tuple()
        for img in imgs:
            src_imgs_filenames += (img.name,)

        c = 0
        while c < len(imgs):
            # Ανάθεση της τρέχουσας εικόνας στην μεταβλητή "img"
            img = imgs[c]

            for cand_img, matches in img.candidate_imgs.items():
                if cand_img not in imgs:
                    continue

                # Ανάθεση πινάκων σε μεταβλητές για καλύερη ευαναγνωσιμότητα
                kp1, kp2 = img.keypoints, cand_img.keypoints
 
                try:
                    # Εύρεση μετασχηματισμού προβολής μεταξύ των σημείων ενδιαφέροντος
                    M = self.get_homography(kp1, kp2, matches)
                except error.NotEnoughMatchesError as e:
                    # continue
                    raise error.StitcherError(e.msg)
                except error.HomographyMatrixNotFoundError as e:
                    # continue
                    raise error.StitcherError(e.msg)

                # Ένωση των δύο εικόνων
                new_img = self.connect_images(img.image, cand_img.image, M)

                # Αφαίρεση πηγαίων εικόνων από τη λίστα με τις τρέχουσες
                imgs.remove(img)
                imgs.remove(cand_img)

                # Αρχικοποίηση της νέας εικόνας
                new_img_name = str(c)
                new_img = my_image.MyImage(new_img_name, new_img)
                # Εύρεση χαρακτηριστικών αυτής
                self.extract_features([new_img])
                new_img.group = img.group
                # Σύγκρισή της με τις υπόλοιπές
                # new_img.candidate_imgs = self.find_m_candidate_images_for_each_image([new_img], imgs)

                # Προσθήκη της νέας εικόνας στην λίστα με τις 
                # τρέχουσες και διακοπή αναζήτησης των υποψήφιων 
                # εικόνων της τρέχουσας και συνέχεια στην επόμενη
                imgs.append(new_img)
                break
            c += 1
        
        # Αυτόματη αποκοπή ανώφελων ορίων εικόνας από τους μετασχηματισμούς που προηγήθηκαν
        # imgs[0].image = filters.automatic_cropping(imgs[0].image)

        # imgs[0].image = filters.equalize_hist(imgs[0].image)

        # Επιστροφή της παραγόμενης καθώς και των ονομάτων των πηγαίων εικόνων της
        return (imgs[0].image, src_imgs_filenames)

    def grouped_stitching(self):
        """
        Συραφφή των εικόνων κάθε υποομάδας.
        """
        results = tuple()
        for imgs in self.grouped_imgs.values():
            try:
                item = self.stitching(imgs)
            except error.StitcherError as e:
                interface.my_print("Group: " + imgs[0].group + " -> " + e.msg, 1)
                continue
            else:
                if item:
                    results += (item, )
        return results
    
    def automatically(self, imgs: list):
        """
        Αυτοματοποιημένη μέθοδος για τη συρραφή των εικόνων, 
        προσεγγίζοντας τη μέθοδο των M. Brown and D. G. Lowe (2003)
        """

        print("extract_features...", end=" ")
        
        # Εξαγωγή χαρακτηριστικών από τις εικόνες εισόδου
        self.extract_features(imgs)

        print("done!")

        # Όριο υποψήφιων εικόνων που θα περιέχει κάθε εικόνα
        # 2/3 (66.66%): σχεδόν βέβαιο ότι όλες οι εικόνες που
        # ταιριάζουν πραγματικά θα ανοίκουν στην ίδια ομάδα
        self.CANDIDATE_IMAGES_LIMIT = int(len(imgs)*(2/3))

        print("find_m_candidate_images_for_each_image...", end=" ")

        self.find_m_candidate_images_for_each_image(imgs, imgs)

        print("done!")
        print("create_grouped_imgs_dct...", end=" ")

        self.remove_noise_or_unused_imgs(imgs)
        self.NUM_OF_GROUPS = self.assign_group_on_imgs(imgs)
        self.create_grouped_imgs_dct(imgs)

        print("done!")
        print("grouped_stitching...", end=" ")

        results = self.grouped_stitching()

        print("done!")
        
        # Επιστροφή συνόλων εικόνων που συρράφτηκαν
        return results
    
    def manually(self, imgs: list):
        """
        Μη αυτοματοποιημένη μέθοδος για τη συρραφή των εικόνων
        """

        print("extract_features...", end=" ")

        # Εξαγωγή χαρακτηριστικών από τις εικόνες εισόδου
        self.extract_features(imgs)

        print("done!")
        print("other...", end=" ")

        # [print(interface.path_leaf(img.name), len(img.keypoints)) for img in imgs]

        src_imgs_filenames = tuple()

        c = 0
        while c < len(imgs):
            try:
                img1, img2 = imgs[0], imgs[1]
                desc1, desc2 = img1.descriptors, img2.descriptors
                matches = self.match_features(desc1, desc2)
                kp1, kp2 = img1.keypoints, img2.keypoints

                try:
                    M = self.get_homography(kp1, kp2, matches)
                except Exception as e:
                    interface.my_print("Unknown homography", 1)
                    # Αφαίρεσε την εικόνα με τα περισσότερα σημεία ενδιαφέροντος
                    # υποψία και υπόθεση: μάλλον εικόνα με επαναλμβανόμενα στοιχεία
                    if len(kp1) > len(kp2):
                        imgs.remove(img1)
                    else:
                        imgs.remove(img2)
                    continue
                
                new_img = self.connect_images(img1.image, img2.image, M)
                src_imgs_filenames += (interface.path_leaf(img1.name),)
                src_imgs_filenames += (interface.path_leaf(img2.name),)
                # new_img = filters.automatic_cropping(new_img)
                
                imgs[0].image = new_img
                
                imgs.remove(imgs[1])
                c += 1
            except:
                break
        
        print("done!")
        
        results = ((imgs[0].image, src_imgs_filenames),)
        return results
    
    def run(self, paths: tuple, method="automatically"):
        # Ανάκτηση εικόνων βάση των δοθέν path
        self.imgs = self.read_imgs(paths)

        # Έλεγχος πλήθους εικόνων και μεθόδου
        try:
            self.check_given_images(self.imgs)
            self.check_given_method(method)
        except error.StitcherError as e:
            interface.my_print(e.msg, 2)
            # Επιστροφή κενού αποτελέσματος
            return tuple()
        
        new_imgs = tuple()
        try:
            if method == AUTOMATICALLY:
                new_imgs = self.automatically(self.imgs)
            elif method == MANUALLY:
                new_imgs = self.manually(self.imgs)
        except error.StitcherError:
            interface.my_print("Something went wrong!!!", 2)
        except Exception as e:
            interface.my_print(e, 2)
        
        return new_imgs
    
    # @staticmethod
    # def find_image_intervals(img: numpy.ndarray, flag: str, num: int):
    #     """
    #     Δημιουργία τμημάτων στην εικόνα (οριζόντια ή κάθετα) και επιστροφή συντεταγμένων αυτών.
    #     """

    #     if flag == "vertical":
    #         limit = img.shape[0]
    #     elif flag == "horizontal":
    #         limit = img.shape[1]
    #     else:
    #         limit = img.shape[1]

    #     if num <= 1 or num >= limit:
    #         num = 3
    #     split = int(limit / num)
    #     intervals = tuple()
    #     prev_val = 0
    #     for c in range(0, num, 1):
    #         if c == 0:
    #             intervals += (0, split)
    #         elif c == num - 1:
    #             intervals += (prev_val+1, limit-1)
    #         else:
    #             intervals += (prev_val+1, prev_val + split)
    #         prev_val = intervals[c][1]
    #     return intervals
