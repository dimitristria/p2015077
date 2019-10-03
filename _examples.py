"""
Αρχείο χωρισμένο ανά ακολουθίες σειρών που περιέχουν επανελειμένα το σύμβολο "#".
Με την αφαίρεση των σχολίων στον κώδικα ενδιάμεσα των ορίων, εκτελούνται διαφορετικά παραδείγματα.
"""

#################################################################

# import cv2
# import stitcher
# import my_image

# st = stitcher.Stitcher()
# img = my_image.MyImage(name="testtest", img=cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\1.png"))
# st.extract_features([img])
# print(img.keypoints)
# print(dir(img.keypoints[0]))

#################################################################

# import cv2
# import stitcher

# dirs = tuple()
# dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\111.jpg",)
# dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\222.jpg",)
# # dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\33.png",)

# st = stitcher.Stitcher()
# new_imgs = st.run(paths=dirs, method=stitcher.AUTOMATICALLY)
# if new_imgs:
#     new_img = new_imgs[0][0]
#     cv2.imshow("final image", new_img)
#     cv2.imwrite("C:\\Users\\jimo4\\Desktop\\result.png", new_img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#################################################################

import cv2
import stitcher
import filters

dirs = tuple()
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\1.png",)
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\2.png",)
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\3.png",)
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\11.png",)
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\22.png",)
dirs += ("C:\\Users\\jimo4\\Downloads\\_input\\33.png",)

st = stitcher.Stitcher()
new_img = st.run(dirs, stitcher.MANUALLY)
if len(new_img) > 0:
    cv2.imshow("test", new_img[0][0])
    # cv2.imwrite("C:\\Users\\jimo4\\Desktop\\result.png", new_img[0][0])
    cv2.waitKey()
    cv2.destroyAllWindows()

#################################################################

# import cv2
# import filters

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\1.png")
# h = filters.create_histogram(img)
# print(h)

#################################################################

# import numpy
# import cv2
# import filters

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\1.png")
# new_img = filters.automatic_cropping(img)
# cv2.imshow("before", img)
# cv2.imshow("after", new_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#################################################################

# import cv2
# import filters

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\1.png")
# new_img = filters.equalize_histograms(img)
# cv2.imshow("test", new_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#################################################################

# import cv2
# import filters

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\11.png")
# new_img = filters.fix_aspect_ratio(img)
# cv2.imshow("before", img)
# print("img_shape:", img.shape[:2])
# cv2.imshow("after", new_img)
# print("new_img_shape:", new_img.shape[:2])
# cv2.waitKey()
# cv2.destroyAllWindows()

#################################################################

# from website import db, User, Image, StitchedImage

# db.drop_all()
# db.create_all()

# user = User()
# user.username = "jkl"
# db.session.add(user)
# db.session.commit()

# img1 = Image()
# img1.id = "01234"
# img1.user_id = user.id
# db.session.add(img1)
# db.session.commit()

# img2 = Image()
# img2.id = "56789"
# img2.user_id = user.id
# db.session.add(img2)
# db.session.commit()

# new_img = Image()
# new_img.id = "0123456789"
# new_img.user_id = user.id
# db.session.add(new_img)
# db.session.commit()

# st_img = StitchedImage()
# st_img.image_id = "123"
# st_img.user_id = user.id
# st_img.contains.append(img1)
# st_img.contains.append(img2)
# db.session.add(st_img)
# db.session.commit()

#################################################################

# import cv2

# src1 = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\1.png")
# src2 = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\2.png")

# alpha = 0.5
# beta = (1.0 - alpha)
# dst = cv2.addWeighted(src1, beta, src2, beta, 0.0)

# cv2.imshow("dst", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

#################################################################

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import filters

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\Lenna.png")
# # cv2.imshow("original img", img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256]);
# plt.show()

# img = filters.equalize_histograms(img, automatic=False)
# cv2.imwrite("C:\\Users\\jimo4\\Downloads\\_input\\he_output_lenna.png", img)
# # cv2.imshow("final img", img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256]);
# plt.show()

#################################################################

# import cv2

# img = cv2.imread("C:\\Users\\jimo4\\Downloads\\_input\\11_noise.png")
# median = cv2.medianBlur(img,5)
# # cv2.imwrite("C:\\Users\\jimo4\\Downloads\\_input\\11_noise_removed.png", median)
# cv2.imshow("final img", median)
# cv2.waitKey()
# cv2.destroyAllWindows()

#################################################################

# print("current example")

#################################################################
