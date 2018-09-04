import os
import cv2
import numpy as np
import pdf2image
import imutils

DATASET_PDF_FOLDER = 'dataset_pdf'
DATASET_IMAGES_FOLDER = 'dataset_images'
hashcode_length = 37


def extract_logos(dataset_images, debug=False):
    for filename in os.listdir(dataset_images):
        if filename.endswith(".jpg"):
            os.rename(os.path.join(dataset_images, filename)
                      , os.path.join(dataset_images, filename[hashcode_length:]))

    print("Total Pages ", len(os.listdir(dataset_images)))
    page = 1
    for filename in os.listdir(dataset_images):
        print("Processing Page ", page)
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(dataset_images, filename))

            startpoint = 140
            height, width = image.shape[:2]
            endpoint = height
            image = image[startpoint:endpoint, 0:0 + width]
            backupImage = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)  # 11
            # //TODO 11 FOR OFFLINE MAY NEED TO TUNE TO 5 FOR ONLINE
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV |
                                   cv2.THRESH_OTSU)[1]

            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

            # find external contours in the thresholded image and allocate memory
            # for the convex hull image
            cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            hullImage = np.zeros(gray.shape[:2], dtype="uint8")

            # loop over the contours
            heights = []
            y_axis = []
            for (i, c) in enumerate(cnts):
                # compute the area of the contour along with the bounding box
                # to compute the aspect ratio
                area = cv2.contourArea(c)
                (x, y, w, h) = cv2.boundingRect(c)

                # compute the aspect ratio of the contour, which is simply the width
                # divided by the height of the bounding box
                aspectRatio = w / float(h)

                # use the area of the contour and the bounding box area to compute
                # the extent
                extent = area / float(w * h)
                # cv2.imshow("thresh", thresh[y:y+h, x:x+w])
                # cv2.waitKey(0)

                try:
                    # compute the convex hull of the contour, then use the area of the
                    # original contour and the area of the convex hull to compute the
                    # solidity
                    hull = cv2.convexHull(c)
                    hullArea = cv2.contourArea(hull)
                    solidity = area / float(hullArea)
                    heights.append(h)
                    y_axis.append(y)

                    # visualize the original contours and the convex hull and initialize
                    # the name of the shape
                    if h > 15:
                        cv2.drawContours(hullImage, [hull], -1, 255, -1)
                    cv2.drawContours(image, [c], -1, (240, 0, 159), 3)

                except:
                    pass

            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            closing = cv2.morphologyEx(hullImage, cv2.MORPH_CLOSE, rectKernel)
            if debug: cv2.imshow("closing", closing)

            _, countours, hierarcy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if debug: print('length of countours ', len(countours))
            avg = np.sum(heights)
            if debug: print('Sum. Height ', avg)
            if debug: print('Total Heights ', len(heights))
            avg = avg / len(heights)
            if debug: print('Avg. Height ', avg)

            minima_y_axes = []
            for pos in range(len(heights)):
                if heights[pos] < avg:
                    minima_y_axes.append(y_axis[pos])

            minimum_y_ax = np.min(minima_y_axes)
            if debug: print("Minimum Y ax ", minimum_y_ax)
            crop = backupImage[0:minimum_y_ax, 0:0 + width]
            if debug: cv2.imshow("crop ", crop)

            if debug: imageCopy = image.copy()
            if debug: cv2.imshow('drawn countours', cv2.drawContours(imageCopy, countours, -1, (0, 255, 0), 2))
            cv2.imwrite(os.path.join(dataset_images, filename), crop)
            page = page + 1
            # cv2.waitKey(0)


dataset_path = os.path.join(DATASET_PDF_FOLDER)
if os.path.exists(dataset_path) == False:
    os.mkdir(dataset_path)

dataset_images = os.path.join(DATASET_IMAGES_FOLDER)
if os.path.exists(dataset_images) == False:
    os.mkdir(dataset_images)

if len(os.listdir(dataset_path)) == 0:
    print('Please provide validator pdf')
else:

    dataset_filename = os.listdir(dataset_path)[0]
    dataset_pdf = os.path.join(dataset_path, dataset_filename)
    print('Reading pdf images from  ', dataset_pdf)

    images = pdf2image.convert_from_path(pdf_path=dataset_pdf,
                                         output_folder=dataset_images,
                                         first_page=1,
                                         dpi=100,
                                         thread_count=3,
                                         fmt='.jpg')
    print('Total images generated from pdf ', len(images))

    print("Extracting... logos according to page number")
    extract_logos(dataset_images)
