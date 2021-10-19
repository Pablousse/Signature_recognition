import cv2
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup


def show_bounding_boxes(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    big_cnts = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
            big_cnts.append(c)
            # print(cv2.boundingRect(c))
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

    # plt.imshow(image)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    # plt.show()

    return big_cnts


def get_bounding_box_from_xml(path):
    f = open(path)
    soup = BeautifulSoup(f)
    signatures = soup.select("DL_ZONE[gedi_type='DLSignature']")
    bounding_box_array = []
    for signature in signatures:
        bounding_box_array.append([signature['col'], signature['row'], signature['width'], signature['height']])

    return bounding_box_array


def intersection_check(rect_one, rect_two):
    if int(rect_one[0]) > (int(rect_two[0]) + int(rect_two[2])) or (int(rect_one[0]) + int(rect_one[2])) < int(rect_two[0]):
        return False
    if int(rect_one[1]) > (int(rect_two[1]) + int(rect_two[3])) or (int(rect_one[1]) + int(rect_one[3])) < int(rect_two[1]):
        return False
    return True

def show_intersecting_boxes(filename):

    train_directory = "assets/train"
    xml_directory = "assets/train_xml"

    test_path = os.path.join(train_directory, filename)
    test_xml_path = os.path.join(xml_directory, (os.path.splitext(filename)[0] + ".xml"))
    cnts = show_bounding_boxes(test_path)
    xml_cnts = get_bounding_box_from_xml(test_xml_path)

    if len(xml_cnts) == 0:
        bbox_flag = True
    else:
        bbox_flag = False

    image = cv2.imread(test_path)
    for xml_cnt in xml_cnts:
        xml_area = int(xml_cnt[2]) * int(xml_cnt[3])
        for cnt in cnts:
            if intersection_check(xml_cnt, cv2.boundingRect(cnt)):
                x = cv2.boundingRect(cnt)[0]
                y = cv2.boundingRect(cnt)[1]
                w = cv2.boundingRect(cnt)[2]
                h = cv2.boundingRect(cnt)[3]
                if (w * h) > (20 * xml_area):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36,12,255), 3)
                else:
                    bbox_flag = True
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

        x = int(xml_cnt[0])
        y = int(xml_cnt[1])
        w = int(xml_cnt[2])
        h = int(xml_cnt[3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255,36,12), 3)


    plt.imshow(image)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
    print(filename)

    return bbox_flag


result = []
for filename in os.listdir("assets/train"):
    result.append(show_intersecting_boxes(filename))

print("len is " + str(len(result)))
print("sum is " + str(sum(result)))
print("Wrong detection box is " + str(int(len(result)-sum(result))))

# train_directory = "assets/train"
# xml_directory = "assets/train_xml"

#     show_bounding_boxes(os.path.join(train_directory, filename))
#     print(os.path.join(xml_directory, (os.path.splitext(filename)[0] + ".xml")))