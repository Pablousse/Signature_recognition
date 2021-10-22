import cv2
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
from shapely.geometry import Polygon


file_directory = "assets/Prediction/test"

def get_intersecting_area(rect_one, rect_two):
    polygon = create_rect(rect_one)
    other_polygon = create_rect(rect_two)
    intersection = polygon.intersection(other_polygon)

    return intersection.area


def create_rect(rect):
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]

    polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return polygon


def show_bounding_boxes(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    big_cnts = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
            big_cnts.append(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)

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
        bounding_box_array.append([int(signature['col']), int(signature['row']), int(signature['width']), int(signature['height'])])

    return bounding_box_array


def merge_boxes(bboxes_list):
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    for box in bboxes_list:
        if box[0] < x or x == 0:
            x = box[0]
        if box[1] < y or y == 0:
            y = box[1]
        if (box[0] + box[2]) > (x2) or x2 == 0:
            x2 = (box[0] + box[2])
        if (box[1] + box[3]) > (y2) or y2 == 0:
            y2 = (box[1] + box[3])

    w = x2 - x
    h = y2 - y

    return [x, y, w, h]


def intersection_check(rect_one, rect_two):
    if int(rect_one[0]) > (int(rect_two[0]) + int(rect_two[2])) or (int(rect_one[0]) + int(rect_one[2])) < int(rect_two[0]):
        return False
    if int(rect_one[1]) > (int(rect_two[1]) + int(rect_two[3])) or (int(rect_one[1]) + int(rect_one[3])) < int(rect_two[1]):
        return False
    return True


def show_intersecting_boxes(filename):

    train_directory = file_directory
    xml_directory = "assets/train_xml"

    test_path = os.path.join(train_directory, filename)
    test_xml_path = os.path.join(xml_directory, (os.path.splitext(filename)[0] + ".xml"))
    cnts = show_bounding_boxes(test_path)
    xml_cnts = get_bounding_box_from_xml(test_xml_path)

    if len(xml_cnts) == 0:
        bbox_flag = True
    else:
        bbox_flag = False

    bboxes = {}
    i = 0

    for cnt in cnts:
        x = cv2.boundingRect(cnt)[0]
        y = cv2.boundingRect(cnt)[1]
        w = cv2.boundingRect(cnt)[2]
        h = cv2.boundingRect(cnt)[3]
        bboxes[i] = [[x, y, w, h], False]

        i += 1

    i = 0
    image = cv2.imread(test_path)
    for xml_cnt in xml_cnts:
        xml_area = int(xml_cnt[2]) * int(xml_cnt[3])
        for cnt in bboxes.items():
            x = cnt[1][0][0]
            y = cnt[1][0][1]
            w = cnt[1][0][2]
            h = cnt[1][0][3]
            if intersection_check(xml_cnt, cnt[1][0]):
                if (w * h) > (15 * xml_area):
                    if not cnt[1][1]:
                        bboxes[cnt[0]] = [cnt[1][0], False, None]
                else:
                    bbox_flag = True
                    bboxes[cnt[0]] = [cnt[1][0], True, i]

            else:
                if not cnt[1][1]:
                    bboxes[cnt[0]] = [cnt[1][0], False, None]

        x = int(xml_cnt[0])
        y = int(xml_cnt[1])
        w = int(xml_cnt[2])
        h = int(xml_cnt[3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 36, 12), 3)

        i += 1

    boxes_to_merge = []

    xml_cnt_id = 0
    for xml_cnt in xml_cnts:
        for i, box in bboxes.items():
            if box[1] and box[2] == xml_cnt_id:
                x = box[0][0]
                y = box[0][1]
                w = box[0][2]
                h = box[0][3]

                intersecting_area = get_intersecting_area(xml_cnt, box[0])
                if intersecting_area is None:
                    intersecting_area = 0
                if intersecting_area < (0.2 * w * h) and intersecting_area < (0.2 * xml_cnt[3] * xml_cnt[2]) :
                    box[1] = False
                    box[2] = None
                    bboxes[i] = box

                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)

        output = [k for k in bboxes if bboxes[k][2] == xml_cnt_id]
        bboxes_list = []

        if len(output) > 0:
            for x in output:
                bboxes_list.append([x, bboxes[x]])

            boxes_to_merge.append(bboxes_list)

        xml_cnt_id += 1

    for i, box in bboxes.items():
        x = box[0][0]
        y = box[0][1]
        w = box[0][2]
        h = box[0][3]
        if box[1]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), 3)

    merged_boxes = []
    # print("-------------------------")
    # print(boxes_to_merge)
    if len(boxes_to_merge) > 0:
        for box_to_merge in boxes_to_merge:
            if len(box_to_merge) > 1:
                # print([box_to_merge[i][1][0] for i in range(len(box_to_merge))])
                merged_box = merge_boxes([box_to_merge[i][1][0] for i in range(len(box_to_merge))])
                x = merged_box[0]
                y = merged_box[1]
                w = merged_box[2]
                h = merged_box[3]
                cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 3)
                merged_boxes.append(merged_box)
                for box in box_to_merge:
                    # print(box[0])
                    bboxes.pop(box[0])

                max_key = max(bboxes.keys()) + 1
                bboxes[max_key] = [merged_box, True, None]

    # print(bboxes)
    # print(merged_boxes)
    # print(filename)
    # plt.imshow(image)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    # plt.show()

    return bbox_flag, bboxes


def crop_box_and_save(bbox, bbox_number, is_good, filename):
    x, y, w, h = bbox
    image = cv2.imread(os.path.join(file_directory, filename))
    crop = image[y: y + h, x: x + w]

    if is_good is not None:
        if is_good:
            directory = "assets/Cropped_image_without_val/train/good"
        else:
            directory = "assets/Cropped_image_without_val/train/bad"
    else:
        directory = "assets/prediction_cropped/test"
    cropped_image_path = os.path.join(directory, os.path.splitext(filename)[0] + "_" + str(bbox_number) + ".tif")
    cv2.imwrite(cropped_image_path, crop)


# show_intersecting_boxes("ad729ba5891d6199f01d12d59086bb86_7.tif")
# result = []
# for filename in os.listdir(file_directory):
#     is_correct, bboxes = show_intersecting_boxes(filename)
#     result.append(is_correct)

#     # if not is_correct:
#     #     print(filename)
#     #     os.remove(os.path.join(file_directory, filename))
#     # bboxes

#     for key, value in bboxes.items():
#         crop_box_and_save(value[0], key, value[1], filename)

for filename in os.listdir(file_directory):
    cnts = show_bounding_boxes(os.path.join(file_directory, filename))

    i = 0
    bboxes = []
    for cnt in cnts:
        x = cv2.boundingRect(cnt)[0]
        y = cv2.boundingRect(cnt)[1]
        w = cv2.boundingRect(cnt)[2]
        h = cv2.boundingRect(cnt)[3]
        bboxes.append([[x, y, w, h]])

        i += 1

    for i in range(0, len(bboxes)):
        crop_box_and_save(bboxes[i][0], i, None, filename)


# print("len is " + str(len(result)))
# print("sum is " + str(sum(result)))
# print("Wrong detection box is " + str(int(len(result) - sum(result))))
