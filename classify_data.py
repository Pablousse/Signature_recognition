from bs4 import BeautifulSoup
from shutil import copy
import os


def has_signature(path):
    f = open(path, "r")
    soup = BeautifulSoup(f, 'html.parser')

    if len(soup.select('DL_ZONE[gedi_type="DLSignature"]')) > 0:
        return True
    else:
        return False


def copy_to_right_folder(filename):
    train_folder = "assets/train_xml/"
    path = os.path.join(train_folder, filename)
    folder = ""
    if has_signature(path):
        folder = "assets/train_complete_page/good"
    else:
        folder = "assets/train_complete_page/bad"

    copy(os.path.join("assets/train", (os.path.splitext(filename)[0] + ".tif")), folder)


# signature_number = 0
# no_signature = 0
# for filename in os.listdir("assets/train_xml"):
#     # copy_to_right_folder(filename)
#     train_folder = "assets/train_xml/"
#     path = os.path.join(train_folder, filename)
#     if has_signature(path):
#         signature_number += 1
#     else:
#         no_signature += 1

# print("Signature : " + str(signature_number))
# print("No signature : " + str(no_signature))
