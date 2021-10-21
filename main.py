# from bs4 import BeautifulSoup

# f = open("assets/train_xml/0a2c344efb5dd5b88450eec236a2aa3b_2.xml", "r")
# soup = BeautifulSoup(f, 'html.parser')

# # print(soup.select('DL_ZONE[gedi_type="DLSignature"]'))
# print(soup.select('[gedi_type]'))

import tensorflow as tf
print("GPU available: ", tf.config.list_physical_devices('GPU'))