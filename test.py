from skimage import measure, morphology
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import regionprops

n = 12
l = 256

# np.random.seed(1)
img = cv2.imread("assets/train/0d178d095434170eac2cb58cc244bb8c_2.tif") 
plt.imshow(img)
plt.show()

kernel = np.ones((3, 3), np.uint8)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# plt.imshow(img)
# plt.show()

# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# plt.imshow(img)
# plt.show()
img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

# kernel = np.ones((6,6),np.uint8)
# img = cv2.erode(img,kernel,iterations = 1)
plt.imshow(img)
plt.show()




# plt.imshow(img)
# plt.show()

# img = np.mean(img, axis=2)

im = img
# im = np.zeros((l, l))
# points = l * np.random.random((2, n ** 2))
# im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
# im = filters.gaussian(im, sigma= l / (4. * n))
blobs = im > im.mean()


all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=1)

# plt.figure(figsize=(9, 3.5))
# plt.subplot(131)
# plt.imshow(blobs, cmap='gray')
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(all_labels, cmap='nipy_spectral')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(blobs_labels, cmap='nipy_spectral')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0
for region in regionprops(blobs_labels):
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1
    # print region.area # (for debugging)
    # take regions with large enough areas
    if (region.area >= 250):
        if (region.area > the_biggest_component):
            the_biggest_component = region.area


average = (total_area / counter)
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))

# the parameters are used to remove small size connected pixels outliar 
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100

# the parameter is used to remove big size connected pixels outliar
constant_parameter_4 = 18

# experimental-based ratio calculation, modify it for your cases
# a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
# are smaller than a4_small_size_outliar_constant for A4 size scanned documents
a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
a4_small_size_outliar_constant = average * 3
print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

# experimental-based ratio calculation, modify it for your cases
# a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
# are bigger than a4_big_size_outliar_constant for A4 size scanned documents
a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4
print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

# remove the connected pixels are smaller than a4_small_size_outliar_constant
pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
for region in regionprops(pre_version):
    print(region.bbox)
# remove the connected pixels are bigger than threshold a4_big_size_outliar_constant 
# to get rid of undesired connected pixels such as table headers and etc.
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outliar_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0
# save the the pre-version which is the image is labelled with colors
# as considering connected components

plt.imshow(pre_version)
plt.show()

print(pre_version.shape)
# im1 = img.crop((102, 535, 135, 581))
# im1 = pre_version[102:135, 535:581]

# plt.imshow(im1)
# plt.show()

region_array = []
pixel_range = 50
for region in regionprops(pre_version):
    if len(region_array) == 0:
        region_array.append(region.bbox)
    elif abs(region_array[len(region_array) - 1][0] - region.bbox[0]) <= pixel_range or abs(region_array[len(region_array) - 1][3] - region.bbox[3]) <= pixel_range:
        previous_region = region_array[len(region_array) - 1]
        region_array[len(region_array) - 1] = (min(previous_region[0], region.bbox[0]),
                                               min(previous_region[1], region.bbox[1]), region.bbox[2],
                                               max(previous_region[3], region.bbox[3]),
                                               max(previous_region[4], region.bbox[4]), region.bbox[5])
    else:
        region_array.append(region.bbox)

print(region_array)

for region in region_array:
    im1 = pre_version[region[0]:region[3], region[1]:region[4]]
    plt.imshow(im1)
    plt.show()
