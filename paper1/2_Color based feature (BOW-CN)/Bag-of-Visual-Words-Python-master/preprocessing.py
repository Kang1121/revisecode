# Get/Cut 1 image from each folder and rename as the folder's name
import os
import shutil

path = 'E:/train'
test_path = 'E:/test'

for number in os.listdir(path):
    count = 0
    number_path = os.path.join(path, number)
    test_path_number = os.path.join(test_path, number)
    # print(test_path_number)
    os.makedirs(test_path_number)

    for im in os.listdir(number_path):
        if count == 2:
            break
        im_path = os.path.join(number_path, im)
        #print(test_path_number)
        shutil.move(im_path, test_path_number+'/'+im)
        count = count + 1
