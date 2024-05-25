"""
dataset format：
datasets
    original_img
        class1
        class2
        -----
    train
        class1
        class2
        -----
    val
        class1
        class2
        -----
    test
        class1
        class2
        -----
"""
import os
from sklearn.model_selection import train_test_split
import shutil

original_img_path = './datasets/original_img'
train_img_path = "./datasets/train/"
val_img_path = "./datasets/val/"

images_name = os.listdir(original_img_path)


for i in range(len(images_name)):
    if os.path.exists(train_img_path + str(images_name[i])):
        pass
    else:
        os.mkdir(train_img_path + str(images_name[i]))
    if os.path.exists(val_img_path + str(images_name[i])):
        pass
    else:
        os.mkdir(val_img_path + str(images_name[i]))

for i in range(len(images_name)):
    path1 = train_img_path + str(images_name[i])
    path2 = train_img_path + str(images_name[i])
    print(path1)
    images = os.listdir(original_img_path + '/' + str(images_name[i]))  # all image
    train, val = train_test_split(images, train_size=0.9, random_state=42)  # Take out all the photos for segmentation

    for t in train:
        train_path = original_img_path + "/" + str(images_name[i]) + '/' + str(t)
        shutil.copyfile(train_path, path1 + '/' + str(t))  # Move image to another path

    for v in val:
        val_path = original_img_path + "/" + str(images_name[i]) + '/' + str(v)
        shutil.copyfile(val_path, path2 + '/' + str(v))  # Move image to another path





