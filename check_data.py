import os
import glob

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('originals' + os.sep)
tra_label_dir = os.path.join('masks' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

missing_images = []
missing_labels = []

for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    label_path = data_dir + tra_label_dir + imidx + label_ext

    if not os.path.isfile(label_path):
        missing_labels.append(label_path)

    if not os.path.isfile(img_path):
        missing_images.append(img_path)

print("Missing images:")
for missing_img in missing_images:
    print(missing_img)

print("Missing labels:")
for missing_lbl in missing_labels:
    print(missing_lbl)
