import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
import cv2

# region Globals

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
english_to_hebrew = {
    'B': 'ב', 'C': 'כ', 'D': 'ו', 'F': 'ט', 'I': 'י', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'R': 'ר', 'S': 'ס',
    'T': 'ת', 'W': 'ש', 'Z': 'ז'
}
# endregion
# region Image manipulation

def cut(image, size=None):
    """
    :param image: cv2 image
    :param size: (width,height)
    :return: cv2 image
    """
    image = image.astype(np.uint8)
    processed_image = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if processed_image.multi_hand_landmarks:
        hand_landmarks = processed_image.multi_hand_landmarks[0]
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_min_adjust = int(x_min * image.shape[1] - 80)
        y_min_adjust = int(y_min * image.shape[0] - 80)
        x_max_adjust = int(x_max * image.shape[1] + 80)
        y_max_adjust = int(y_max * image.shape[0] + 80)
        if x_min_adjust < 0:
            x_min_adjust = int(x_min * image.shape[1] - 15)
            if x_min_adjust < 0:
                x_min_adjust = 0
        if y_min_adjust < 0:
            y_min_adjust = int(y_min * image.shape[0] - 15)
            if y_min_adjust < 0:
                y_min_adjust = 0
        x_min = x_min_adjust
        y_min = y_min_adjust
        x_max = x_max_adjust
        y_max = y_max_adjust
        hand_region = image[y_min:y_max, x_min:x_max]
        hand_region_uint8 = hand_region.astype(np.uint8)
        hand_region_bgr = cv2.cvtColor(hand_region_uint8, cv2.COLOR_RGB2BGR)
        hand_region_bgr = cv2.resize(hand_region_bgr, dsize=size)
        return hand_region_bgr
    else:
        return None


def transform(image):
    if image is None or image.shape[0] == 0:
        return None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Gray scale

    sharpening_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    sharpened_image = cv2.filter2D(image_gray, -1, sharpening_kernel)  # applying sharpening kernal
    return sharpened_image


# endregion
def create_new_dirs(preprocess_imgs_path, subdir):
    # Create the new directory path
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\processed_images')

    if subdir is None:
        new_dir = root / 'new_crop'
    else:
        new_dir = root / subdir
    new_dir.mkdir(parents=True, exist_ok=True)

    # Path to the original images directory
    if preprocess_imgs_path is None:
        images_dir_origin = Path(r"C:\Users\40gil\Desktop\final_project\tensor_training\images")
    else:
        images_dir_origin = Path(preprocess_imgs_path)
    return images_dir_origin, new_dir


def process_images(preprocess_imgs_path=None, subdir_name=None, size=None, to_cut=False,to_transform=False):
    """
copy images from preprocess_imgs_path to subdir_name and manipulate them.
no need for preprocess path if this is your tree from current directory.

tensor_training (current directory):\n
├───.idea\n
├───create_models\n
├───DataCollectionProtocol\n
├───images (preprocess)\n
└───running_outputs\n

    :param preprocess_imgs_path:
    :param subdir_name: name of the directory with the images after manipulation.
    :param size:Tuples (x,y) will resize image.
    :return: new dir created path
    """
    if subdir_name is None:
        subdir = 'new_crop'
    else:
        subdir = subdir_name

    images_dir, new_dir = create_new_dirs(preprocess_imgs_path=preprocess_imgs_path, subdir=subdir)

    # Loop through all directories in the original images directory
    for dir_name in os.listdir(images_dir):
        print(f'--------------{dir_name}')
        if dir_name == 'asl_alphabet_test':
            # in test dir. this dir is different because images are inside and not in subdir with letter name
            for img_name in os.listdir(images_dir / dir_name):
                print(f'-{img_name}')
                img_path = images_dir / dir_name / img_name

                this_path = new_dir / dir_name
                this_path.mkdir(parents=True, exist_ok=True)

                # Load img
                img = cv2.imread(str(img_path))
                if img is not None:
                    if to_cut:
                        img = cut(image=img, size=size)
                    if to_transform and img is not None:
                        img = transform(img)
                    # Save the cropped image to the new location
                    if img is not None:
                        new_img_path = this_path / img_name
                        cv2.imwrite(str(new_img_path), img)
            continue
        dir_path = images_dir / dir_name
        if os.path.isdir(dir_path):
            new_subdir = new_dir / dir_name
            new_subdir.mkdir(parents=True, exist_ok=True)

            # Loop through all subdirectories in the current directory
            for subdir_name in os.listdir(dir_path):
                print(f'------{subdir_name}')
                subdir_path = dir_path / subdir_name
                if os.path.isdir(subdir_path):
                    new_subsubdir = new_subdir / subdir_name
                    new_subsubdir.mkdir(parents=True, exist_ok=True)

                    # Loop through all images in the current subdirectory
                    for img_name in os.listdir(subdir_path):
                        print(f'-{img_name}')
                        img_path = subdir_path / img_name

                        # Load and crop the image
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            if to_cut:
                                img = cut(image=img, size=size)
                            if to_transform and img is not None:
                                img = transform(img)
                            # Save the cropped image to the new location
                            if img is not None:
                                new_img_path = new_subsubdir / img_name
                                cv2.imwrite(str(new_img_path), img)
    return new_dir
def create_train_meta_data():
    df = pd.DataFrame()
    images_list = []
    labels_list = []
    eng_heb_list=[]


if __name__ == '__main__':
    new_dir=process_images()
    #class_encoding = create_train_meta_data()
    #create_test_meta_data(class_encoding)
