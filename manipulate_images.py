import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# region Globals

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
english_to_hebrew = {
    'B': 'ב', 'C': 'כ', 'D': 'ו', 'F': 'ט', 'I': 'י', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'R': 'ר', 'S': 'ס',
    'T': 'ת', 'W': 'ש', 'Z': 'ז'
}
hebrew_dict = {
    'א': 'ale', 'ב': 'bet', 'ג': 'gim', 'ד': 'dal', 'ה': 'hey', 'ו': 'vav', 'ז': 'zay', 'ח': 'het', 'ט': 'tet',
    'י': 'yud', 'כ': 'kaf',
    'ל': 'lam', 'מ': 'mem', 'נ': 'nun', 'ס': 'sam',
    'ע': 'ain', 'פ': 'pey', 'צ': 'tza', 'ק': 'kuf', 'ר': 'rey', 'ש': 'shi', 'ת': 'taf'
}


# endregion
# region Image manipulation
def draw_hand_skeleton(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image
    return image


def isolate_and_crop_hand(image, padding=60):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        image = draw_hand_skeleton(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for hand_landmarks in results.multi_hand_landmarks:
            points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                      for landmark in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)

            # Apply padding via dilation
            kernel = np.ones((padding * 2, padding * 2), np.uint8)
            padded_mask = cv2.dilate(mask, kernel, iterations=1)

            # Create a black background image
            black_background = np.zeros_like(image)

            # Isolate the hand by combining it with the black background
            isolated_hand = cv2.bitwise_and(image, image, mask=padded_mask)
            final_image = cv2.bitwise_or(black_background, isolated_hand)

            # Calculate bounding box for the isolated hand with padding
            x, y, w, h = cv2.boundingRect(padded_mask)
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x_start + w + 2 * padding, image.shape[1])
            y_end = min(y_start + h + 2 * padding, image.shape[0])

            # Crop the image to the bounding box with padding
            cropped_image = final_image[y_start:y_end, x_start:x_end]
            return cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return None


def cut(image, size=None):
    """
    :param image: cv2 image
    :param size: (width,height)
    :return: cv2 image
    """
    image = image.astype(np.uint8)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = hands.process(rgb_image)
    if processed_image.multi_hand_landmarks:
        hand_landmarks = processed_image.multi_hand_landmarks[0]
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        val_to_adjust = max(image.shape[0], image.shape[1]) * 0.08
        x_min_adjust = int(x_min * image.shape[1] - val_to_adjust)
        y_min_adjust = int(y_min * image.shape[0] - val_to_adjust)
        x_max_adjust = int(x_max * image.shape[1] + val_to_adjust)
        y_max_adjust = int(y_max * image.shape[0] + val_to_adjust)
        # x_min_adjust = int(x_min * image.shape[1])
        # y_min_adjust = int(y_min * image.shape[0])
        # x_max_adjust = int(x_max * image.shape[1])
        # y_max_adjust = int(y_max * image.shape[0])
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
        hand_region = rgb_image[y_min:y_max, x_min:x_max]
        hand_region_uint8 = hand_region.astype(np.uint8)
        # hand_region_bgr = cv2.cvtColor(hand_region_uint8, cv2.COLOR_RGB2BGR)
        if size is not None:
            hand_region_bgr = cv2.resize(hand_region_uint8, dsize=size)
        return hand_region_uint8
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
def create_new_dirs(preprocess_imgs_path, subdir=None):
    # Create the new directory path
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\processed_images')

    if subdir is None:
        new_dir = root / f'new_crop_{datetime.now().strftime("%m%Y%d-%H%M")}'
    else:
        new_dir = root / subdir
    new_dir.mkdir(parents=True, exist_ok=True)

    # Path to the original images directory
    if preprocess_imgs_path is None:
        images_dir_origin = Path(r"C:\Users\40gil\Desktop\final_project\tensor_training\images")
    else:
        images_dir_origin = Path(preprocess_imgs_path)
    return images_dir_origin, new_dir


def process_images(preprocess_imgs_path=None, subdir_name=None, size=None, to_cut=False, to_transform=False,
                   to_rotate=False, hebrew_path=False):
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
    :param to_rotate: whether to rotate. if True, will rotate img in 90 deg clockwise
    :param to_transform: whether to transform according to transformation matrix
    :param to_cut: whether to cut image according to given size
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
                        new_img_path = new_subsubdir / img_name
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(new_img_path))
            continue
        elif dir_name == 'outer':
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
                    if hebrew_path:
                        subdir_name = hebrew_dict[subdir_name]
                    new_subsubdir = new_subdir / subdir_name
                    new_subsubdir.mkdir(parents=True, exist_ok=True)

                    # Loop through all images in the current subdirectory
                    for img_name in os.listdir(subdir_path):
                        print(f'-{img_name}')
                        img_path = subdir_path / img_name

                        # Load and crop the image
                        if hebrew_path:
                            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        else:
                            img = cv2.imread(str(img_path))
                        if img is not None:
                            if to_rotate:
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            if to_cut:
                                img = cut(image=img, size=size)
                            if to_transform and img is not None:
                                img = transform(img)
                            # Save the cropped image to the new location
                            if img is not None:
                                new_img_path = new_subsubdir / img_name
                                pil_img = Image.fromarray(img)
                                pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')


def process_images_sivan(preprocess_imgs_path=None, subdir_name=None, size=None, to_cut=False, to_transform=False,
                         to_rotate=False, hebrew_path=False):
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
                        new_img_path = new_subsubdir / img_name
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(new_img_path))
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
                    if hebrew_path:
                        subdir_name = hebrew_dict[subdir_name]
                    new_subsubdir = new_subdir / subdir_name
                    new_subsubdir.mkdir(parents=True, exist_ok=True)

                    # Loop through all images in the current subdirectory
                    for img_name in os.listdir(subdir_path):
                        print(f'-{img_name}')
                        img_path = subdir_path / img_name

                        # Load and crop the image
                        if hebrew_path:
                            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        else:
                            img = cv2.imread(str(img_path))
                        if img is not None:
                            if to_rotate:
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            if to_cut:
                                img = cut(image=img, size=size)
                            if to_transform and img is not None:
                                img = transform(img)
                            # Save the cropped image to the new location
                            if img is not None:
                                new_img_path = new_subsubdir / img_name
                                pil_img = Image.fromarray(img)
                                pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')


def create_train_meta_data():
    df = pd.DataFrame()
    images_list = []
    labels_list = []
    eng_heb_list = []


def process_images_to_half_landmarks(preprocess_imgs_path=None, subdir_name=None, hebrew_path=False):
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
                    img = isolate_and_crop_hand(img)
                    # draw landmark and crop
                    if img is not None:
                        new_img_path = new_subsubdir / img_name
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(new_img_path))
            continue
        elif dir_name == 'outer':
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
                    if hebrew_path:
                        subdir_name = hebrew_dict[subdir_name]
                    new_subsubdir = new_subdir / subdir_name
                    new_subsubdir.mkdir(parents=True, exist_ok=True)

                    # Loop through all images in the current subdirectory
                    for img_name in os.listdir(subdir_path):
                        print(f'-{img_name}')
                        img_path = subdir_path / img_name

                        # Load and crop the image
                        if hebrew_path:
                            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        else:
                            img = cv2.imread(str(img_path))
                        if img is not None:
                            img=isolate_and_crop_hand(img)
                            if img is not None:
                                new_img_path = new_subsubdir / img_name
                                pil_img = Image.fromarray(img)
                                pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')

def process_images_to_half_landmarks_sivan(preprocess_imgs_path=None, subdir_name=None, hebrew_path=False,to_rotate=False):
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
                    if hebrew_path:
                        subdir_name = hebrew_dict[subdir_name]
                    new_subsubdir = new_subdir / subdir_name
                    new_subsubdir.mkdir(parents=True, exist_ok=True)

                    # Loop through all images in the current subdirectory
                    for img_name in os.listdir(subdir_path):
                        print(f'-{img_name}')
                        img_path = subdir_path / img_name

                        # Load and crop the image
                        if hebrew_path:
                            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        else:
                            img = cv2.imread(str(img_path))
                        if img is not None:
                            if to_rotate:
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            img=isolate_and_crop_hand(img)
                            if img is not None:
                                new_img_path = new_subsubdir / img_name
                                pil_img = Image.fromarray(img)
                                pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')

if __name__ == '__main__':
    # process_images(preprocess_imgs_path=r'C:\Users\40gil\Desktop\final_project\tensor_training\images',
    # # subdir_name='NewCut',to_cut=True,to_rotate=False,hebrew_path=False)

    # process_images_sivan(preprocess_imgs_path=r'C:\Users\40gil\Desktop\final_project\tensor_training\images\outer',
    #                      subdir_name='NewCut', to_cut=True, to_rotate=True, hebrew_path=True)

    #process_images_to_half_landmarks(
     #   preprocess_imgs_path=r'C:\Users\40gil\Desktop\final_project\tensor_training\images',
      #  subdir_name='half_landmarks')
    process_images_to_half_landmarks_sivan(preprocess_imgs_path=r'C:\Users\40gil\Desktop\final_project\tensor_training\images\outer',
                                           subdir_name="half_landmarks",hebrew_path=True,to_rotate=True)
    # image = cv2.imread(r"C:\Users\40gil\Desktop\final_project\tensor_training\images\arabic_to_english\D\Beh_219.jpg")
    # cut(image=image)
