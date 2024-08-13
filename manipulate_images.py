import os
from pathlib import Path
from datetime import datetime
import images_augmentations as aug
import cv2
from PIL import Image
import numpy as np

# region Globals


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

def create_new_dirs(preprocess_imgs_path, subdir=None):
    # Create the new directory path
    root = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\processed_images')

    if subdir is None:
        new_dir = root / f'new_crop_{datetime.now().strftime("%m%Y%d-%H%M")}'
    else:
        new_dir = root / subdir
    new_dir.mkdir(parents=True, exist_ok=True)

    # Path to the original images directory
    if preprocess_imgs_path is None:
        images_dir_origin = Path(r"C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images")
    else:
        images_dir_origin = Path(preprocess_imgs_path)
    return images_dir_origin, new_dir


def process_images(preprocess_imgs_path=None, new_dir_name=None, size=None, to_cut=False, to_transform=False,
                   to_rotate=False, hebrew_path=False, to_print_process=False):
    """
copy images from preprocess_imgs_path to subdir_name and manipulate them.
no need for preprocess path if the following tree is your tree from current directory.

tensor_training (current directory):\n
├───.idea\n
├───images (preprocess)\n
└───running_outputs\n

    :param preprocess_imgs_path:
    :param new_dir_name: name of the directory with the images after manipulation.
    :param size:Tuples (x,y) will resize image.
    :param to_rotate: whether to rotate. if True, will rotate img in 90 deg clockwise
    :param to_transform: whether to transform according to transformation matrix
    :param to_cut: whether to cut image according to given size
    :param hebrew_path: is the images dirs named with hebrew letters
    :return: new dir created path
    """
    if new_dir_name is None:
        subdir = 'new_crop'
    else:
        subdir = new_dir_name

    images_dir, new_dir = create_new_dirs(preprocess_imgs_path=preprocess_imgs_path, subdir=subdir)

    # Loop through all directories in the original images directory
    for subdir_name in os.listdir(images_dir):
        print(f'------{subdir_name}')
        subdir_path = images_dir / subdir_name
        if os.path.isdir(subdir_path):
            if hebrew_path:
                subdir_name = hebrew_dict[subdir_name]
            new_subsubdir = new_dir / subdir_name
            new_subsubdir.mkdir(parents=True, exist_ok=True)

            # Loop through all images in the current subdirectory
            for img_name in os.listdir(subdir_path):
                if to_print_process:
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
                        img = aug.cut(image=img, size=size)
                    if to_transform and img is not None:
                        img = aug.transform(img)
                    # Save the cropped image to the new location
                    if img is not None:
                        new_img_path = new_subsubdir / img_name
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')


def process_words_images(preprocess_imgs_path=None, new_dir_name=None, size=None, to_cut=False, to_transform=False,
                         to_rotate=False, to_print_process=False):
    if new_dir_name is None:
        subdir = 'new_crop'
    else:
        subdir = new_dir_name

    images_dir, new_dir = create_new_dirs(preprocess_imgs_path=preprocess_imgs_path, subdir=subdir)
    # Loop through all directories in the original images directory
    for dir_name in os.listdir(images_dir):
        print(f'--------------{dir_name}')
        dir_path = images_dir / dir_name
        if os.path.isdir(dir_path):
            new_sub_dir = new_dir / dir_name
            new_sub_dir.mkdir(parents=True, exist_ok=True)
            # Loop through all images in the current subdirectory
            for img_name in os.listdir(dir_path):
                if to_print_process:
                    (f'-{img_name}')
                img_path = dir_path / img_name

                # Load and crop the image
                img = cv2.imread(str(img_path))
                if img is not None:
                    if to_rotate:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    if to_cut:
                        img = aug.cut(image=img, size=size)
                    if to_transform and img is not None:
                        img = aug.transform(img)
                    # Save the cropped image to the new location
                    if img is not None:
                        new_img_path = new_sub_dir / img_name
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(new_img_path))
    print(f'processed images saved in: {new_dir}')


if __name__ == '__main__':
    process_images(preprocess_imgs_path=r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images'
                                        r'\example\sivan_example',
                   new_dir_name='sivan_example_processed', to_cut=True, to_rotate=True, hebrew_path=True,
                   to_transform=False)

    # process_words_images(preprocess_imgs_path=r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images\example'
    #                                           r'\words_example',
    #                     new_dir_name='sivan_words_exmaple', to_cut=True, to_rotate=False)
