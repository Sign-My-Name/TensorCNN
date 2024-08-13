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

def create_new_dirs(preprocess_imgs_path: str = None, subdir: str = None) -> tuple[Path, Path]:
    """Creates the new directory for processed images and returns the original and new paths."""
    root = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\processed_images')

    new_dir = root / (subdir if subdir else f'new_crop_{datetime.now().strftime("%m%Y%d-%H%M")}')
    new_dir.mkdir(parents=True, exist_ok=True)

    images_dir_origin = Path(preprocess_imgs_path) if preprocess_imgs_path else Path(
        r"C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images")

    return images_dir_origin, new_dir


def process_images(preprocess_imgs_path: str = None, new_dir_name: str = None, size: tuple = None, to_cut: bool = False,
                   to_transform: bool = False, to_rotate: bool = False, hebrew_path: bool = False,
                   to_print_process: bool = False) -> Path:
    """
    Copies images from preprocess_imgs_path to subdir_name and manipulates them.

    :param preprocess_imgs_path: Path to the original images directory.
    :param new_dir_name: Name of the directory for the processed images.
    :param size: Tuple (x, y) to resize the image.
    :param to_rotate: Whether to rotate images 90 degrees clockwise.
    :param to_transform: Whether to apply a transformation matrix.
    :param to_cut: Whether to cut images to a specified size.
    :param hebrew_path: If True, interpret directory names as Hebrew characters.
    :param to_print_process: If True, prints the processing steps.
    :return: Path to the new directory with processed images.
    """
    images_dir, new_dir = create_new_dirs(preprocess_imgs_path=preprocess_imgs_path, subdir=new_dir_name)

    for subdir_name in os.listdir(images_dir):
        print(f'Processing directory: {subdir_name}')

        subdir_path = images_dir / subdir_name
        if os.path.isdir(subdir_path):
            subdir_name = hebrew_dict.get(subdir_name, subdir_name) if hebrew_path else subdir_name

            new_trn_subsubdir = new_dir / 'trn' / subdir_name
            new_trn_subsubdir.mkdir(parents=True, exist_ok=True)
            new_tst_subsubdir = new_dir / 'tst'
            new_tst_subsubdir.mkdir(parents=True, exist_ok=True)

            is_moved_to_tst = False
            for img_name in os.listdir(subdir_path):
                if to_print_process:
                    print(f'Processing image: {img_name}')

                img_path = subdir_path / img_name
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED) if hebrew_path else cv2.imread(str(img_path))

                if img is not None:
                    img = aug.augment_image(img, to_cut=to_cut, to_transform=to_transform, to_rotate=to_rotate,
                                            size=size)
                    if img is not None:
                        if not is_moved_to_tst:
                            tst_img_name = subdir_name + '.jpg'
                            new_img_path = new_tst_subsubdir / tst_img_name
                            Image.fromarray(img).save(str(new_img_path))
                            is_moved_to_tst = True
                            continue
                        new_img_path = new_trn_subsubdir / img_name
                        Image.fromarray(img).save(str(new_img_path))

    print(f'Processed images saved in: {new_dir}')
    return new_dir


def process_words_images(preprocess_imgs_path: str = None, new_dir_name: str = None, size: tuple = None,
                         to_cut: bool = False, to_transform: bool = False, to_rotate: bool = False,
                         to_print_process: bool = False) -> Path:
    """
    Processes images in word directories similar to `process_images`.
    """
    return process_images(preprocess_imgs_path=preprocess_imgs_path, new_dir_name=new_dir_name, size=size,
                          to_cut=to_cut, to_transform=to_transform, to_rotate=to_rotate, hebrew_path=False,
                          to_print_process=to_print_process)


if __name__ == '__main__':
    # process_images(
    #     preprocess_imgs_path=r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images\example\sivan_example',
    #     new_dir_name='sivan_example_processed', to_cut=True, to_rotate=True, hebrew_path=True, to_transform=False,
    #     to_print_process=True)

    process_words_images(preprocess_imgs_path=r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\images\example\words_example',
                         new_dir_name='sivan_words_example', to_cut=True, to_rotate=False,
                         to_print_process=True)
