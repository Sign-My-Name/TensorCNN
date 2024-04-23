import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
import cv2

english_to_hebrew = {
    'B': 'ב', 'C': 'כ', 'D': 'ו', 'F': 'ט', 'I': 'י', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'R': 'ר', 'S': 'ס',
    'T': 'ת', 'W': 'ש', 'Z': 'ז'
}


def create_tst_df(images_dir, _class_encoding=None):
    """
    assums that all tst images are in the same directory (no subdirs) and each img name starts with the letter it represent and then underline
    for example, an img represent the letter B will be named B_something
    :param images_dir:
    :param _class_encoding:
    :return:
    """
    df = pd.DataFrame()
    images_list = []
    labels_list = []
    lable_class_encoding = []
    eng_heb_list = []
    for img in os.listdir(images_dir):
        cur_label = img.split('_')[0]
        if len(cur_label) > 1:  # not a letter
            continue
        images_list.append(os.path.join(images_dir, img))
        labels_list.append(cur_label)
        lable_class_encoding.append(class_encoding[cur_label])
        eng_heb_list.append(english_to_hebrew[cur_label])

    df["filename"] = images_list
    df["label"] = labels_list
    df['class_encoding'] = lable_class_encoding
    df['hebrew'] = eng_heb_list

    return df


def create_trn_df(images_dir, _class_encoding=None, weight=1):
    """
    Assums that in imgs_dir there are dirs named by the letter only (len=1)
    :param imgs_dir:
    :return: metadata df. if _class_encoding is None, returns class_encoding dict
    """
    df = pd.DataFrame()
    images_list = []
    labels_list = []
    eng_heb_list = []
    weights = []
    lable_class_encoding = []
    class_counter = 0

    if _class_encoding is None:
        class_encoding = {}
    else:
        class_encoding = _class_encoding

    for dir in os.listdir(images_dir):
        if len(dir) > 1:
            continue
        if _class_encoding is None:  # creates class encoding
            class_encoding[dir] = class_counter
            class_counter += 1
        for img in os.listdir(os.path.join(images_dir, dir)):
            images_list.append(os.path.join(images_dir, dir, img))
            labels_list.append(dir)
            lable_class_encoding.append(class_encoding[dir])
            eng_heb_list.append(english_to_hebrew[dir])
            weights.append(weight)
    df["filename"] = images_list
    df["label"] = labels_list
    df['class_encoding'] = lable_class_encoding
    df['weights'] = weights
    df['hebrew'] = eng_heb_list
    if _class_encoding is None:
        return df, class_encoding
    else:
        return df


def create_new_dirs(subdir=None):
    # Create the new directory path
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\metadata')

    if subdir is None:
        new_dir = root / f'new_metadata_{datetime.now().strftime("%m%Y%d-%H%M")}'
    else:
        new_dir = root / subdir
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def save_df(df, path, name):
    """
    if df.size > 1, concating the dfs
    :param dfs: []
    :param path:
    :return:
    """

    if len(dfs) > 1:
        final_df = pd.concat(df, ignore_index=True)
    else:
        final_df = df[0]
    final_df.to_csv(os.path.join(path, f'{name}.csv'), index=False,encoding='utf-8-sig')
    print(f'metadata saved in: {path} with the name: {name}.csv')


if __name__ == '__main__':
    images_dir = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\processed_images')
    subdir_name = f'equalWeights_{datetime.now().strftime("%m%Y%d-%H%M")}'
    new_dir = create_new_dirs(subdir=subdir_name)
    dfs = []

    # region train dfs
    df,class_encoding = create_trn_df(images_dir=r'C:\Users\40gil\Desktop\degree\year_4\sm2\final_project\cropped_128X128)01_37_19\asl_alphabet_train')
    dfs.append(df)
    df = create_trn_df(images_dir=r'C:\Users\40gil\Desktop\degree\year_4\sm2\final_project\cropped_128X128)18_57_14\arabic_to_english', _class_encoding=class_encoding)
    dfs.append(df)

    save_df(df=dfs, path=new_dir, name='trn_metadata')

    # endregion

    # region test df
    dfs=[]
    df = create_tst_df(images_dir=r'C:\Users\40gil\Desktop\degree\year_4\sm2\final_project\cropped_128X128)01_37_19\asl_alphabet_test', _class_encoding=class_encoding)
    dfs.append(df)
    save_df(df=dfs, path=new_dir, name='tst_metadata')
    # endregion
