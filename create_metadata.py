import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2

english_to_hebrew = {
    'B': 'ב', 'C': 'כ', 'D': 'ו', 'F': 'ט', 'I': 'י', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'R': 'ר', 'S': 'ס',
    'T': 'ת', 'W': 'ש', 'Z': 'ז'
}

hebrew_dict = {
    'א': 0, 'ב': 1, 'ג': 2, 'ד': 3, 'ה': 4, 'ו': 5, 'ז': 6, 'ח': 7, 'ט': 8, 'י': 9, 'כ': 10, 'ל': 11, 'מ': 12, 'נ': 13,
    'ס': 14,
    'ע': 15, 'פ': 16, 'צ': 17, 'ק': 18, 'ר': 19, 'ש': 20, 'ת': 21
}

hehbrew_letters_in_english = {
    'ale': 'א', 'bet': 'ב', 'gim': 'ג', 'dal': 'ד', 'hey': 'ה', 'vav': 'ו', 'zay': 'ז', 'het': 'ח', 'tet': 'ט',
    'yud': 'י'
    , 'kaf': 'כ', 'lam': 'ל', 'mem': 'מ', 'nun': 'נ', 'sam': 'ס', 'ain': 'ע', 'pey': 'פ', 'tza': 'צ', 'kuf': 'ק'
    , 'rey': 'ר', 'shi': 'ש', 'taf': 'ת'

}


def create_tst_df(images_dir, _class_encoding=None, weight=1):
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
    weights = []
    for img in os.listdir(images_dir):
        #cur_label = img.split('_')[0]
        cur_label=img.split('.')[0] # ain.jpeg for example
        if len(cur_label) > 3:  # not a letter
            continue
        images_list.append(os.path.join(images_dir, img))
        labels_list.append(cur_label)
        lable_class_encoding.append(class_encoding[cur_label])
        eng_heb_list.append(hebrew_dict[hehbrew_letters_in_english[cur_label]])
        weights.append(weight)

    df["filename"] = images_list
    df["label"] = labels_list
    df['class_encoding'] = lable_class_encoding
    # df['hebrew'] = eng_heb_list
    df['weights'] = weights

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

    for dir in os.listdir(images_dir):  # dir is the letter
        if len(dir) > 3:
            continue
        if _class_encoding is None:  # creates class encoding. Change this if you dont want class encoding to be from counter
            class_encoding[dir] = hebrew_dict[hehbrew_letters_in_english[dir]]
            class_counter += 1
            #class_encoding[dir] = class_counter
        for img in os.listdir(os.path.join(images_dir, dir)):
            images_list.append(os.path.join(images_dir, dir, img))
            labels_list.append(dir)
            lable_class_encoding.append(class_encoding[dir])
            eng_heb_list.append(hebrew_dict[hehbrew_letters_in_english[dir]])
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

    if len(df) > 1:
        final_df = pd.concat(df, ignore_index=True)
    else:
        final_df = df[0]
    final_df.to_csv(os.path.join(path, f'{name}.csv'), index=False, encoding='utf-8-sig')
    print(f'metadata saved in: {path} with the name: {name}.csv')


if __name__ == '__main__':
    images_dir = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\processed_images\half_landmarks')
    subdir_name = f'Sivan_{datetime.now().strftime("%m%Y%d-%H%M")}'
    new_dir = create_new_dirs(subdir=subdir_name)
    TRdfs = []

    # region train dfs
    TRdf, class_encoding = create_trn_df(images_dir=images_dir / 'sivan')
    TRdfs.append(TRdf)
    # TRdf = create_trn_df(images_dir=images_dir / 'arabic_to_english', _class_encoding=class_encoding)
    # TRdfs.append(TRdf)

    # endregion

    # region test df
    TSdfs = []
    TSdf = create_tst_df(images_dir=images_dir / 'sivan_tst', _class_encoding=class_encoding)
    TSdfs.append(TSdf)
    #TSdf = create_trn_df(images_dir=images_dir / 'hebrew_to_english', _class_encoding=class_encoding,weight=15)
    # ----- add 80% friends to train
    # freinds_to_train = TSdf.sample(frac=0.8, random_state=666,ignore_index=False)
    #friends_to_train, TSdf = train_test_split(TSdf, test_size=0.2, stratify=TSdf.class_encoding,random_state=666)
    #TRdfs.append(friends_to_train)
    # ------
    #TSdfs.append(TSdf)
    save_df(df=TRdfs, path=new_dir, name='trn_metadata')
    save_df(df=TSdfs, path=new_dir, name='tst_metadata')

    # endregion
