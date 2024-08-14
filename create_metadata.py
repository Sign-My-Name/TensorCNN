import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Global Mappings
english_to_hebrew = {
    'B': 'ב', 'C': 'כ', 'D': 'ו', 'F': 'ט', 'I': 'י', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'R': 'ר', 'S': 'ס',
    'T': 'ת', 'W': 'ש', 'Z': 'ז'
}

hebrew_dict = {
    'א': 0, 'ב': 1, 'ג': 2, 'ד': 3, 'ה': 4, 'ו': 5, 'ז': 6, 'ח': 7, 'ט': 8, 'י': 9, 'כ': 10, 'ל': 11, 'מ': 12, 'נ': 13,
    'ס': 14, 'ע': 15, 'פ': 16, 'צ': 17, 'ק': 18, 'ר': 19, 'ש': 20, 'ת': 21
}

words_dict = {
    'bye': 0, 'excellent': 1, 'idf': 2, 'ok': 3, 'soldier': 4
}

hebrew_letters_in_english = {
    'ale': 'א', 'bet': 'ב', 'gim': 'ג', 'dal': 'ד', 'hey': 'ה', 'vav': 'ו', 'zay': 'ז', 'het': 'ח', 'tet': 'ט',
    'yud': 'י', 'kaf': 'כ', 'lam': 'ל', 'mem': 'מ', 'nun': 'נ', 'sam': 'ס', 'ain': 'ע', 'pey': 'פ', 'tza': 'צ',
    'kuf': 'ק', 'rey': 'ר', 'shi': 'ש', 'taf': 'ת'
}


def create_df(images_dir: Path, mode: str, _class_encoding: dict = None, weight: int = 1) -> pd.DataFrame:
    """
    Creates a DataFrame for training or testing images.

    :param images_dir: Directory containing images.
    :param mode: 'trn' for training or 'tst' for testing.
    :param _class_encoding: Optional dictionary for class encoding.
    :param weight: Weight to assign to each image.
    :return: DataFrame with image metadata and optionally, a class encoding dictionary.
    """
    df = pd.DataFrame()
    images_list, labels_list, lable_class_encoding, eng_heb_list, weights = [], [], [], [], []

    # Initialize class encoding if not provided
    class_encoding = _class_encoding if _class_encoding else {}

    if mode == 'trn':
        # Process training images
        for subdir in os.listdir(images_dir):
            subdir_path = images_dir / subdir

            if not os.path.isdir(subdir_path):
                continue

            if _class_encoding is None:
                if subdir in words_dict:
                    class_encoding[subdir] = words_dict[subdir]
                else:
                    class_encoding[subdir] = hebrew_dict[hebrew_letters_in_english[subdir]]

            for img_name in os.listdir(subdir_path):
                img_path = subdir_path / img_name
                cur_label = subdir

                images_list.append(str(img_path))
                labels_list.append(cur_label)
                lable_class_encoding.append(class_encoding[cur_label])
                eng_heb_list.append(hebrew_dict.get(hebrew_letters_in_english.get(cur_label)))
                weights.append(weight)

        # Create DataFrame
        df["filename"] = images_list
        df["label"] = labels_list
        df['class_encoding'] = lable_class_encoding
        df['weights'] = weights
        df['hebrew'] = eng_heb_list

    elif mode == 'tst':
        # Process testing images
        for img_name in os.listdir(images_dir):
            img_path = images_dir / img_name
            cur_label = img_name.split('.')[0]  # Assuming filename structure like 'ale.jpg'

            images_list.append(str(img_path))
            labels_list.append(cur_label)
            lable_class_encoding.append(class_encoding[cur_label])
            weights.append(weight)

        # Create DataFrame
        df["filename"] = images_list
        df["label"] = labels_list
        df['class_encoding'] = lable_class_encoding
        df['weights'] = weights

    return df if _class_encoding else (df, class_encoding)


def create_new_dirs(new_dir: str = None) -> Path:
    """
    Creates a new directory for metadata files.

    :param new_dir: Optional subdirectory name.
    :return: Path to the new directory.
    """
    root = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\metadata')
    if not new_dir:
        new_dir = root / f'new_metadata_{datetime.now().strftime("%m%Y%d-%H%M")}'
    else:
        new_dir = root / f'{new_dir}_{datetime.now().strftime("%m%Y%d-%H%M")}'
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def save_df(df: pd.DataFrame, path: Path, name: str):
    """
    Saves a DataFrame to a CSV file.

    :param df: DataFrame to save.
    :param path: Directory to save the file in.
    :param name: Name of the CSV file.
    """
    df.to_csv(path / f'{name}.csv', index=False, encoding='utf-8-sig')
    print(f'Metadata saved in: {path} with the name: {name}.csv')


if __name__ == '__main__':

    # letters dfs
    images_dir = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\processed_images'
                      r'\sivan_example_processed')
    new_dir = create_new_dirs(new_dir='SivanLettersExample')

    trn_df, class_encoding = create_df(images_dir=images_dir / 'trn', mode='trn')
    tst_df = create_df(images_dir=images_dir / 'tst', mode='tst', _class_encoding=class_encoding)
    save_df(df=trn_df, path=new_dir, name='trn_metadata')
    save_df(df=tst_df, path=new_dir, name='tst_metadata')

    # words dfs
    images_dir = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\processed_images'
                      r'\sivan_words_example')
    new_dir = create_new_dirs(new_dir='SivanWordsExample')

    trn_df = create_df(images_dir=images_dir / 'trn', mode='trn', _class_encoding=words_dict)
    tst_df = create_df(images_dir=images_dir / 'tst', mode='tst', _class_encoding=words_dict)
    save_df(df=trn_df, path=new_dir, name='trn_metadata')
    save_df(df=tst_df, path=new_dir, name='tst_metadata')

