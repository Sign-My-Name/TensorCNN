import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re


def create_run_dir(model_name, model_date, isWords=False):
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze')
    root = root / "loaded_models_outputs"
    if isWords:
        root = root / f"words"
    else:
        root = root / f"letters"
    root = root / model_date / model_name
    root.mkdir(exist_ok=True, parents=True)
    return root


def preprocess_metadata(test_metapath):
    df_tst = pd.read_csv(test_metapath)  # get metadata
    df_tst['class_encoding'] = df_tst['class_encoding'].apply(lambda x: [x])
    df_tst_pp = df_tst.copy()
    return df_tst_pp


def create_gens(test_metapath, ts):
    df_tst_pp = preprocess_metadata(test_metapath)
    tst_gen = ImageDataGenerator().flow_from_dataframe(df_tst_pp, x_col='filename',
                                                       y_col='class_encoding',
                                                       target_size=ts, shuffle=False,
                                                       weight_col='weights',
                                                       class_mode='categorical')

    return {
        'tst_gen': tst_gen,
        'df_tst_pp': df_tst_pp
    }


def get_xset_pred_matrix(model, gen, df_pp):
    pred_raw = model.predict(gen)
    pred = pred_raw.argmax(axis=1)
    labels = np.array([ii[0] for ii in df_pp.class_encoding])
    acc = sum(pred == labels) / len(labels)
    conf_mat = confusion_matrix(labels, pred)
    return pred_raw, pred, acc, conf_mat


def save_plot(conf_mat, ticks_list, acc, dir, name):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    sns.heatmap(conf_mat, annot=True, ax=ax)
    ax.set_xticklabels(ticks_list, rotation=45)
    ax.set_yticklabels(ticks_list, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.title(f'acc={acc}')
    plt.savefig(dir / f'{name}.png')
    with open(dir / 'acc.txt', 'w') as f:
        f.write(str(acc))


def save_run_res_csv(df_pp, raw_pred, pred, class_encoding_dict, class_encoding_revers, dir, name):
    df_pp['prediction'] = pred
    df_pp['predicted_letter'] = df_pp.prediction.map(class_encoding_dict)
    df_pp.reset_index(inplace=True)
    for ind_img in range(df_pp.shape[0]):
        for ind_letter, l in enumerate(class_encoding_revers):
            try:
                df_pp.loc[ind_img, 'raw_pred_' + l] = raw_pred[ind_img, ind_letter]
            except IndexError:
                continue

    df_pp.to_csv(dir / f'{name}_res.csv')


def save_run_history(history, dir, name='history', to_json=False, to_csv=True):
    hist_df = pd.DataFrame(history.history)

    if to_json:
        hist_json_file = f'{name}.json'
        hist_json_path = os.path.join(dir, hist_json_file)
        hist_df.to_json(hist_json_path, orient='records')

    if to_csv:
        hist_csv_file = f'{name}.csv'
        hist_csv_path = os.path.join(dir, hist_csv_file)
        hist_df.to_csv(hist_csv_path, index=False)


def save_run_data(fname, model, gen, df, df_pp, rootdir):
    df = df.sort_values('class_encoding')
    class_encoding_revers = df.label.unique()
    class_encoding_dict = dict(zip(np.arange(22), class_encoding_revers))
    ticks_list = [str(ii) + '(' + jj + ')' for ii, jj in class_encoding_dict.items()]

    raw_pred, pred, acc, conf_mat = get_xset_pred_matrix(model, gen, df_pp)
    save_plot(conf_mat, ticks_list, acc, rootdir, fname)

    save_run_res_csv(df_pp=df_pp, raw_pred=raw_pred, pred=pred, class_encoding_dict=class_encoding_dict,
                     class_encoding_revers=class_encoding_revers, dir=rootdir, name=fname)


def get_all_models(models_dir_path):
    """
    :param models_dir_path: all models parent dir where sub dirs are dirs named after the date
    :return: models_metadta = {<date> : {<model name>: <full path of model.h5>, ...}}
    """
    models_metadata = {}
    for subdir in os.listdir(models_dir_path):
        models_metadata[subdir] = []
        curr_date_dir = f"{models_dir_path}\\{subdir}"
        for model_dir in os.listdir(curr_date_dir):
            curr_model_dir = f"{curr_date_dir}\\{model_dir}"
            if model_dir == "Words":
                continue
            for file in os.listdir(curr_model_dir):
                if file.endswith(".h5"):
                    models_metadata[subdir].append({model_dir: f"{curr_model_dir}\\{file}"})
                    continue
    return models_metadata

def run_all_models_on_test(models_parent_dir_path, metadata_parent_path):
    models = get_all_models(models_dir_path=models_parent_dir_path)
    metadata_dir_path = Path(metadata_parent_path)
    test_path = metadata_dir_path / 'tst_metadata.csv'

    ts = ()
    for date in models:
        loaded_model_date = date
        for model_path in models[date]:
            for key, val in model_path.items():
                loaded_model_name = ''.join(val.split('\\')[-2].split('__')[:-1])
                tag = loaded_model_name  # run tag
                loaded_model_dir = val
                match = re.search(r'ts(\d+)X(\d+)', val)
                if match:
                    ts = (int(match.group(1)), int(match.group(2)))
                else:
                    print(f"no size was found in: \n\t{loaded_model_dir}")
                    continue
                root_dir = create_run_dir(model_name=tag, model_date=loaded_model_date)
                gens = create_gens(test_metapath=test_path, ts=ts)
                model = tf.keras.models.load_model(loaded_model_dir)
                save_run_data(fname="tst", model=model, gen=gens['tst_gen'], df=pd.read_csv(test_path),
                              df_pp=gens['df_tst_pp'],
                              rootdir=root_dir)
if __name__ == '__main__':
    # run_all_models_on_test(models_parent_dir_path=r"C:\Users\40gil\Desktop\final_project\tensor_training"
    #                                               r"\running_outputs\train_outputs",
    #                        metadata_parent_path=r'C:\Users\40gil\Desktop\final_project\tensor_training\metadata'
    #                                             r'\FriendsCut__06202403-1729')
    print("Done!")

