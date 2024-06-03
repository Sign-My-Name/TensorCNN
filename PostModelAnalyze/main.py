import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def create_run_dir(model_name, model_date, isWords=False):
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze')
    root = root / "loaded_models_outputs"
    if isWords:
        root = root / f"words"
    else:
        root = root / f"letters"
    root = root / model_date / model_name
    plotsdir = root / 'plots'
    plotsdir.mkdir(exist_ok=True, parents=True)
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


def save_run_res_csv(df_pp, raw_pred, pred, class_encoding_dict, class_encoding_revers, dir, name):
    df_pp['prediction'] = pred
    df_pp['predicted_letter'] = df_pp.prediction.map(class_encoding_dict)
    df_pp.reset_index(inplace=True)
    for ind_img in range(df_pp.shape[0]):
        for ind_letter, l in enumerate(class_encoding_revers):
            df_pp.loc[ind_img, 'raw_pred_' + l] = raw_pred[ind_img, ind_letter]
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
    plotsdir = Path(rootdir) / 'plots'
    save_plot(conf_mat, ticks_list, acc, plotsdir, fname)

    save_run_res_csv(df_pp=df_pp, raw_pred=raw_pred, pred=pred, class_encoding_dict=class_encoding_dict,
                     class_encoding_revers=class_encoding_revers, dir=plotsdir, name=fname)


if __name__ == '__main__':
    is_words = False
    loaded_model_dir = (r"C:\Users\40gil\Desktop\final_project\tensor_training\running_outputs\train_outputs\27-05"
                        r"-2024\Letters_JustCut_NoResnetLayers__bs32__ts128X128__epochs120__lr0.001__Time_16-10-26"
                        r"\model-letters.h5")

    metadata_dir_path = Path(
        r'C:\Users\40gil\Desktop\final_project\tensor_training\metadata\Sivan_05202426-1211')
    test_path = metadata_dir_path / 'tst_metadata.csv'

    ts = (128, 128)  # target size

    loaded_model_splits = loaded_model_dir.split('\\')
    loaded_model_name = loaded_model_splits[len(loaded_model_splits) - 2]
    loaded_model_date = loaded_model_splits[len(loaded_model_splits) - 3]
    tag = loaded_model_name  # run tag

    root_dir = create_run_dir(model_name=tag, model_date=loaded_model_date, isWords=is_words)

    gens = create_gens(test_metapath=test_path, ts=ts)

    model = tf.keras.models.load_model(loaded_model_dir)

    save_run_data(fname="tst", model=model, gen=gens['tst_gen'], df=pd.read_csv(test_path), df_pp=gens['df_tst_pp'],
                  rootdir=root_dir)

    print("DONE!!!!")
