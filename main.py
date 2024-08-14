import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Directory and Tag Creation

def create_run_dir(tag, loaded_model, is_words=False):
    root = Path(r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\running_outputs')
    root = root / "loaded_models_outputs" if loaded_model else root / "train_outputs" / datetime.now().strftime("%d-%m-%Y")
    if is_words:
        root = root / "Words"
    tagdir = root / f'{tag}__Time_{datetime.now().strftime("%H-%M-%S")}'
    tagdir.mkdir(parents=True, exist_ok=True)
    plotsdir = tagdir / 'plots'
    plotsdir.mkdir(parents=True, exist_ok=True)
    return tagdir, plotsdir


def create_tag(loaded_model, bs, ts, lr, epochs, extra_run_tag_str):
    if loaded_model:
        test_name = Path(test_path).name
        loaded_model_name = Path(loaded_model_dir).name
        tag = f'Model={loaded_model_name}__LoadedModel_tests={test_name}'
    else:
        tag = f'{extra_run_tag_str}__bs{bs}__ts{ts[0]}X{ts[1]}__epochs{epochs}__lr{lr}'
    return tag


# Data Preprocessing and Splitting

def split_with_class_balance(df, test_size, random_state):
    df['class_encoding_str'] = df['class_encoding'].astype(str)
    dfs = [group for _, group in df.groupby('class_encoding_str')]

    train_dfs, test_dfs = [], []
    for df_group in dfs:
        df_train, df_test = train_test_split(df_group, test_size=test_size, random_state=random_state)
        train_dfs.append(df_train)
        test_dfs.append(df_test)

    df_train_final = pd.concat(train_dfs).drop(columns=['class_encoding_str'])
    df_test_final = pd.concat(test_dfs).drop(columns=['class_encoding_str'])

    return df_test_final, df_train_final


def preprocess_metadata(train_metapath, test_metapath, validation_split):
    df_trn = pd.read_csv(train_metapath)
    df_tst = pd.read_csv(test_metapath)

    df_trn['class_encoding'] = df_trn['class_encoding'].apply(lambda x: [x])
    df_tst['class_encoding'] = df_tst['class_encoding'].apply(lambda x: [x])

    df_trn_pp, df_val_pp = train_test_split(df_trn, test_size=validation_split, random_state=666,
                                            stratify=df_trn['class_encoding'])

    return df_trn_pp, df_val_pp, df_tst


# Data Generators

def create_gens(train_metapath, test_metapath, ts, validation_split):
    df_trn_pp, df_val_pp, df_tst_pp = preprocess_metadata(train_metapath, test_metapath, validation_split)

    trn_gen = ImageDataGenerator(rotation_range=10, width_shift_range=5, height_shift_range=5,
                                 zoom_range=0.1, channel_shift_range=10, shear_range=10, horizontal_flip=True
                                 ).flow_from_dataframe(df_trn_pp, x_col='filename', y_col='class_encoding',
                                                       target_size=ts, class_mode='categorical',
                                                       weight_col='weights', shuffle=True)

    val_gen = ImageDataGenerator().flow_from_dataframe(df_val_pp, x_col='filename', y_col='class_encoding',
                                                       target_size=ts, class_mode='categorical',
                                                       weight_col='weights', shuffle=False)

    tst_gen = ImageDataGenerator().flow_from_dataframe(df_tst_pp, x_col='filename', y_col='class_encoding',
                                                       target_size=ts, class_mode='categorical',
                                                       weight_col='weights', shuffle=False)

    return {'trn_gen': trn_gen, 'val_gen': val_gen, 'tst_gen': tst_gen}, {'df_trn_pp': df_trn_pp,
                                                                          'df_val_pp': df_val_pp,
                                                                          'df_tst_pp': df_tst_pp}


# Model Training and Evaluation

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
    df_pp['predicted_letter'] = df_pp['prediction'].map(class_encoding_dict)
    df_pp.reset_index(inplace=True)
    for ind_img in range(df_pp.shape[0]):
        for ind_letter, l in enumerate(class_encoding_revers):
            df_pp.loc[ind_img, f'raw_pred_{l}'] = raw_pred[ind_img, ind_letter]
    df_pp.to_csv(dir / f'{name}_res.csv')


def save_run_history(history, dir, name='history', to_json=False, to_csv=True):
    hist_df = pd.DataFrame(history.history)
    if to_json:
        hist_df.to_json(dir / f'{name}.json', orient='records')
    if to_csv:
        hist_df.to_csv(dir / f'{name}.csv', index=False)


def save_run_data(fname, model, gen, df, df_pp, plotsdir):
    class_encoding_revers = np.sort(df.label.unique())
    class_encoding_dict = dict(zip(np.arange(26), class_encoding_revers))
    ticks_list = [str(ii) + '(' + jj + ')' for ii, jj in class_encoding_dict.items()]
    raw_pred, pred, acc, conf_mat = get_xset_pred_matrix(model, gen, df_pp)
    save_plot(conf_mat, ticks_list, acc, plotsdir, fname)
    save_run_res_csv(df_pp=df_pp, raw_pred=raw_pred, pred=pred, class_encoding_dict=class_encoding_dict,
                     class_encoding_revers=class_encoding_revers, dir=plotsdir, name=fname)


def save_script(dir, tag='script.py'):
    shutil.copy(__file__, dir / tag)
    print(f"{tag} saved in {dir}")


# Main Execution

if __name__ == '__main__':
    _loaded_model = False
    is_words_train = False
    loaded_model_dir = r"C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\running_outputs\train_outputs\27-05-2024\Letters_JustCut_NoResnetLayers__bs32__ts128X128__epochs120__lr0.001__Time_16-10-26\model-letters.h5"

    metadata_dir_path = Path(
        r'C:\Users\40gil\Desktop\AltDegree\final_project\tensor_training\metadata\SivanLettersExample_08202414-1103')
    train_path = metadata_dir_path / 'trn_metadata.csv'
    test_path = metadata_dir_path / 'tst_metadata.csv'

    params = {
        'bs': 32,
        'ts': (128, 128),
        'x_col': 'filename',
        'y_col': 'class_encoding',
        'validation_split': 0.2,
        'lr': 1e-3,
        'epochs': 120,
        'steps': 100,
        'extra_run_tag_str': "Letters_JustCut_ResnetLayers",
        'loaded_model': _loaded_model
    }
    tag = create_tag(loaded_model=_loaded_model, bs=params['bs'], ts=params['ts'], lr=params['lr'],
                     epochs=params['epochs'], extra_run_tag_str=params['extra_run_tag_str'])

    running_dir, plots_dir = create_run_dir(tag, loaded_model=_loaded_model, is_words=is_words_train)

    save_script(dir=plots_dir)

    # Prepare data generators
    gens, pp = create_gens(train_metapath=train_path, test_metapath=test_path, ts=params['ts'],
                           validation_split=params['validation_split'])

    fit_dict = {
        'epochs': params['epochs'],
        'steps_per_epoch': int(np.minimum(np.floor(pp['df_trn_pp'].shape[0] / params['bs']), params['steps'])),
        'verbose': 1
    }

    # Callbacks for model training
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=np.sqrt(0.1),
                                                        min_delta=1e-4, patience=5, min_lr=1e-5, verbose=1, cooldown=3)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1e-4,
                                                    patience=15, verbose=1, baseline=None, restore_best_weights=True)
    callbacks = [reduce_lr_cb, earlystop_cb]

    # Train or load model
    if _loaded_model:
        model = tf.keras.models.load_model(loaded_model_dir)
    else:
        from create_models.create_models import create_resnet50

        model = create_resnet50(input_shape=(params['ts'][0], params['ts'][1], 3), lr=params['lr'],
                                n_classes=len(pp['df_trn_pp']['class_encoding'].explode().unique()))
        model.summary()
        history = model.fit(gens['trn_gen'], validation_data=gens['val_gen'], callbacks=callbacks, **fit_dict)

    # Save the model
    if not _loaded_model:
        model_fn = 'model-words.h5' if is_words_train else 'model-letters.h5'
        model.save(running_dir / model_fn)
        print(f'Model saved to : {running_dir / model_fn}')

    # Save validation and test data predictions
    save_run_data(fname="val", model=model, gen=gens['val_gen'], df=pd.read_csv(train_path), df_pp=pp['df_val_pp'],
                  plotsdir=plots_dir)
    save_run_data(fname="tst", model=model, gen=gens['tst_gen'], df=pd.read_csv(train_path), df_pp=pp['df_tst_pp'],
                  plotsdir=plots_dir)

    print("DONE!!!!")

