import os
from pathlib import Path
from datetime import datetime

from h5py._hl import dataset
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


def create_run_dir(tag, loaded_model,isWords= False):
    root = Path(r'C:\Users\40gil\Desktop\final_project\tensor_training\running_outputs')
    if loaded_model:
        root = root / "loaded_models_outputs"
    else:
        root = root / "train_outputs" / datetime.now().strftime("%d-%m-%Y")
    if isWords:
        root = root / "Words"
    tagdir = root / f'{tag}__Time_{datetime.now().strftime("%H-%M-%S")}'
    tagdir.mkdir(exist_ok=True, parents=True)
    plotsdir = tagdir / 'plots'
    plotsdir.mkdir(exist_ok=True, parents=True)
    return tagdir, plotsdir


def create_tag(loaded_model, bs, ts, lr, epochs, extra_run_tag_str):
    if not loaded_model:
        tag = f'{extra_run_tag_str}__bs{bs}__ts{ts[0]}X{ts[1]}__epochs{epochs}__lr{lr}'  # run tag name
    elif loaded_model:
        test_paths = str(test_path).split('\\')
        test_name = test_paths[len(test_paths) - 1]
        loaded_model_name = loaded_model_dir.split('\\')[len(loaded_model_dir.split('\\')) - 1]
        tag = f'Model={loaded_model_name}__LoadedModel_tests={test_name}'  # run tag
    return tag

def split_with_class_balance(df, test_size, random_state):
    """
    make sure all clases from df are splited into train and test sets
    :param df:
    :param test_size:
    :param random_state:
    :return:
    """
    # Convert the class_encoding column to a string for grouping
    df['class_encoding_str'] = df['class_encoding'].astype(str)

    # Group the DataFrame by class_encoding and split each group
    dfs = [group for _, group in df.groupby('class_encoding_str')]
    train_dfs, test_dfs = [], []
    for df_group in dfs:
        df_train, df_test = train_test_split(df_group, test_size=test_size, random_state=random_state)
        train_dfs.append(df_train)
        test_dfs.append(df_test)

    # Concatenate the splits to get the final train and test DataFrames
    df_train_final = pd.concat(train_dfs)
    df_test_final = pd.concat(test_dfs)

    # Drop the temporary 'class_encoding_str' column
    df_train_final.drop(columns=['class_encoding_str'], inplace=True)
    df_test_final.drop(columns=['class_encoding_str'], inplace=True)

    return df_test_final,df_train_final


def preprocess_metadata(train_metapath, test_metapath, validation_split):
    df_trn = pd.read_csv(train_metapath)  # get metadata
    df_tst = pd.read_csv(test_metapath)  # get metadata
    df_trn.class_encoding = df_trn.class_encoding
    df_trn['class_encoding'] = df_trn['class_encoding'].apply(lambda x: [x])
    df_tst['class_encoding'] = df_tst['class_encoding'].apply(lambda x: [x])
    df_tst_pp = df_tst.copy()
    df_trn_pp = df_trn.copy()  # make df_pp only on train df
    # df_tst_pp, df_val_pp = split_with_class_balance(df_tst, test_size=0.5,
    #                                         random_state=666)  # splits tst and val from df_tst
    df_trn_pp, df_val_pp = train_test_split(df_trn_pp, test_size=validation_split,
                                            random_state=666,stratify=df_trn_pp.class_encoding)  # splits tst and val from df_tst
    # friends_indxs = df_val_pp.filename.str.contains('hebrew_to_english')
    # df_val_new_pp = df_val_pp[friends_indxs]
    # df_new_train_pp = df_val_pp[~friends_indxs]
    # df_trn_pp = pd.concat([df_trn_pp, df_new_train_pp], axis=0)
    return df_trn_pp, df_val_pp, df_tst_pp


def create_gens(train_metapath, test_metapath, ts, validation_split):
    df_trn_pp, df_val_pp, df_tst_pp = preprocess_metadata(train_metapath, test_metapath, validation_split)

    trn_gen = ImageDataGenerator(rotation_range=10, width_shift_range=5,
                                 height_shift_range=5, zoom_range=0.1,
                                 channel_shift_range=10, shear_range=10, horizontal_flip=True).flow_from_dataframe(
        df_trn_pp,
        x_col='filename',
        y_col='class_encoding',
        target_size=ts,
        class_mode='categorical',
        weight_col='weights',
        shuffle=True)
    val_gen = ImageDataGenerator().flow_from_dataframe(df_val_pp, x_col='filename',
                                                       y_col='class_encoding',
                                                       target_size=ts, shuffle=False,
                                                       weight_col='weights',
                                                       class_mode='categorical')
    tst_gen = ImageDataGenerator().flow_from_dataframe(df_tst_pp, x_col='filename',
                                                       y_col='class_encoding',
                                                       target_size=ts, shuffle=False,
                                                       weight_col='weights',
                                                       class_mode='categorical')

    return {
        'trn_gen': trn_gen,
        'val_gen': val_gen,
        'tst_gen': tst_gen
    }, {
        'df_trn_pp': df_trn_pp,
        'df_val_pp': df_val_pp,
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


def save_run_data(fname, model, gen, df, df_pp, plotsdir):
    class_encoding_revers = np.sort(df.label.unique())
    class_encoding_dict = dict(zip(np.arange(26), class_encoding_revers))
    ticks_list = [str(ii) + '(' + jj + ')' for ii, jj in class_encoding_dict.items()]
    raw_pred, pred, acc, conf_mat = get_xset_pred_matrix(model, gen, df_pp)

    save_plot(conf_mat, ticks_list, acc, plotsdir, fname)

    save_run_res_csv(df_pp=df_pp, raw_pred=raw_pred, pred=pred, class_encoding_dict=class_encoding_dict,
                     class_encoding_revers=class_encoding_revers, dir=plotsdir, name=fname)

def save_script(dir,tag='script.py'):
    if tag.endswith('.py'):
        shutil.copy(__file__, plots_dir / tag) # save script
        print (f"{tag} saved in {dir}\n you can change the script now for the next train")
    else:
        raise ValueError("Invalid script tag. Must end with '.py'")



if __name__ == '__main__':
    _loaded_model = False
    is_words_train = True
    loaded_model_dir = r"C:\Users\40gil\Desktop\degree\year_4\sm2\final_project\running_outputs\asl_new_NoWeights_bs=32_ts=(128, 128)_valSplit=0.2_lr=0.001_epochs=120_DateTime=03_03_35\asl_new_NoWeights_bs=32_ts=(128, 128)_valSplit=0.2_lr=0.001_epochs=120.h5"

    # region Paths

    metadata_dir_path = Path(
        r'C:\Users\40gil\Desktop\final_project\tensor_training\metadata\WordsCut__05202421-2101')
    train_path = metadata_dir_path / 'trn_metadata.csv'
    test_path = metadata_dir_path / 'tst_metadata.csv'

    paths = {
        'train_metapath': train_path,
        'test_metapath': test_path
    }
    # endregion

    # region Params
    params = {
        'bs': 32,  # batch size
        'ts': (128, 128),  # target size
        'x_col': 'filename',  # the column in the dataframe that contains the path to the images
        'y_col': 'class_encoding',
        'validation_split': 0.2,  # train validation split
        'lr': 1e-3,
        'epochs': 120,
        'steps': 100,
        'extra_run_tag_str': "Words_JustCut_SmallResnetConv",  # will appear in the beggining of the running dir name
        'loaded_model': _loaded_model
    }


    # endregion

    tag = create_tag(loaded_model=_loaded_model, bs=params['bs'], ts=params['ts'], lr=params['lr'],
                     epochs=params['epochs'], extra_run_tag_str=params['extra_run_tag_str'])

    running_dir, plots_dir = create_run_dir(tag, loaded_model=_loaded_model,isWords=True)

    save_script(dir=plots_dir)
    # region Prepare train

    gens, pp = create_gens(train_metapath=train_path, test_metapath=test_path, ts=params['ts'],
                           validation_split=params['validation_split'])

    fit_dict = {
        'epochs': params['epochs'],
        'steps_per_epoch': int(np.minimum(np.floor(pp['df_trn_pp'].shape[0] / params['bs']), params['steps'])),
        'verbose': 1
    }
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=np.sqrt(0.1),
                                                        min_delta=1e-4, patience=5, min_lr=1e-5, verbose=1, cooldown=3)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1e-4,
                                                    patience=15, verbose=1, baseline=None, restore_best_weights=True)
    callbacks = [reduce_lr_cb, earlystop_cb]

    # endregion

    # region Train
    if _loaded_model:  # use loaded model and don't train
        model = tf.keras.models.load_model(loaded_model_dir)
    else:
        ### NOTICE TO CHANGE N_CLASSES ACCORDING TO NUM OF LETTERS ON TRAIN
        from create_models.create_models import create_small_resnet50_with_conv

        ts = params['ts']
        lr = params['lr']
        trn_gen = gens['trn_gen']
        val_gen = gens['val_gen']
        model = create_small_resnet50_with_conv(input_shape=(ts[0], ts[1], 3), lr=lr, n_classes=len(pp['df_trn_pp']['class_encoding'].explode().unique()))
        model.summary()
        # --- fit model ---
        history = model.fit(trn_gen, validation_data=val_gen, callbacks=callbacks, **fit_dict)
    # endregion

    # -- save model (if new model) --
    save_run_history(history=model.history, dir=plots_dir, to_csv=True, to_json=True)
    if not _loaded_model:
        if is_words_train:
            fn = running_dir / 'model-words.h5'
        else:
            fn = running_dir / 'model-letters.h5'
        model.save(fn)  # saving tensorflow model
        print(f'Model saved to : {fn}')

    save_run_data(fname="val", model=model, gen=gens['val_gen'], df=pd.read_csv(train_path), df_pp=pp['df_val_pp'],
                  plotsdir=plots_dir)
    save_run_data(fname="tst", model=model, gen=gens['tst_gen'], df=pd.read_csv(train_path), df_pp=pp['df_tst_pp'],
                  plotsdir=plots_dir)


    print("DONE!!!!")