# -*- coding: utf-8 -*-
import sys
sys.path.append('../hse_workshop_classification-main/src')
sys.path.append('../hse_workshop_classification-main/models')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import *
from data.preprocess import *
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import config as cfg
import json 
from statistics import mean


@click.command()
@click.argument('train_data_filepath', type=click.Path(exists=True)) 
@click.argument('target_data_filepath', type=click.Path(exists=True)) 
@click.argument('model_name') # catboost
@click.argument('output_model_filepath', type=click.Path()) 
def main(train_data_filepath, target_data_filepath, model_name, output_model_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('training model on final data')

    train = pd.read_pickle(train_data_filepath)
    target = pd.read_pickle(target_data_filepath)

    metrics = {}

    result = []

    X_train, X_val, y_train, y_val = train_test_split(train, target, train_size=0.8, random_state=42)

    if model_name == 'catboost':
        for col in y_train.columns:
            model = CatBoostClassifier(learning_rate=0.001,early_stopping_rounds=200, verbose=100, auto_class_weights='Balanced', eval_metric='F1')
            model.fit(X_train, y_train[col], cat_features=cfg.CAT_COLS, eval_set=(X_val, y_val[col]))
            save_model(model, output_model_filepath + '/catboost_' + col + '.sav')

            y_pred = model.predict(X_val)
            result.append([precision_score(y_val[col], y_pred), recall_score(y_val[col], y_pred), f1_score(y_val[col], y_pred)])


    metrics['precision'] = mean([metrics[0] for metrics in result])
    metrics['recall'] = mean([metrics[1] for metrics in result])
    metrics['f1'] = mean([metrics[2] for metrics in result])

    with open("metrics.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
