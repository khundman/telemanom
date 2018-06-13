import sys
import operator
import os
import numpy as np
import pandas as pd
import time
import json
from operator import itemgetter
import csv
import scipy.stats as stats
from itertools import groupby
from operator import itemgetter
from datetime import datetime as dt

from telemanom._globals import Config
import telemanom.errors as err
import telemanom.helpers as helpers
import telemanom.modeling as models


def run(config, _id, logger):
    ''' Top-level function for running experiment.

    Args:
        config (dict): Parameters for modeling, execution levels, and error calculations loaded from config.yaml
        _id (str): Unique id for each processing run generated from current time
        logger (obj): Logger obj from logging module

    Returns:
        None

    '''

    with open("results/%s.csv" %_id, "a") as out:

        writer = csv.DictWriter(out, config.header) # line by line results written to csv
        writer.writeheader()

        path_to_chan_data = 'data/train/' # MOVE TO CONFIG
        chans = [x[:-4] for x in os.listdir(path_to_chan_data) if x[-4:] == '.npy']
        print("Chans being evaluated: %s" %chans)
    
        for i, chan in enumerate(chans):
            if i >= 0:

                anom = {}
                anom['chan_id'] = chan
                anom['run_id'] = _id
                logger.info("Chan: %s (%s of %s)" %(chan, i, len(chans)))

                X_train, y_train, X_test, y_test = helpers.load_data(anom)
                
                # Generate or load predictions
                # ===============================
                y_hat = []
                if config.predict:
                    model = models.get_model(anom, X_train, y_train, logger, train=config.train)
                    y_hat = models.predict_in_batches(y_test, X_test, model, anom)
                        
                else:
                    y_hat = [float(x) for x in list(np.load(os.path.join("data", config.use_id, "y_hat", anom["chan_id"] + ".npy")))]

                # Error calculations
                # ====================================================================================================
                e = err.get_errors(y_test, y_hat, anom, smoothed=False)
                e_s = err.get_errors(y_test, y_hat, anom, smoothed=True)

                anom["normalized_error"] = np.mean(e) / np.ptp(y_test)
                logger.info("normalized prediction error: %s" %anom["normalized_error"])

                # Error processing (batch)
                # =========================

                E_seq, E_seq_scores = err.process_errors(y_test, y_hat, e_s, anom, logger)
                
                anom['anomaly_sequences'] = E_seq
                anom['scores'] = E_seq_scores
                anom['num_anoms'] = len(anom['anomaly_sequences'])
                anom["num_values"] = y_test.shape[0]
                
                writer.writerow(anom)



if __name__ == "__main__":
    config = Config("config.yaml")
    _id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
    helpers.make_dirs(_id)  
    logger = helpers.setup_logging(config, _id)
    run(config, _id, logger)



    