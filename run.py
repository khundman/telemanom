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

    stats = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0
    }

    with open("labeled_anomalies.csv", "rU") as f:
        reader = csv.DictReader(f)

        with open("results/%s.csv" %_id, "a") as out:

            writer = csv.DictWriter(out, config.header) # line by line results written to csv
            writer.writeheader()
        
            for i, anom in enumerate(reader):
                if reader.line_num >= 1:

                    anom['run_id'] = _id
                    logger.info("Stream # %s: %s" %(reader.line_num-1, anom['chan_id']))
                    model = None

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
                    anom['scores'] = E_seq_scores

                    anom = err.evaluate_sequences(E_seq, anom)
                    anom["num_values"] = y_test.shape[0] + config.l_s + config.n_predictions

                    for key, value in stats.items():
                        stats[key] += anom[key]

                    helpers.anom_stats(stats, anom, logger)
                    writer.writerow(anom)

    helpers.final_stats(stats, logger)


if __name__ == "__main__":
    config = Config("config.yaml")
    _id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
    helpers.make_dirs(_id)  
    logger = helpers.setup_logging(config, _id)
    run(config, _id, logger)



    