import numpy as np
import os
from telemanom._globals import Config
import logging
from datetime import datetime
import sys
import csv
import pandas as pd
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode
import cufflinks as cf
import glob

config = Config("config.yaml")

def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

    if not config.train or not config.predict:
        if not os.path.isdir('data/%s' %config.use_id):
            raise ValueError("Run ID %s is not valid. If loading prior models or predictions, must provide valid ID.")

    paths = ['data', 'results', 'data/%s' %_id, 'data/%s/models' %_id, 'data/%s/smoothed_errors' %_id, 'data/%s/y_hat' %_id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

def setup_logging(config, _id):
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''

    logger =  logging.getLogger('telemanom')
    hdlr = logging.FileHandler('data/%s/params.log' %_id)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    logger.info("Runtime params:")
    logger.info("----------------")
    for attr in dir(config):    
        if not "__" in attr and not attr in ['header', 'date_format', 'path_to_config', 'build_group_lookup']:
            logger.info('%s: %s' %(attr, getattr(config, attr)))
    logger.info("----------------\n")

    return logger



def load_data(anom):
    '''Load train and test data from repo. If not in repo need to download from source.

    Args:
        anom (dict): contains anomaly information for a given input stream

    Returns:
        X_train (np array): array of train inputs with dimensions [timesteps, l_s, input dimensions]
        y_train (np array): array of train outputs corresponding to true values following each sequence
        X_test (np array): array of test inputs with dimensions [timesteps, l_s, input dimensions)
        y_test (np array): array of test outputs corresponding to true values following each sequence
    '''
    try:
        train = np.load(os.path.join("data", "train", anom['chan_id'] + ".npy"))
        test = np.load(os.path.join("data", "test", anom['chan_id'] + ".npy"))

    except:
        raise ValueError("Source data not found, may need to add data to repo: <link>")

    # shape, split data
    X_train, y_train = shape_data(train)
    X_test, y_test = shape_data(test, train=False)

    return X_train, y_train, X_test, y_test


def shape_data(arr, train=True):
    '''Shape raw input streams for ingestion into LSTM. config.l_s specifies the sequence length of 
    prior timesteps fed into the model at each timestep t. 

    Args:
        arr (np array): array of input streams with dimensions [timesteps, 1, input dimensions]
        train (bool): If shaping training data, this indicates data can be shuffled

    Returns:
        X (np array): array of inputs with dimensions [timesteps, l_s, input dimensions)
        y (np array): array of outputs corresponding to true values following each sequence. 
            shape = [timesteps, n_predictions, 1)
        l_s (int): sequence length to be passed to test shaping (if shaping train) so they are consistent
    '''

    data = [] 
    for i in range(len(arr) - config.l_s - config.n_predictions):
        data.append(arr[i:i + config.l_s + config.n_predictions])
    data = np.array(data) 

    assert len(data.shape) == 3

    if train == True:
        np.random.shuffle(data)

    X = data[:,:-config.n_predictions,:]
    y = data[:,-config.n_predictions:,0] #telemetry value is at position 0

    return X, y
    

def final_stats(stats, logger):
    '''Log final stats at end of experiment.

    Args:
        stats (dict): Count of true positives, false positives, and false negatives from experiment
        logger (obj): logging object
    '''

    logger.info("Final Totals:")
    logger.info("-----------------")
    logger.info("True Positives: %s " %stats["true_positives"])
    logger.info("False Positives: %s " %stats["false_positives"])
    logger.info("False Negatives: %s\n" %stats["false_negatives"])
    try:
        logger.info("Precision: %s" %(float(stats["true_positives"])/float(stats["true_positives"]+stats["false_positives"])))
        logger.info("Recall: %s" %(float(stats["true_positives"])/float(stats["true_positives"]+stats["false_negatives"])))
    except:
        logger.info("Precision: NaN")
        logger.info("Recall: NaN")


def anom_stats(stats, anom, logger):
    '''Log stats after processing of each stream.

    Args:
        stats (dict): Count of true positives, false positives, and false negatives from experiment
        anom (dict): contains all anomaly information for a given input stream
        logger (obj): logging object
    '''

    logger.info("TP: %s  FP: %s  FN: %s" %(anom["true_positives"], anom["false_positives"], anom["false_negatives"]))
    logger.info('Total true positives: %s' %stats["true_positives"])
    logger.info('Total false positives: %s' %stats["false_positives"])
    logger.info('Total false negatives: %s\n' %stats["false_negatives"])



def view_results(results_fn, plot_errors=True, plot_train=False, rows=None):
    ''' Reads data from data dir and generates interactive plots for display in `results-viewer.ipynb` using 
    plotly offline mode. A chart showing y_hat and y_test values for each stream is generated by default. 

    Args:
        results_fn (str): name of results csv to plot results for
        plot_errors (bool): If True, a chart displaying the smoothed errors for each stream will be generated
        plot_train (bool): If True, a chart displaying the telemetry from training data is 
            be generated (command data not plotted)
        rows (tuple): Start and end row indicating rows to plot results for in results csv file

    Returns:
        None
    '''

    def create_shapes(ranges, range_type, _min, _max):
        ''' Create shapes for regions to highlight in plotly vizzes (true and predicted anomaly sequences)'''

        if range_type == 'true':
            color = 'red'
        elif range_type == 'predicted':
            color = 'blue'
        
        shapes = []
        if len(ranges) > 0:
        
            for r in ranges:

                shape = {
                    'type': 'rect',
                    'x0': r[0],
                    'y0': _min,
                    'x1': r[1],
                    'y1': _max,
                    'fillcolor': color,
                    'opacity': 0.2,
                    'line': {
                        'width': 0,
                    },
                }
            
                shapes.append(shape)
            
        return shapes



    vals = {}

    with open(results_fn, "r") as f:
        reader = csv.DictReader(f)
        for anom in reader:

            chan = anom["chan_id"]
            vals[chan] = {}
            dirs = ["y_hat", "smoothed_errors"]
            raw_dirs = ["test", "train"]

            for d in dirs:
                if config.predict:
                    vals[chan][d] = list(np.load(os.path.join("../data", anom['run_id'], d, anom["chan_id"]) + ".npy"))
                else:
                    vals[chan][d] = list(np.load(os.path.join("../data", config.use_id, d, anom["chan_id"]) + ".npy"))
            for d in raw_dirs:
                vals[chan][d] = list(np.load(os.path.join("../data", d, anom["chan_id"]) + ".npy"))

            row_start = 0
            row_end = 100000
            if not rows == None:
                try:
                    row_start = rows[0]
                    row_end = rows[1]
                except:
                    raise ValueError("Rows not in correct format, please use (<first row>, <last row>)")

            # Info
            # ================================================================================================
            if reader.line_num - 1 >= row_start and reader.line_num -1 <= row_end:
                print("Spacecraft: %s" %anom['spacecraft'])
                print("Channel: %s" %anom["chan_id"])
                print('Normalized prediction error: %.3f' %float(anom['normalized_error']))
                print('Anomaly class(es): %s' %anom['class'])
                print("------------------")
                print('True Positives: %s' %anom['true_positives'])
                print("False Positives: %s" %anom["false_positives"])
                print("False Negatives: %s" %anom["false_negatives"])
                print("------------------")
                print('Predicted anomaly scores: %s' %anom['scores'])
                print("Number of values: %s"%len(vals[chan]["test"]))

                # Extract telemetry values from test data
                # ================================================================================================

                y_test = np.array(vals[chan]['test'])[:,0] 

                # Create highlighted regions (red = true anoms / blue = predicted anoms)
                # ================================================================================================
                y_shapes = create_shapes(eval(anom['anomaly_sequences']), "true", -1, 1)
                y_shapes += create_shapes(eval(anom['tp_sequences']) + eval(anom['fp_sequences']), "predicted", -1, 1)

                e_shapes = create_shapes(eval(anom['anomaly_sequences']), "true", 0, max(vals[chan]['smoothed_errors']))
                e_shapes += create_shapes(eval(anom['tp_sequences']) + eval(anom['fp_sequences']), "predicted", 
                                          0, max(vals[chan]['smoothed_errors']))

                # Move data into dataframes and plot with Plotly
                # ================================================================================================
                train_df = pd.DataFrame({
                    'train': [x[0] for x in vals[chan]['train']]
                })

                y = y_test[config.l_s:-config.n_predictions]
                if not len(y) == len(vals[chan]['y_hat']):
                    modified_l_s = len(y_test) - len(vals[chan]['y_hat']) - 1
                    y = y_test[modified_l_s:-1]
                y_df = pd.DataFrame({
                    'y_hat': vals[chan]['y_hat'],
                    'y': y
                })

                e_df = pd.DataFrame({
                    'e_s': vals[chan]['smoothed_errors']
                })

                y_layout = {
                    'title': "y / y_hat comparison",
                    'shapes': y_shapes,
                } 

                e_layout = {
                    'title': "Smoothed Errors (e_s)",
                    'shapes': e_shapes,
                } 

                if plot_train:
                    train_df.iplot(kind='scatter', color='green')
                
                y_df.iplot(kind='scatter', layout=y_layout)
                
                if plot_errors:
                    e_df.iplot(kind='scatter', layout=e_layout, color='red')



