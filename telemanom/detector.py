import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging

from telemanom.helpers import Config
from telemanom.errors import Errors
import telemanom.helpers as helpers
from telemanom.channel import Channel
from telemanom.modeling import Model

logger = helpers.setup_logging()

class Detector:
    def __init__(self, labels_path=None, result_path='results/',
                 config_path='config.yaml'):
        """
        Top-level class for running anomaly detection over a group of channels
        with values stored in .npy files. Also evaluates performance against a
        set of labels if provided.

        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml

        Attributes:
            labels_path (str): see Args
            results (list of dicts): holds dicts of results for each channel
            result_df (dataframe): results converted to pandas dataframe
            chan_df (dataframe): holds all channel information from labels .csv
            result_tracker (dict): if labels provided, holds results throughout
                processing for logging
            config (obj):  Channel class object containing train/test data
                for X,y for a single channel
            y_hat (arr): predicted channel values
            id (str): datetime id for tracking different runs
            result_path (str): see Args
        """

        self.labels_path = labels_path
        self.results = []
        self.result_df = None
        self.chan_df = None

        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

        self.config = Config(config_path)
        self.y_hat = None

        if not self.config.predict and self.config.use_id:
            self.id = self.config.use_id
        else:
            self.id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')

        helpers.make_dirs(self.id)

        # add logging FileHandler based on ID
        hdlr = logging.FileHandler('data/logs/%s.log' % self.id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        self.result_path = result_path

        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = [x.split('.')[0] for x in os.listdir('data/test/')]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})

        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))

    def evaluate_sequences(self, errors, label_row):
        """
        Compare identified anomalous sequences with labeled anomalous sequences.

        Args:
            errors (obj): Errors class object containing detected anomaly
                sequences for a channel
            label_row (pandas Series): Contains labels and true anomaly details
                for a channel

        Returns:
            result_row (dict): anomaly detection accuracy and results
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores

        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']

        else:
            true_indices_grouped = [list(range(e[0], e[1]+1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:

                    result_row['tp_sequences'].append(e_seq)

                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1

                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'],
                                                          matched_true_seqs, axis=0))

        logger.info('Channel Stats: TP: {}  FP: {}  FN: {}'.format(result_row['true_positives'],
                                                                   result_row['false_positives'],
                                                                   result_row['false_negatives']))

        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row

    def log_final_stats(self):
        """
        Log final stats at end of experiment.
        """

        if self.labels_path:

            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('True Positives: {}'
                        .format(self.result_tracker['true_positives']))
            logger.info('False Positives: {}'
                        .format(self.result_tracker['false_positives']))
            logger.info('False Negatives: {}\n'
                        .format(self.result_tracker['false_negatives']))
            try:
                logger.info('Precision: {0:.2f}'
                            .format(float(self.result_tracker['true_positives'])
                                    / float(self.result_tracker['true_positives']
                                            + self.result_tracker['false_positives'])))
                logger.info('Recall: {0:.2f}'
                            .format(float(self.result_tracker['true_positives'])
                                    / float(self.result_tracker['true_positives']
                                            + self.result_tracker['false_negatives'])))
            except ZeroDivisionError:
                logger.info('Precision: NaN')
                logger.info('Recall: NaN')

        else:
            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('Total channel sets evaluated: {}'
                        .format(len(self.result_df)))
            logger.info('Total anomalies found: {}'
                        .format(self.result_df['n_predicted_anoms'].sum()))
            logger.info('Avg normalized prediction error: {}'
                        .format(self.result_df['normalized_pred_error'].mean()))
            logger.info('Total number of values evaluated: {}'
                        .format(self.result_df['num_test_values'].sum()))


    def run(self):
        """
        Initiate processing for all channels.
        """
        for i, row in self.chan_df.iterrows():
            logger.info('Stream # {}: {}'.format(i+1, row.chan_id))
            channel = Channel(self.config, row.chan_id)
            channel.load_data()

            if self.config.predict:
                model = Model(self.config, self.id, channel)
                channel = model.batch_predict(channel)
            else:
                channel.y_hat = np.load(os.path.join('data', self.id, 'y_hat',
                                                     '{}.npy'
                                                     .format(channel.id)))

            errors = Errors(channel, self.config, self.id)
            errors.process_batches(channel)

            result_row = {
                'run_id': self.id,
                'chan_id': row.chan_id,
                'num_train_values': len(channel.X_train),
                'num_test_values': len(channel.X_test),
                'n_predicted_anoms': len(errors.E_seq),
                'normalized_pred_error': errors.normalized,
                'anom_scores': errors.anom_scores
            }

            if self.labels_path:
                result_row = {**result_row,
                              **self.evaluate_sequences(errors, row)}
                result_row['spacecraft'] = row['spacecraft']
                result_row['anomaly_sequences'] = row['anomaly_sequences']
                result_row['class'] = row['class']
                self.results.append(result_row)

                logger.info('Total true positives: {}'
                            .format(self.result_tracker['true_positives']))
                logger.info('Total false positives: {}'
                            .format(self.result_tracker['false_positives']))
                logger.info('Total false negatives: {}\n'
                            .format(self.result_tracker['false_negatives']))

            else:
                result_row['anomaly_sequences'] = errors.E_seq
                self.results.append(result_row)

                logger.info('{} anomalies found'
                            .format(result_row['n_predicted_anoms']))
                logger.info('anomaly sequences start/end indices: {}'
                            .format(result_row['anomaly_sequences']))
                logger.info('number of test values: {}'
                            .format(result_row['num_test_values']))
                logger.info('anomaly scores: {}\n'
                            .format(result_row['anom_scores']))

            self.result_df = pd.DataFrame(self.results)
            self.result_df.to_csv(
                os.path.join(self.result_path, '{}.csv'.format(self.id)),
                index=False)

        self.log_final_stats()