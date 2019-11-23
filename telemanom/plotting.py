import numpy as np
import os
import pandas as pd
import sys
from telemanom.helpers import Config
sys.path.append('..')

class Plotter:
    def __init__(self, run_id, config_path='config.yaml'):
        """
        For plotting results in Jupyter notebook.

        Args:
            run_id (str): Datetime referencing set of predictions in use
            config_path (str): path to config.yaml file

        Attributes:
            config (obj): Config object containing parameters for processing
            run_id (str): see Args
            result_df (dataframe): holds anomaly detection results for each
                channel
            labels_available (bool): True if labeled anomalous ranges provided
                else False
        """

        self.config = Config(config_path)
        self.run_id = run_id
        self.result_df = pd.read_csv(os.path.join('..', 'results', '{}.csv'
                                                  .format(self.run_id)))
        self.labels_available = True if 'true_positives' \
                                        in self.result_df.columns else False

        self.plot_values = {}

    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values):
        """
        Create shapes for regions to highlight in plotly vizzes (true and
        predicted anomaly sequences). Will plot labeled anomalous ranges if
        available.

        Args:
            ranges (list of tuples): tuple of start and end indices for anomaly
                sequences for a channel
            sequence_type (str): "predict" if predicted values else
                "true" if actual values. Determines colors.
            _min (float): min y value of series
            _max (float): max y value of series
            plot_values (dict): dictionary of different series to be plotted
                (predicted, actual, errors, training data)

        Returns:
            (dict) shape specifications for plotly
        """

        if not _max:
            _max = max(plot_values['smoothed_errors'])

        color = 'red' if sequence_type == 'true' else 'blue'
        shapes = []

        for r in ranges:
            shape = {
                'type': 'rect',
                'x0': r[0] - self.config.l_s,
                'y0': _min,
                'x1': r[1] - self.config.l_s,
                'y1': _max,
                'fillcolor': color,
                'opacity': 0.2,
                'line': {
                    'width': 0,
                }
            }
            shapes.append(shape)

        return shapes

    def all_result_summary(self):
        """
        Print aggregated results.
        """

        if self.labels_available:
            print('True Positives: {}'
                  .format(self.result_df['true_positives'].sum()))
            print('False Positives: {}'
                  .format(self.result_df['false_positives'].sum()))
            print('False Negatives: {}\n'
                  .format(self.result_df['false_negatives'].sum()))
            try:
                print('Precision: {0:.2f}'.format(
                    float(self.result_df['true_positives'].sum()) /
                    float(self.result_df['true_positives'].sum()
                          + self.result_df['false_positives'].sum())))
                print('Recall: {0:.2f}'.format(
                    float(self.result_df['true_positives'].sum()) /
                    float(self.result_df['true_positives'].sum()
                        + self.result_df['false_negatives'].sum())))
            except ZeroDivisionError:
                print('Precision: NaN')
                print('Recall: NaN')
        else:
            print('Total channel sets evaluated: {}'
                  .format(len(self.result_df)))
            print('Total anomalies found: {}'
                  .format(self.result_df['n_predicted_anoms'].sum()))
            print('Avg normalized prediction error: {}'
                  .format(self.result_df['normalized_pred_error'].mean()))
            print('Total number of values evaluated: {}'
                  .format(self.result_df['num_test_values'].sum()))

    def channel_result_summary(self, channel, plot_values):
        """
        Print results for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            plot_values (dict): dictionary of different series to be plotted
                (predicted, actual, errors, training data)
        """

        if 'spacecraft' in channel:
            print('Spacecraft: {}'.format(channel['spacecraft'].values[0]))

        print('Channel: {}'.format(channel['chan_id'].values[0]))
        print('Normalized prediction error: {0:.2f}'
              .format(float(channel['normalized_pred_error'].values[0])))

        if self.labels_available:
            print('Anomaly class(es): {}'.format(channel['class'].values[0]))
            print("------------------")
            print('True Positives: {}'
                  .format(channel['true_positives'].values[0]))
            print('False Positives: {}'
                  .format(channel['false_positives'].values[0]))
            print('False Negatives: {}'
                  .format(channel['false_negatives'].values[0]))
            print('------------------')
        else:
            print('Number of anomalies found: {}'
                  .format(channel['n_predicted_anoms'].values[0]))
            print('Anomaly sequences start/end indices: {}'
                  .format(channel['anomaly_sequences'].values[0]))

        print('Predicted anomaly scores: {}'.format(channel['anom_scores']
                                                    .values[0]))
        print('Number of values: {}'.format(len(plot_values['test'])))

    def plot_channel(self, channel_id, plot_train=False, plot_errors=True):
        """
        Generate interactive plots for a channel. By default it prints actual
        and predicted telemetry values.

        Args:
            channel_id (str): channel id
            plot_train (bool): If true, plot training data in separate plot
            plot_errors (bool): If true, plot prediction errors in separate plot
        """
        channel = self.result_df[self.result_df['chan_id'] == channel_id]

        plot_values = {
            'y_hat': np.load(os.path.join('..', 'data', self.run_id, 'y_hat',
                                          '{}.npy'.format(channel_id))),
            'smoothed_errors': np.load(os.path.join('..', 'data', self.run_id,
                                                    'smoothed_errors',
                                                    '{}.npy'.format(channel_id))),
            'test': np.load(os.path.join('..', 'data', 'test', '{}.npy'
                                         .format(channel_id))),
            'train': np.load(os.path.join('..', 'data', 'train', '{}.npy'
                                          .format(channel_id)))
        }

        self.channel_result_summary(channel, plot_values)

        sequence_type = 'true' if self.labels_available else 'predicted'
        y_shapes = self.create_shapes(eval(channel['anomaly_sequences'].values[0]),
                                      sequence_type, -1, 1, plot_values)
        e_shapes = self.create_shapes(eval(channel['anomaly_sequences'].values[0]),
                                      sequence_type, 0, None, plot_values)

        if self.labels_available:
            y_shapes += self.create_shapes(eval(channel['tp_sequences'].values[0])
                                           + eval(channel['fp_sequences'].values[0]),
                                           'predicted', -1, 1, plot_values)
            e_shapes += self.create_shapes(eval(channel['tp_sequences'].values[0])
                                           + eval(channel['fp_sequences'].values[0]),
                                           'predicted', 0, None, plot_values)

        train_df = pd.DataFrame({
            'train': plot_values['train'][:,0]
        })

        y_df = pd.DataFrame({
            'y_hat': plot_values['y_hat'].reshape(-1,)
        })

        y = plot_values['test'][self.config.l_s:-self.config.n_predictions][:,0]
        y_df['y'] = y
        if not len(y) == len(plot_values['y_hat']):
            modified_l_s = len(plot_values['y_test']) \
                           - len(plot_values['y_hat']) - 1
            y_df['y'] = y[modified_l_s:-1]

        e_df = pd.DataFrame({
            'e_s': plot_values['smoothed_errors'].reshape(-1,)
        })

        y_layout = {
            'title': 'y / y_hat comparison',
            'shapes': y_shapes,
        }

        e_layout = {
            'title': "Smoothed Errors (e_s)",
            'shapes': e_shapes,
        }

        if plot_train:
            train_df.iplot(kind='scatter', color='green',
                           layout={'title': "Training Data"})

        y_df.iplot(kind='scatter', layout=y_layout)

        if plot_errors:
            e_df.iplot(kind='scatter', layout=e_layout, color='red')

    def plot_all(self, plot_train=False, plot_errors=True):
        """
        Loop through all channels and plot.

        Args:
            plot_train (bool): If true, plot training data in separate plot
            plot_errors (bool): If true, plot prediction errors in separate plot
        """

        for idx, channel in self.result_df.iterrows():

            self.plot_channel(channel['chan_id'], plot_train=plot_train,
                              plot_errors=plot_errors)






