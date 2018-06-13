# Telemanom

### Branch info:
- `master`: for use with data containing labeled anomalies
- `no-labels`: for use with unlabeled data (a set of time-series streams)

## Anomaly Detection in Time Series Data Using LSTMs and Automatic Thresholding

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Telemanom employs vanilla LSTMs using [Keras](https://github.com/keras-team/keras)/[Tensorflow](https://github.com/tensorflow/tensorflow) to identify anomalies in multivariate sensor data. LSTMs are trained to learn normal system behaviors using encoded command information and prior telemetry values. Predictions are generated at each time step and the errors in predictions represent deviations from expected behavior. Telemanom then uses a novel nonparametric, unsupervised approach for thresholding these errors and identifying anomalous sequences of errors.

This repo along with the linked data can be used to re-create the experiments in our 2018 KDD paper, "[Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)", which describes the background, methodologies, and experiments in more detail. While the system was originally deployed to monitor spacecraft telemetry, it can be easily adapted to similar problems.

# Getting Started

Clone the repo (only available from source currently):

```sh
git clone https://github.com/khundman/telemanom.git && cd telemanom
```

From root of repo, curl and unzip data:

```sh
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
``` 

Install dependencies using **python 3.6+** (recommend using a virtualenv):

```sh
pip install -r requirements.txt
```

Configure system/modeling parameters in `config.yaml` file. For example:
- `train: True`  if `True`, a new model will be trained for each input stream. If `False` existing trained model will be loaded and used to generate predictions
- `predict: True`  Generate new predictions using models. If `False`, use existing saved predictions in evaluation (useful for tuning error thresholding and skipping prior processing steps)
- `l_s: 250` Determines the number of previous timesteps input to the model at each timestep `t` (used to generate predictions)  

Begin processing:

```sh
python run.py
```

A jupyter notebook for evaluating results for a run and comparing multiple runs with different params is provided in `results/`. To launch notebook:

```sh
jupyter notebook results/result-viewer.ipynb
``` 

Plotly is used to generate interactive inline plots, e.g.:

<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/telemanom/result-viewer.png" alt="drawing2" height="350"/>
</p>

# Data

## Using your own (unlabeled) data

Pre-split training and test sets must be placed in directories named `data/train/` and `data/test`. One `.npy` file should be generated for each channel or stream (for both train and test) with shape (`n_timesteps`, `n_inputs`). The filename should be a unique channel name or ID. The telemetry values being predicted in the test data *must* be the first feature in the input. 

For example, a channel `T-1` should have train/test sets named `T-1.npy` with shapes akin to `(4900,61)` and `(3925, 61)`, where the number of input dimensions are matching (`61`). The actual telemetry values should be along the first dimension `(4900,1)` and `(3925,1)`. 

## Raw experiment data

The raw data available for download represents real spacecraft telemetry data and anomalies from the Soil Moisture Active Passive satellite (SMAP) and the Curiosity Rover on Mars (MSL). All data has been anonymized with regard to time and all telemetry values are pre-scaled between `(-1,1)` according to the min/max in the test set. Channel IDs are also anonymized, but the first letter gives indicates the type of channel (`P` = power, `R` = radiation, etc.). Model input data also includes one-hot encoded information about commands that were sent or received by specific spacecraft modules in a given time window. No identifying information related to the timing or nature of commands is included in the data. For example:

<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/telemanom/example-combined.png" alt="drawing" height="570"/>
</p>

This data also includes pre-split test and training data, pre-trained models, predictions, and smoothed errors generated using the default settings in `config.yaml`. When getting familiar with the repo, running the `result-viewer.ipynb` notebook to visualize results is useful for developing intuition. The included data also is useful for isolating portions of the system. For example, if you wish to see the effects of changes to the thresholding parameters without having to train new models, you can set `Train` and `Predict` to `False` in `config.yaml` to use previously generated predictions from prior models. 

# Processing

Each time the system is started a unique datetime ID (ex. `2018-05-17_16.28.00`) will be used to create the following
- a **results** file (in `results/`) that includes identified anomalous sequences and related info 
- a **data subdirectory** containing data files for created models, predictions, and smoothed errors for each channel. A file called `params.log` is also created that contains parameter settings and logging output during processing. 

As mentioned, the jupyter notebook `results/result-viewer.ipynb` can be used to visualize results for each stream.

# Citation

If you use this work, please cite: 

``` @article{hundman2018detecting,
  title={Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding},
  author={Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and Colwell, Ian and Soderstrom, Tom},
  journal={arXiv preprint arXiv:1802.04431},
  year={2018}
}
```

# License 

Telemanom is distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).

Contact: Kyle Hundman (khundman@gmail.com)

# Contributors
- Kyle Hundman (NASA JPL)
- [Valentinos Constantinou](https://github.com/vc1492a) (NASA JPL)
- Chris Laporte (NASA JPL)
- [Ian Colwell](https://github.com/iancolwell) (NASA JPL)