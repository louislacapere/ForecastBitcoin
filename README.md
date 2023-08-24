# Case study ST4 MDS

This repo contains all starting files for ST4 MDS case study

# Clone this repository

Clone or download this repository to start working

Via ssh

`git clone git@gitlab.com:centralesupelec_ds/st4-mds.git`

Via http

`git clone https://gitlab.com/centralesupelec_ds/st4-mds.git`

## Installation

### Go to path

`cd /path/to/st4_mds_prepa`

### Create a virtual environment (you may skip this step if mds_env already exists in your directory)

Using virtualenv

`virtualenv -p python3 mds_env`

Using conda

`conda create -n mds_env python=3`

### Activate virtual environment

Using virtualenv

`. mds_env/bin/activate`

Using conda

`source activate mds_env`

### Install requirements

`pip install -r requirements.txt`

## Project Organization

    ├── README.md          <- Contains desctiption of the project
    ├── notebook           <- folder for notebook
        ├── case_study_arima_group_n.ipynb <- starter notebook for ARIMA modeling, replace n by the group number
        ├── case_study_rnn_group_n.ipynb <- starter notebook for RNN modeling, replace n by the group number
    ├── data               <- folder for data
        ├── df_blockchain.csv <- bitcoin and blockchain data from https://blockchain.info/
    ├── .gitignore
