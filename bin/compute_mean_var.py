import os, sys
import numpy as np
import torch
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import utils.mel_spectrogram as M
from argparse import ArgumentParser
import yaml
import warnings
import bin.compute_features as F
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_stats', type=str, default='stats.npz')
    args=parser.parse_args()
       
    input_mean, input_var, output_mean, output_var = F.compute_mean_var(args.input_csv)
    print(input_mean.shape, input_var.shape, output_mean.shape, output_var.shape)
    F.save_mean_var(input_mean, input_var, output_mean, output_var, args.output_stats)
    
