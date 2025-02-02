import os, sys
import numpy as np
import torch
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import utils.mel_spectrogram as M
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
    input_csv format:  key, clean, noisy, 
    output_csv format: key, clean, noisy
'''
def compute_features(input_csv, output_csv, output_dir, num_mels, magnitude):
    keys, cleans, noisys = [], [], []

    df = pd.read_csv(input_csv)
    for idx, row in df.iterrows():
        clean = M.get_mel_spectrogram(row['clean'], num_mels=num_mels)
        clean_path = os.path.join(output_dir, row['key']+'_clean') + '.pt'
        torch.save(clean, clean_path)

        if magnitude:
            noisy = M.get_mag_spectrogram(row['noisy'])
        else:
            noisy = M.get_mel_spectrogram(row['noisy'], num_mels=num_mels)
        noisy_path = os.path.join(output_dir, row['key']+'_noisy') + '.pt'
        torch.save(noisy, noisy_path)
        
        keys.append(row['key'])
        cleans.append(clean_path)
        noisys.append(noisy_path)
    
    out_df = pd.DataFrame(index=None)
    out_df['key'], out_df['clean'], out_df['noisy'] = keys, cleans, noisys

    out_df.to_csv(output_csv, index=False)

def compute_mean_var(input_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    df = pd.read_csv(input_csv)

    input_total_frames = 0
    input_sum=0.
    input_sum_square=0.
    output_total_frames = 0
    output_sum=0.
    output_sum_square=0.

    for idx, row in df.iterrows():
        melspec = torch.load(row['clean']).to(device)
        output_total_frames += melspec.shape[-1]
        output_sum += torch.sum(melspec, dim=-1)
        output_sum_square += torch.sum(torch.square(melspec), dim=-1)
        
        melspec = torch.load(row['noisy']).to(device)
        input_total_frames += melspec.shape[-1]
        input_sum += torch.sum(melspec, dim=-1)
        input_sum_square += torch.sum(torch.square(melspec), dim=-1)
        
    input_mean = input_sum/input_total_frames 
    input_var = input_sum_square/input_total_frames - torch.square(input_mean)
    output_mean = output_sum/output_total_frames 
    output_var = output_sum_square/output_total_frames - torch.square(output_mean)

    return input_mean, torch.sqrt(input_var + 1.e-8), output_mean, torch.sqrt(output_var + 1.e-8)

def save_mean_var(input_mean, input_var, output_mean, output_var, path):
    #if isinstance(mean):
    input_mean = input_mean.detach().cpu().numpy()
    input_var = input_var.detach().cpu().numpy()
    output_mean = output_mean.detach().cpu().numpy()
    output_var = output_var.detach().cpu().numpy()

    np.savez(path, input_mean=input_mean, input_var=input_var, 
             output_mean=output_mean, output_var=output_var)

def load_mean_var(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    npz = np.load(path)
    input_mean = torch.from_numpy(npz['input_mean']).to(device)
    input_var = torch.from_numpy(npz['input_var']).to(device)
    output_mean = torch.from_numpy(npz['output_mean']).to(device)
    output_var = torch.from_numpy(npz['output_var']).to(device)

    return input_mean, input_var, output_mean, output_var

def remove_short_long_features(df, min_frames=200, max_frames=2500):
    for idx, row in df:
        spec = torch.load(row['melspec'])
        if spec.shape[-1] < min_frames:
            df.drop(index=idx, inplace=True)
            continue
        if spec.shape[-1] < max_frames:
            df.drop(index=idx, inplace=True)
            continue

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--output_stats', type=str, default='stats.npz')
    parser.add_argument('--num_mels', type=int, default=80)
    parser.add_argument('--magnitude', action='store_true')
    args=parser.parse_args()
       
    compute_features(args.input_csv, args.output_csv, args.output_dir, args.num_mels, args.magnitude)
    input_mean, input_var, output_mean, output_var = compute_mean_var(args.output_csv)
    save_mean_var(input_mean, input_var, output_mean, output_var, args.output_stats)
    
