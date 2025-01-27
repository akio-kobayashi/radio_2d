import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LitDAE
import torch.utils.data as dat
import torch.multiprocessing as mp
from speech_dataset import SpeechDataset
import utils.split_tensor as S
import speech_dataset
import bin.compute_features as C
from einops import rearrange
from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main(args, config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    input_mean, input_var, output_mean, output_var = C.load_mean_var(config['stat_path'])
    input_mean = rearrange(input_mean, ' b (f c) -> b f c', c=1).to(device)
    input_var = rearrange(input_var, ' b (f c) -> b f c', c=1).to(device)
    output_mean = rearrange(output_mean, ' b (f c) -> b f c', c=1).to(device)
    output_var = rearrange(output_var, ' b (f c) -> b f c', c=1).to(device)
    model = LitDAE.load_from_checkpoint(args.checkpoint,
                                        strict=False,
                                        config=config).to(device)
    model.eval()

    df = pd.read_csv(args.csv)
    for index, row in df.iterrows():
        mel = torch.load(df['noisy']).to(device)
        mel = (mel - input_mean)/input_var
        original_length = mel.shape[-1]
        split_mel = S.split_and_reshape(mel, config['num_frames'])
        mask = torch.ones_like(split_mel, device=split_mel.device)
        output = model.forward(split_mel, mask)
        output = S.reshape_back(output, original_length)
        output = output * output_var + output_mean
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
        
    main(args, config) 
