import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib

from config import Config
from model.train import train
from model.test import test
from model.util import load_hrtf
from preprocessing.hrtf_sphere import HRTF_Sphere
from preprocessing.utils import convert_to_sofa

from baselines.barycentric_interpolation import run_barycentric_interpolation
from baselines.hrtf_selection import run_hrtf_selection
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation

import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)

def main(config, mode):
    # Initialize Config
    data_dir = config.raw_hrtf_dir / config.dataset
    print(os.getcwd())
    print(config.dataset)

    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)

    if mode == 'preprocess':
        ds_left = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                  'side': 'left', 'domain': 'magnitude_db'}})
        ds_right = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                  'side': 'right', 'domain': 'magnitude_db'}})
        
        # Split data into train and test sets
        train_size = int(len(set(ds_left.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds_left.subject_ids)), train_size, replace=False)
        val_sample = list(set(ds_left.subject_ids) - set(train_sample))
        print("num train samples: ", len(train_sample))
        print(train_sample)
        print("num validation samples: ", len(val_sample))
        print(val_sample)
        id_file_dir = config.train_val_id_dir
        if not os.path.exists(id_file_dir):
            os.makedirs(id_file_dir)
        id_filename = id_file_dir + '/train_val_id.pickle'
        with open(id_filename, "wb") as file:
            pickle.dump((train_sample, val_sample), file)

        valid_target_path = config.valid_target_path
        shutil.rmtree(Path(valid_target_path), ignore_errors=True)
        Path(valid_target_path).mkdir(parents=True, exist_ok=True)

        # collect all train_hrtfs to get mean and sd
        num_rows = len(ds_left.row_angles)
        num_columns = len(ds_left.column_angles)
        j = 0
        train_hrtfs = torch.empty(size=(2 * train_size, 1, num_rows, num_columns, config.nbins_hrtf))
        for i in range(len(ds_left)):
            left = ds_left[i]['features'][:, :, :, 1:]
            right = ds_right[i]['features'][:, :, :, 1:]
            merge = np.ma.concatenate([left, right], axis=3)
            merge = torch.from_numpy(merge.data).permute(2, 0, 1, 3) # r x w x h x nbins
            merge = 10 ** (merge/20)
            if ds_left.subject_ids[i] in train_sample:
                train_hrtfs[j] = merge[:, :, :, :config.nbins_hrtf] # add left
                j += 1
                train_hrtfs[j] = merge[:, :, :, config.nbins_hrtf:] # add right
                j += 1
            else: # store test HRTFs
                subject_id = str(ds_left.subject_ids[i])
                file_name = '/' + f"{config.dataset}_{subject_id}.pickle"
                with open(valid_target_path + file_name, "wb") as file:
                    pickle.dump(merge, file)

        if config.gen_sofa_flag:
            convert_to_sofa(valid_target_path, config, ds_left.row_angles, ds_left.column_angles)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        print("using cuda? ", torch.cuda.is_available())
        batch_size, lr_G, lr_D, latent_dim, critic_iters, max_degree = config.get_train_params()

        # create path
        path = f'{config.path}/{config.upscale_factor}'
        shutil.rmtree(Path(path), ignore_errors=True)
        Path(path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{config.path}/{config.upscale_factor}/log.txt", "a") as f:
            f.write(f"batch size: {batch_size}\n")
            f.write(f"generator lr: {lr_G}\n")
            f.write(f"discriminator lr: {lr_D}\n")
            f.write(f"latent_dim: {latent_dim}\n")
            f.write(f"critic iters: {critic_iters}\n")
            f.write(f"max degree:  {max_degree}\n")
        if config.transform_flag:
            mean_std_dir = config.mean_std_coef_dir
            mean_std_full = mean_std_dir + "/mean_std_full.pickle"
            with open(mean_std_full, "rb") as f:
                mean_full, std_full = pickle.load(f)
            
            mean_std_lr = mean_std_dir + f"/mean_std_{config.upscale_factor}.pickle"
            with open(mean_std_lr, "rb") as f:
                mean_lr, std_lr = pickle.load(f)
            mean = (mean_lr, mean_full)
            std = (std_lr, std_full)
            train_prefetcher, _ = load_hrtf(config, mean, std)
        else:
            train_prefetcher, _ = load_hrtf(config)
        train_prefetcher, _ = load_hrtf(config)
        print("transform applied? ", config.transform_flag)
        print("train fetcher: ", len(train_prefetcher))

        # Trains the model, according to the parameters specified in Config
        train(config, train_prefetcher)
    
    elif mode == 'test':
        print("using cuda? ", torch.cuda.is_available())
        with open(f"{config.path}/{config.upscale_factor}/test_log.txt", "a") as f:
            f.write(f"upscale factor: {config.upscale_factor}\n")
        if config.transform_flag:
            mean_std_dir = config.mean_std_coef_dir
            mean_std_full = mean_std_dir + "/mean_std_full.pickle"
            with open(mean_std_full, "rb") as f:
                mean_full, std_full = pickle.load(f)
            
            mean_std_lr = mean_std_dir + f"/mean_std_{config.upscale_factor}.pickle"
            with open(mean_std_lr, "rb") as f:
                mean_lr, std_lr = pickle.load(f)
            mean = (mean_lr, mean_full)
            std = (std_lr, std_full)
            _, test_prefetcher = load_hrtf(config, mean, std)
        else:
            _, test_prefetcher = load_hrtf(config)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)
        sr_dir = config.valid_recon_path + f'/{config.upscale_factor}/mag'
        run_lsd_evaluation(config, sr_dir)
        run_localisation_evaluation(config, sr_dir)

    elif mode == 'barycentric_baseline':
        barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder

        run_barycentric_interpolation(config, barycentric_output_path)
        if config.gen_sofa_flag:
            ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                 'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
            row_angles = ds.row_angles
            column_angles = ds.column_angles
            convert_to_sofa(barycentric_output_path, config, row_angles, column_angles)
            print('Created barycentric baseline sofa files')

        config.path = config.barycentric_hrtf_dir
        file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_lsd_evaluation(config, barycentric_output_path, file_ext)

        file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_localisation_evaluation(config, barycentric_output_path, file_ext)

    elif mode == 'hrtf_selection_baseline':
        config.domain = "magnitude"
        run_hrtf_selection(config, config.hrtf_selection_dir)
        if config.gen_sofa_flag:
            ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                 'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
            row_angles = ds.row_angles
            column_angles = ds.column_angles
            convert_to_sofa(config.hrtf_selection_dir, config, row_angles, column_angles)

        config.path = config.hrtf_selection_dir

        file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
        file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')

        file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')
        file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')

    print("finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-r", "--remote")
    args = parser.parse_args()

    if args.remote == "True":
        remote = True
    elif args.remote == "False":
        remote = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the remote tag (-r/--remote)")
    
    config = Config(remote)
    main(config, args.mode)