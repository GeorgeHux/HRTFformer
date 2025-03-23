from trainer.utils import spectral_distortion_metric
from data.preprocessing.utils import convert_to_sofa

import shutil
from pathlib import Path
import importlib

import glob
import torch
import pickle
import os
import re
import numpy as np

from data.dataset import get_sample_coords
from data.utils import get_dataset_info

def replace_nodes(config, sr_dir, file_name):
    with open(config.valid_target_path + file_name, "rb") as f:
        hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)   # w x h x r x nbins

    selected_coords = get_sample_coords(config.upscale_factor)
    for coord in selected_coords:
        sr_hrtf[coord[0], coord[1], :] = hr_hrtf[coord[0], coord[1], :]

    generated = torch.permute(sr_hrtf[None, :], (0, 4, 3, 1, 2)) # 1 x nbins x r x w x h
    target = torch.permute(hr_hrtf[None, :], (0, 4, 3, 1, 2))

    return target, generated

def run_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_target_path, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
        # Overwrite the generated points that exist in the original data
            with open(config.valid_target_path + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{sr_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3)) 
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))  # 1 x nbins x r x w x h

            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
        with open(f'{sr_dir}/{file_ext}', "wb") as file:
            pickle.dump(lsd_errors, file)
    else:
        val_data_paths = glob.glob(f"{sr_dir}/{config.dataset}_*")
        val_data_file_names = ['/' + os.path.basename(x) for x in val_data_paths]

        lsd_errors = []
        for file_name in val_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)
            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))

        # with open(f'{config.valid_recon_path}/{config.upscale_factor}/mag/{file_ext}', "wb") as file:
        # with open(f'{config.path}/{config.upscale_factor}/{file_ext}', "wb") as file:
        with open(f"{sr_dir}/{file_ext}", "wb") as file:
            pickle.dump(lsd_errors, file)
    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open('log.txt', 'a') as f:
        f.write('Mean LSD Error: %0.3f \n' % np.mean([error[1] for error in lsd_errors]))
    

def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
    # imp = importlib.import_module('data.hrtfdata.full')
    # load_function = getattr(imp, config.dataset)
    # data_dir = config.raw_hrtf_dir / config.dataset
    # ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
    #                                                      'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    # row_angles = ds.row_angles
    # column_angles = ds.column_angles
    row_angles, column_angles, _ = get_dataset_info(config)

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        nodes_replaced_path = sr_dir
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_target_path + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file) # r x w x h x nbins

        convert_to_sofa(nodes_replaced_path, config, row_angles, column_angles)
        print('Created valid sofa files')