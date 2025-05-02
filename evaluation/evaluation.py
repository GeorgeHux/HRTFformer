from trainer.utils import spectral_distortion_metric
from data.preprocessing.utils import convert_to_sofa
from configs.config import Config

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

import matlab.engine

def replace_nodes(config: Config, sr_dir, file_name):
    with open(config.valid_target_path + file_name, "rb") as f:
        hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)   # w x h x r x nbins

    selected_coords = get_sample_coords(config.num_initial_points)
    for coord in selected_coords:
        sr_hrtf[coord[0], coord[1], :] = hr_hrtf[coord[0], coord[1], :]

    generated = torch.permute(sr_hrtf[None, :], (0, 4, 3, 1, 2)) # 1 x nbins x r x w x h
    target = torch.permute(hr_hrtf[None, :], (0, 4, 3, 1, 2))

    return target, generated

def run_lsd_evaluation(config: Config, sr_dir, file_ext=None, hrtf_selection=None):
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
            with open(f'{sr_dir}/log.txt', 'a') as f:
                f.write('LSD Error of subject %s: %0.4f \n' % (subject_id, float(error.detach())))

        # with open(f'{config.valid_recon_path}/{config.upscale_factor}/mag/{file_ext}', "wb") as file:
        # with open(f'{config.path}/{config.upscale_factor}/{file_ext}', "wb") as file:
        with open(f"{sr_dir}/{file_ext}", "wb") as file:
            pickle.dump(lsd_errors, file)
    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open(f'{sr_dir}/log.txt', 'a') as f:
        f.write('Mean LSD Error: %0.3f \n' % np.mean([error[1] for error in lsd_errors]))
    

def run_localisation_evaluation(config: Config, sr_dir, file_ext=None, hrtf_selection=None):
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
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path + '/sofa_min_phase')]

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    # s = eng.genpath(config.data_dir_path)
    s = eng.genpath('C:/Users/steph/Desktop/XuyiHu/HRTF-neurips')
    eng.addpath(s, nargout=0)

    loc_errors = []
    for file in hrtf_file_names:
        target_sofa_file = config.valid_target_path + '/sofa_min_phase/' + file
        if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
            generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
        else:
            generated_sofa_file = nodes_replaced_path + '/sofa_min_phase/' + file

        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)
        with open(f'{sr_dir}/loc_test.txt', 'a') as f:
            f.write(f"subject {subject_id}: pol_acc1: {pol_acc1}, pol_rms1: {pol_rms1}, querr1: {querr1}\n")

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{sr_dir}/loc_test.txt', 'a') as f:
        f.write('Mean ACC Error: %0.3f \n' % np.mean([error[1] for error in loc_errors]))
        f.write('Mean RMS Error: %0.3f \n' % np.mean([error[2] for error in loc_errors]))
        f.write('Mean QUERR Error: %0.3f \n' % np.mean([error[3] for error in loc_errors]))

    with open(f'{sr_dir}/{file_ext}', "wb") as file:
        pickle.dump(loc_errors, file)