import cmath
import pickle
import os

import sofar as sf
import numpy as np
import torch
import pandas as pd
import scipy
from scipy.signal import hilbert
import shutil
from pathlib import Path
import re
import glob

from configs.config import Config
from trainer.utils import spectral_distortion_metric, ILD_metric

def get_train_data_statistics(config: Config, train_samples, mask):
    train_samples = [x.permute(3, 0, 1, 2).unsqueeze(0) for x in train_samples]
    left_hrtfs = [x[:,:config.nbins_hrtf,...] for x in train_samples]
    right_hrtfs = [x[:,config.nbins_hrtf:,...] for x in train_samples]
    mask = mask.unsqueeze(0)

    sd = []
    ild = []
    
    for cur in range(len(train_samples)):
        running_sd = 0
        running_ild = 0
        for ref in range(len(train_samples)):
            if cur != ref:
                sd_right = spectral_distortion_metric(right_hrtfs[cur], right_hrtfs[ref], mask)
                sd_left = spectral_distortion_metric(left_hrtfs[cur], left_hrtfs[ref], mask)
                running_sd += (sd_right + sd_left) / 2.

                running_ild += ILD_metric(config.nbins_hrtf, train_samples[cur], train_samples[ref], mask)
        sd.append(running_sd / len(train_samples) - 1)
        ild.append(running_ild / len(train_samples) - 1)
    sd = torch.tensor(sd)
    ild = torch.tensor(ild)
    sd_mean, sd_std = torch.mean(sd).item(), torch.std(sd).item()
    ild_mean, ild_std = torch.mean(ild).item(), torch.std(ild).item()
    print(f"sd_mean: {sd_mean}, sd_std: {sd_std}, ild_mean: {ild_mean}, ild_std: {ild_std}")
    with open(config.train_sd_ild_mean_std_filename, 'wb') as file:
        pickle.dump((sd_mean, sd_std, ild_mean, ild_std), file)

def add_itd(az, el, hrir, side, fs=48000, r=0.0875, c=343):

    az = np.radians(az)
    el = np.radians(el)
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))
    delay_in_sec = (r / c) * (interaural_azimuth + np.sin(interaural_azimuth))
    fractional_delay = delay_in_sec * fs

    sample_delay = int(abs(fractional_delay))

    if (delay_in_sec > 0 and side == 'right') or (delay_in_sec < 0 and side == 'left'):
        N = len(hrir)
        delayed_hrir = np.zeros(N)
        delayed_hrir[sample_delay:] = hrir[0:N - sample_delay]
        sofa_delay = sample_delay
    else:
        sofa_delay = 0
        delayed_hrir = hrir

    return delayed_hrir, sofa_delay

def gen_sofa_file(config, left_hrtf, right_hrtf, az, el):
    source_position = [az + 360 if az < 0 else az, el, 1.5]

    left_hrtf[left_hrtf == 0.0] = 1.0e-08
    left_phase = np.imag(-hilbert(np.log(np.abs(left_hrtf))))
    right_hrtf[right_hrtf == 0.0] = 1.0e-08
    right_phase = np.imag(-hilbert(np.log(np.abs(right_hrtf))))

    left_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(left_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * left_phase))[:config.nbins_hrtf]
    right_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(right_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * right_phase))[:config.nbins_hrtf]

    left_hrir, left_sample_delay = add_itd(az, el, left_hrir, side='left')
    right_hrir, right_sample_delay = add_itd(az, el, right_hrir, side='right')

    full_hrir = [left_hrir, right_hrir]
    delay = [left_sample_delay, right_sample_delay]

    return source_position, full_hrir, delay

def save_sofa(clean_hrtf, config, row_angles, column_angles, sofa_path_output):
    full_hrirs = []
    source_positions = []
    delays = []
    left_full_hrtf = clean_hrtf[:, :, :, :config.nbins_hrtf]   # r x w x h x nbins
    right_full_hrtf = clean_hrtf[:, :, :, config.nbins_hrtf:]

    for i in range(clean_hrtf.size(1)):  # loop through azimuth
        for j in range(clean_hrtf.size(2)): # loop through elevation
            left_hrtf = np.array(left_full_hrtf[0, i, j]) # only one radius
            right_hrtf = np.array(right_full_hrtf[0, i, j])
            az = row_angles[i]
            el = column_angles[j]
            source_position, full_hrir, delay = gen_sofa_file(config, left_hrtf, right_hrtf, az, el)

            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)
    
    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = full_hrirs
    sofa.Data_SamplingRate = config.hrir_samplerate
    sofa.Data_Delay = delays
    sofa.SourcePosition = source_positions
    sf.write_sofa(sofa_path_output, sofa)

def convert_to_sofa(hrtf_dir, config, row_angles, column_angles, phase_ext='_phase', mag_ext='_mag'):
    sofa_path_output = hrtf_dir + '/sofa_min_phase/'

    hrtf_file_names = [hrir_file_name for hrir_file_name in os.listdir(hrtf_dir)
                       if os.path.isfile(os.path.join(hrtf_dir, hrir_file_name)) and phase_ext not in hrir_file_name]
    
    # Clear/Create directories
    shutil.rmtree(Path(sofa_path_output), ignore_errors=True)
    Path(sofa_path_output).mkdir(parents=True, exist_ok=True)

    nbins = config.nbins_hrtf * 2
    num_rows = len(row_angles)
    num_cols = len(column_angles)
    for f in hrtf_file_names:
        with open(os.path.join(hrtf_dir, f), "rb") as hrtf_file:
            hrtf = pickle.load(hrtf_file) # r x w x h x nbins
            if hrtf.shape == (num_rows, num_cols, 1, nbins):
                hrtf = hrtf.permute(2, 0, 1, 3)
            expected_shape = (1, num_rows, num_cols, nbins)
            assert hrtf.shape == expected_shape, f"Expected shape {expected_shape}, but got shape {hrtf.shape}"
            sofa_filename_output = os.path.basename(hrtf_file.name).replace('.pickle', '.sofa').replace(mag_ext, '')
            sofa_output = sofa_path_output + sofa_filename_output

            save_sofa(hrtf, config, row_angles, column_angles, sofa_output)

def get_feature_for_point_tensor(elevation, azimuth, all_coords, subject_features):
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(subject_features[0][int(all_coords_row.azimuth_index)][int(all_coords_row.elevation_index)]))))

def calc_interpolated_feature(triangle_vertices, coeffs, all_coords, subject_features):
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point_tensor(p[0], p[1], all_coords, subject_features)
        features.append(features_p)
    
    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    if len(features) == 3:
        interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]
    else:
        interpolated_feature = features[0]

    return interpolated_feature

def calc_all_interpolated_features(hrtf_sphere, features,  euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs):
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=hrtf_sphere.get_df(),
                                                   subject_features=features)
            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)
    return selected_feature_interpolated

def calc_hrtf(config, hrirs):
    """FFT to obtain HRTF from HRIR"""
    magnitudes = []
    phases = []

    for hrir in hrirs:
        # remove value that corresponds to 0 Hz
        hrtf = scipy.fft.rfft(hrir, config.nbins_hrtf*2)[1:]
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)
    return magnitudes, phases

def interpolate_fft(config, hrtf_sphere, features, full_size, sphere_coords, sphere_triangles, sphere_coeffs):
    """
    hrtf_sphere: HRTF_Sphere object associated with dataset
    features: features for a given subject, 
    sphere_coords: A list of locations of the gridded cubed sphere points to be interpolated,
                     given as (elevation, azimuth)
    sphere_triangles: A list of lists of triangle vertices for barycentric interpolation, where each list of
                             vertices defines the triangle for the corresponding point in sphere
    sphere_coeffs: A list of barycentric coordinates for each location in sphere, corresponding to the triangles
                          described by sphere_triangles
    """
    interpolated_hrirs = calc_all_interpolated_features(hrtf_sphere, features, sphere_coords, sphere_triangles, sphere_coeffs)
    magnitudes, phases = calc_hrtf(config, interpolated_hrirs)
    magnitudes_raw = [[[[] for _ in range(full_size[1])] for _ in range(full_size[0])] for _ in range(1)]
    count = 0
    for i in range(full_size[0]):
        for j in range(full_size[1]):
            magnitudes_raw[0][i][j] = magnitudes[count]
            count += 1
    
    return torch.tensor(np.array(magnitudes_raw))