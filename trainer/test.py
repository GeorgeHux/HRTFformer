import os
import pickle
import scipy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model.model import HRTF_Transformer

from data.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from data.utils import get_dataset_info, load_mean_std, inverse_sht

from trainer.utils import *
from configs.config import Config
from configs.model_config import  ModelConfig

def test(config: Config, checkpoint_path):
    domain = config.domain

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
    else:
        mean, std = None
    _, test_prefetcher = load_hrtf(config, mean, std)
    print("test set loadded successfully")

    ngpu = config.ngpu
    device = torch.device(config.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    recon_mag_dir = checkpoint_path + '/mag'
    recon_db_dir = checkpoint_path + '/db'
    os.makedirs(recon_mag_dir, exist_ok=True)
    os.makedirs(recon_db_dir, exist_ok=True)

    nbins = config.nbins_hrtf * 2
    max_num_coeffs = (config.max_degree + 1) ** 2
    encoder_config = ModelConfig(in_channels=nbins,
                                 hidden_size=config.hidden_size,
                                 num_transformer_layers=config.num_encoder_transformer_layers,
                                 num_heads=config.num_heads,
                                 num_groups=config.num_groups,
                                 dropout=config.dropout,
                                 num_initial_coeff=config.num_initial_points,
                                 max_num_coeff=max_num_coeffs)
    
    decoder_config = ModelConfig(in_channels=2048,
                                 hidden_size=config.hidden_size,
                                 num_transformer_layers=config.num_decoder_transformer_layers,
                                 num_heads=config.num_heads,
                                 num_groups=config.num_groups,
                                 dropout=config.dropout,
                                 num_initial_coeff=config.num_initial_points,
                                 max_num_coeff=max_num_coeffs)
    # model initialization
    model = HRTF_Transformer(encoder_config, decoder_config).to(device)
    print("Build hrtf transformer model successfully.")
    checkpoint = checkpoint_path + '/transformer.pt'
    model.load_state_dict(checkpoint, map_location=torch.device('cpu'))
    print(f"Load hrtf transformer model weights '{checkpoint_path} successfully.'")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_mb = param_size / 1024 ** 2
    size_buffer_mb = buffer_size / 1024 ** 2
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('param size: {:.3f}MB'.format(size_param_mb))
    print('buffer size: {:.3f}MB'.format(size_buffer_mb))
    print('model size: {:.3f}MB'.format(size_all_mb))

    # Start the verification mode of the model.
    model.eval()

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    plot_min_max_diff = True
    count = 0
    avg_lsd = []
    while batch_data is not None:
        print(f"test {count + 1} / {len(test_prefetcher)}")
        lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hrtf = batch_data["hrtf"]
        mask = batch_data["mask"]
        sample_id = batch_data["id"]

        # upsample lr coefficient
        with torch.no_grad():
            sr = model(lr_coefficient)
        recon = inverse_sht(config, sr, mask)[0]

        # save reconstructed hrtfs into pickle files
        file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
        with open(recon_db_dir + file_name, "wb") as file:
            pickle.dump(recon, file)
        with open(recon_mag_dir + file_name, "wb") as file:
            recon = 10 ** (recon / 20)
            pickle.dump(recon, file)

        ir_id = 0
        max_value = None
        max_id = None
        min_value = None
        min_id = None
        recon = recon.view(nbins, -1).T.detach().cpu()
        original_hrtf = hrtf[0].view(nbins, -1).T.detach().cpu()
        total_all_position = 0
        total_positions = len(recon)
        total_sd_metric = 0
        print("subject: ", sample_id)
        for original, generated in zip(original_hrtf, recon):
            if domain == "meganitude_db":
                original = 10 ** (original / 20)
                generated = 10 ** (generated / 20)

            if domain == "megnitude_db" or domain == "magnitude":
                average_over_frequencies = spectral_distortion_inner(abs(generated), abs(original))
            elif domain == "time":
                nbins = config.nbins_hrtf
                ori_tf_left = abs(scipy.fft.rfft(original[:nbins], nbins*2)[1:])
                ori_tf_right = abs(scipy.fft.rfft(original[nbins:], nbins*2)[1:])
                gen_tf_left = abs(scipy.fft.rfft(generated[:nbins], nbins*2)[1:])
                gen_tf_right = abs(scipy.fft.rfft(generated[nbins:], nbins*2)[1:])

                ori_tf = np.ma.concatenate([ori_tf_left, ori_tf_right])
                gen_tf = np.ma.concatenate([gen_tf_left, gen_tf_right])

                average_over_frequencies = spectral_distortion_inner(gen_tf, ori_tf)
            total_all_position += np.sqrt(average_over_frequencies)

            if max_value is None or np.sqrt(average_over_frequencies) > max_value:
                max_value = np.sqrt(average_over_frequencies)
                max_ir = ir_id
            if min_value is None or np.sqrt(average_over_frequencies) < min_value:
                min_value = np.sqrt(average_over_frequencies)
                min_id = ir_id
            ir_id += 1
        
        sd_metric = total_all_position / total_positions
        total_sd_metric += sd_metric
        avg_lsd.append(sd_metric)
        print("Log SD (across all positions: ", float(sd_metric))

        if plot_min_max_diff:
            plot_test_sample_hrtf(min_id, original_hrtf, recon, is_min=True)
            plot_test_sample_hrtf(max_id, original_hrtf, recon, is_min=False)

        # Preload the next batch of data
        batch_data = test_prefetcher.next()
    print("lsd for all test subject: ", avg_lsd)
    mean_lsd = np.mean(avg_lsd)
    print("avg lsd: ", mean_lsd)
