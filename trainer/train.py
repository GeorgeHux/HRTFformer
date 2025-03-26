import pickle
import importlib
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
import time

from trainer.utils import *
from data.utils import get_dataset_info, load_mean_std, inverse_sht

from configs.config import Config
from configs.model_config import  ModelConfig
from model.model import HRTF_Transformer

def get_model_and_optimizer(config: Config):
    ngpu = config.ngpu
    device = torch.device(config.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    nbins = config.nbins_hrtf * 2 # left and right
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
    hrtf_transformer = HRTF_Transformer(encoder_config, decoder_config).to(device)

    # optimizer
    optimizer = optim.Adam(hrtf_transformer.parameters(), lr=config.lr)

    return hrtf_transformer, optimizer

def train(config: Config, model, optimizer, train_prefetcher):
    """ Train the transformer model

    Args:
        config: Config object containing model hyperparameters
        model: transformer model instance
        train_prefetcher: prefetcher for training data
    """
    domain = config.domain
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(config.log_path, str(config.num_initial_points), current_time)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")
    with open(log_file_path, "a") as f:
        f.write(f"training in {domain} domain\n\n")
    plot_dir = os.path.join(log_dir, "plots", "train")
    os.makedirs(plot_dir, exist_ok=True)
    checkpoint_dir = os.path.join(config.checkpoint_path, str(config.num_initial_points), current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # get data distribution info (row angles, column angles, radii) for latter use
    config.row_angles, config.column_angles, config.radii = get_dataset_info(config)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Assign torch device
    ngpu = config.ngpu

    device = torch.device(config.device_name if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    print(f'Using {ngpu} GPUs. ')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    cos_similarity_criterion = cos_similarity_loss
    content_criterion = sd_ild_loss

    # mean and std for ILD and SD, which are used for normalization
    # computed based on average ILD and SD for training data, when comparing each individual
    # to every other individual in the training data
    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    if config.normalize_input:
        mean, std = load_mean_std(config, device)

    train_loss_list = []
    train_content_loss_list = []
    train_sh_coeff_mse_list = []
    train_sh_coeff_cos_list = []

    for epoch in range(config.num_epochs):
        with open(log_file_path, "a") as f:
            f.write(f"\nEpoch: {epoch}\n")

        times = []
        train_loss = 0.
        train_content_loss = 0.
        train_sh_coeff_mse_loss = 0.
        train_sh_coeff_cos_loss = 0.

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0

        # Initialize the data loader and load the first batch of data
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            # Transfer in-memory data to CUDA devices to speed up training
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"]
            
            sr = model(lr_coefficient)
            sh_coeff_cos_loss = cos_similarity_criterion(sr, hr_coefficient)
            sh_coeff_mse_loss = ((sr - hr_coefficient) ** 2).mean()
            recons = inverse_sht(config, sr, masks)

            # during every 25th epoch and last epoch, save filename for mag spectrum plot
            if epoch % 25 == 0 or epoch == (config.num_epochs - 1):
                generated = recons[0].permute(2, 3, 1, 0)  # w x h x r x nbins
                target = hrtf[0].permute(2, 3, 1, 0)
                id = batch_data['id'][0].item()
                filename = f"magnitude_{id}_{epoch}"
                plot_hrtf(generated.detach().cpu(), target.detach().cpu(), plot_dir, filename)

            # loss
            unweighted_content_loss = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
            content_loss = config.content_weight * unweighted_content_loss
            loss = content_loss + sh_coeff_cos_loss

            train_loss += loss.item()
            train_content_loss += content_loss.item()
            train_sh_coeff_cos_loss += sh_coeff_cos_loss.item()
            train_sh_coeff_mse_loss += sh_coeff_mse_loss.item()
            
            # backward
            loss.backward()

            # optimizer
            optimizer.step()
            optimizer.zero_grad()

            with open(log_file_path, "a") as f:
                f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                f.write(f"loss: {loss.item()}\n")
                f.write(f"content loss: {content_loss.item()}, sh cos: {sh_coeff_cos_loss.item()}, sh mse: {sh_coeff_mse_loss.item()}\n\n")
            
            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                with torch.no_grad():
                    torch.save(model.state_dict(), f'{checkpoint_dir}/transformer.pt')
                    progress(batch_index, batches, epoch, config.num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1
        train_loss_list.append(train_loss / len(train_prefetcher))
        train_content_loss_list.append(train_content_loss / len(train_prefetcher))
        train_sh_coeff_cos_list.append(train_sh_coeff_cos_loss / len(train_prefetcher))
        train_sh_coeff_mse_list.append(train_sh_coeff_mse_loss / len(train_prefetcher))
        print(f"Average epoch loss: {train_loss_list[-1]}")
        print(f"Average content loss: {train_content_loss_list[-1]}")
        print(f"Aberage sh mse loss: {train_sh_coeff_mse_list[-1]}, sh cos loss: {train_sh_coeff_cos_list[-1]}")
    
    # plot loss curves
    plot_path = os.path.join(plot_dir, "losses")
    os.makedirs(plot_path, exist_ok=True)
    plot_losses([train_loss_list], ['Training loss'], ['red'], path=plot_path, filename='loss', title="Training Loss")
    plot_losses([train_sh_coeff_mse_list],['SH mse loss'],['blue'], path=plot_path, filename='SH_mse_loss', title="SH mse loss")
    plot_losses([train_sh_coeff_cos_list],['SH cos loss'],['blue'], path=plot_path, filename='SH_cos_loss', title="SH cos loss")
    plot_losses([train_loss_list, train_content_loss_list, train_sh_coeff_cos_list],
                ['Training loss', 'Content loss', 'coefficient sim loss'],
                ['green', 'purple', 'red'],
                path=plot_path, filename='loss_curves', title="Training loss curves")
    
    with open(f'{log_dir}/train_losses.pickle', "wb") as file:
        pickle.dump((train_loss_list, train_content_loss_list, train_sh_coeff_cos_list, train_sh_coeff_mse_list), file)
    print("TRAINING FINISHED")
    
def train_model(config: Config):
    if config.normalize_input:
        print("normalize input")
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
        mean, std = None, None
    train_prefetcher, _ = load_hrtf(config, mean, std)
    print("train prefetcher: ", len(train_prefetcher))

    hrtf_transformer, optimizer = get_model_and_optimizer(config)
    print("------Start training!--------")
    train(config, hrtf_transformer, optimizer, train_prefetcher)


