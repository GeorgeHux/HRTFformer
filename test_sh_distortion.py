from data.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from configs.config import Config
import importlib
import pickle
import numpy as np
import torch
from trainer.utils import spectral_distortion_metric
from data.utils import get_dataset_info
from data.preprocessing.utils import convert_to_sofa
import os
import re
from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf

imp = importlib.import_module('data.hrtfdata.full')
load_function = getattr(imp, 'SONICOM')

config = Config(True, 'Sonicom')
data_dir = config.raw_hrtf_dir / config.dataset.upper()
domain = config.domain
id_file_dir = config.train_val_id_dir
id_filename = id_file_dir + '/train_val_id.pickle'
with open(id_filename, "rb") as file:
    train_ids, val_ids = pickle.load(file)

left_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': domain}}, subject_ids=val_ids)
right_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': domain}}, subject_ids=val_ids)

eval_resutls = {10: 4.9959,
100: 4.6378,
104: 4.8680,
116: 5.5978,
118: 5.0804,
128: 5.9740,
141: 5.4608,
143: 4.8945,
148: 4.3526,
149: 6.6996,
152: 5.2457,
166: 5.7818,
168: 5.0484,
169: 5.1579, 
172: 4.3209,
173: 6.9212,
175: 5.8346,
179: 5.6394,
191: 5.0009,
193: 5.3035,
196: 5.8408,
198: 5.1136,
200: 6.3556,
22: 5.0258,
26: 5.5922,
30: 4.7008,
32: 4.9714,
33: 5.0752,
37: 4.7476,
40: 5.1324,
48: 5.4669,
59: 5.6712,
68: 6.8152,
71: 5.0628,
73: 5.5157,
78: 5.6630,
80: 5.0355,
82: 5.3881,
83: 4.5802,
88: 4.7158,
89: 5.4505}
lsd_loss_percentage_list = []

for id in range(len(val_ids)):
    left = left_val[id]['features'][:, :, :, 1:]
    right = right_val[id]['features'][:, :, :, 1:]
    sample_id = left_val.subject_ids[id]
    merge = np.ma.concatenate([left, right], axis=3)
    original_mask = np.all(np.ma.getmaskarray(left), axis=3)

    degree = 1
    hr_SHT = SphericalHarmonicsTransform(degree,
                                        left_val.row_angles,
                                        left_val.column_angles,
                                        left_val.radii,
                                        original_mask)
    coefficient = hr_SHT(merge)
    inversed_hrtf = torch.from_numpy(hr_SHT.inverse(coefficient).T).view(256, 1, 72, 12).unsqueeze(0)

    merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1).unsqueeze(0) # b x nbins x r x w x h
    print(merge.shape, inversed_hrtf.shape)
    mask = torch.all(torch.from_numpy(left_val[0]['features'].mask), axis=3)
    sh_only_lsd = spectral_distortion_metric(inversed_hrtf, merge, domain=domain).item()
    model_output_lsd = eval_resutls[sample_id]
    loss_percentage = sh_only_lsd / model_output_lsd
    print(f"id: {sample_id}, sh_only_lsd: {sh_only_lsd} model lsd: {model_output_lsd} percentage: {loss_percentage}")
    lsd_loss_percentage_list.append(loss_percentage)

    # save pickles
    inversed_hrtf_mag = torch.pow(10, inversed_hrtf / 20)[0].permute((1, 2, 3, 0))
    file_path = 'C:/Users/steph/Desktop/XuyiHu/output/checkpoints/3/2026-02-26_12-51-56/sh_distortion/'
    file_name = file_path + f'Sonicom_{sample_id}.pickle'
    with open(file_name, "wb") as file:
        pickle.dump(inversed_hrtf_mag, file) # r x w x h x nbins
print("=" * 20)
print(f"average loss percentage: {np.mean(lsd_loss_percentage_list)}")

# create sofa files
row_angles, column_angles, _, _ = get_dataset_info(config)
convert_to_sofa(file_path, config, row_angles, column_angles)
print('Created valid sofa files')

ild_itd_results = {10: {'ild': 0.9661169704028784, 'itd': 26.017554012345673},
100:{'ild': 1.2120802379875764,'itd': 24.377893518518515},
104:{'ild': 0.882870643303127,'itd': 19.24189814814815},
116:{'ild': 1.069257601006144,'itd': 23.630401234567902},
118:{'ild': 1.4593028407193716,'itd': 27.536651234567902},
128:{'ild': 1.9761930371781726,'itd': 24.715470679012345},
141:{'ild': 1.426447054086157,'itd': 28.38059413580247},
143:{'ild': 1.243504146657829,'itd': 23.919753086419753},
148:{'ild': 1.2279158934744012,'itd': 22.593557098765434},
149:{'ild': 1.9078360393945657,'itd': 41.08796296296296},
152:{'ild': 0.8912183632591761,'itd': 28.115354938271604},
166:{'ild': 1.3814961746578536,'itd': 32.6244212962963},
168:{'ild': 1.4927165988631612,'itd': 24.45023148148148},
169:{'ild': 2.017678045663881,'itd': 25.607638888888886},
172:{'ild': 1.100747656786397,'itd': 21.79783950617284},
173:{'ild': 2.7778534309939165,'itd': 35.46971450617284},
175:{'ild': 1.6286010491952414,'itd': 28.404706790123456},
179:{'ild': 1.7657408483127557,'itd': 23.50983796296296},
191:{'ild': 1.0484457968287344,'itd': 24.956597222222225},
193:{'ild': 1.0785176116059845,'itd': 18.542631172839506},
196:{'ild': 1.6906457046348426,'itd': 21.195023148148145},
198:{'ild': 1.3332456727381556,'itd': 26.37924382716049},
200:{'ild': 2.6106100218416532,'itd': 29.272762345679013},
22:{'ild': 1.4849959016437824,'itd': 23.268711419753085},
26:{'ild': 1.9281942534891032,'itd': 26.475694444444443},
30:{'ild': 1.1851465910854693,'itd': 26.62037037037037},
32:{'ild': 1.1685312844148266,'itd': 22.063078703703702},
33:{'ild': 1.2952475565635782,'itd': 23.341049382716047},
37:{'ild': 1.261407008056914,'itd': 27.464313271604937},
40:{'ild': 1.1125847487975191,'itd': 22.521219135802472},
48:{'ild': 1.5006458666345353,'itd': 28.477044753086417},
59:{'ild': 1.6068744156161183,'itd': 22.111304012345677},
68:{'ild': 1.02028013463547,'itd': 23.55806327160494},
71:{'ild': 1.4600086488776727,'itd': 25.245949074074073},
73:{'ild': 0.7485347839871584,'itd': 22.44888117283951},
78:{'ild': 1.5255571723960244,'itd': 22.47299382716049},
80:{'ild': 1.567226550215804,'itd': 27.295524691358022},
82:{'ild': 1.9040210078684725,'itd': 24.06442901234568},
83:{'ild': 1.4149294876036558,'itd': 22.063078703703702},
88:{'ild': 0.9854262490298732,'itd': 20.254629629629633},
89:{'ild': 1.4726188147815853,'itd': 24.06442901234568}}

hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path + '/sofa_min_phase')]
ild_percentage_list = []
itd_percentage_list = []
for file in hrtf_file_names:
    target_sofa_file = config.valid_target_path + '/sofa_min_phase/' + file
    generated_sofa_file = file_path + '/sofa_min_phase/' + file
    target_hrtf = ld.HRTF(target_sofa_file)
    generated_hrtf = ld.HRTF(generated_sofa_file)
    subject_id = ''.join(re.findall(r'\d+', file))
    ild_diff = hf.calculate_ild_difference(target_hrtf, generated_hrtf)
    itd_diff = hf.calculate_itd_difference(target_hrtf, generated_hrtf)
    model_ild_diff = ild_itd_results[sample_id]['ild']
    model_itd_diff = ild_itd_results[sample_id]['itd']
    ild_percentage = ild_diff / model_ild_diff
    itd_percentage = itd_diff / model_itd_diff
    ild_percentage_list.append(ild_percentage)
    itd_percentage_list.append(itd_percentage)
    print('-' * 20)
    print(f"id: {sample_id}, sh_only_ild: {ild_diff} model ild: {model_ild_diff} percentage: {ild_percentage}")
    print(f"id: {sample_id}, sh_only_itd: {itd_diff} model ild: {model_itd_diff} percentage: {itd_percentage}")
