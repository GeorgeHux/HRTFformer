from data.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from configs.config import Config
import importlib
import pickle
import numpy as np
import torch
from trainer.utils import spectral_distortion_metric

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

for val_id in val_ids:
    left = left_val[val_id]['features'][:, :, :, 1:]
    right = right_val[val_id]['features'][:, :, :, 1:]
    sample_id = left_val.subject_ids[val_id]
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
    print(f"sh_only_lsd: {sh_only_lsd} model lsd: {model_output_lsd} percentage: {loss_percentage}")
    lsd_loss_percentage_list.append(loss_percentage)

print("=" * 20)
print(f"average loss percentage: {np.mean(lsd_loss_percentage_list)}")