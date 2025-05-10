import os
import matlab.engine
from pathlib import Path

upsampled_files = ['LAPtask2_3_1.sofa', 'LAPtask2_3_2.sofa', 'LAPtask2_3_3.sofa',
                   'LAPtask2_5_1.sofa', 'LAPtask2_5_2.sofa','LAPtask2_5_3.sofa',
                   'LAPtask2_19_1.sofa', 'LAPtask2_19_2.sofa', 'LAPtask2_19_3.sofa',
                   'LAPtask2_100_1.sofa', 'LAPtask2_100_2.sofa', 'LAPtask2_100_3.sofa']

baseline = 'GEP_GAN'

upsampled_path = f'C:/Users/steph/Desktop/XuyiHu/LAP_Task_2_Submissions/task2_submissions/{baseline}'
results_file = "C:/Users/steph/Desktop/XuyiHu/LAP_Task_2_Submissions/results.txt"
with open(results_file, 'a') as f:
    f.write(f'baseline: {baseline}\n')

s = '_FreeFieldCompMinPhase_48kHz.sofa'
target_ids = ['P0204', 'P0208', 'P0213',
              'P0203', 'P0207', 'P0212',
              'P0202', 'P0206', 'P0211',
              'P0201', 'P0205', 'P0210']
target_path = 'C:/Users/steph/Desktop/XuyiHu/SONICOM'

exp_name = ['3_1', '3_2', '3_3',
            '5_1', '5_2', '5_3',
            '19_1', '19_2', '19_3',
            '100_1', '100_2', '100_3']

eng = matlab.engine.start_matlab()
s = eng.genpath('C:/Users/steph/Desktop/XuyiHu/amtoolbox-1.6.0')
eng.addpath(s, nargout=0)
s = eng.genpath('C:/Users/steph/Desktop/XuyiHu/HRTF-neurips')
eng.addpath(s, nargout=0)

for i in range(len(upsampled_files)):
    target_id = target_ids[i]
    target_sofa = target_path + f'/{target_id}/HRTF/HRTF/48kHz/{target_id}_FreeFieldCompMinPhase_48kHz.sofa'
    generated_sofa = upsampled_path + f'/{upsampled_files[i]}'
    [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa, target_sofa, nargout=3)
    print(f"{exp_name[i]}: acc: {pol_acc1}, rms: {pol_rms1}, querr: {querr1}")
    with open(results_file, "a") as f:
        f.write(f"{exp_name[i]}: acc: {pol_acc1}, rms: {pol_rms1}, querr: {querr1}\n")