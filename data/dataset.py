import torch
import numpy as np
from torch.utils.data import Dataset

from .hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

def get_sample_coords(num_initial_points, dataset):
    """
    SONICOM: 72 x 12
        row = [-180., -175., -170., -165., -160., -155., -150., -145., -140., -135., -130., -125.,
               -120., -115., -110., -105., -100.,  -95.,  -90.,  -85.,  -80.,  -75.,  -70.,  -65.,
                -60.,  -55.,  -50.,  -45.,  -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,
                  0.,    5.,   10.,   15.,   20.,   25.,   30.,   35.,   40.,   45.,   50.,   55.,
                 60.,   65.,   70.,   75.,   80.,   85.,   90.,   95.,  100.,  105.,  110.,  115.,
                120.,  125.,  130.,  135.,  140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.,]

        col = [-45., -30., -20., -10.,   0.,  10.,  20.,  30.,  45.,  60.,  75.,  90.]
    
    HUTUBS: 72 x 19
        row = [-180., -170., -168., -165., -160., -156., -150., -144., -140., -135., -132., -130.,
               -120., -110., -108., -105., -100.,  -96.,  -90.,  -84.,  -80.,  -75.,  -72.,  -70.,
                -60.,  -50.,  -48.,  -45.,  -40.,  -36.,  -30.,  -24.,  -20.,  -15.,  -12.,  -10.,
                  0.,   10.,   12.,   15.,   20.,   24.,   30.,   36.,   40.,   45.,   48.,   50.,   
                  60.,  70.,   72.,   75.,   80.,   84.,   90.,   96.,  100.,  105.,  108.,  110.,  
                 120.,  130.,  132., 135.,  140.,  144.,  150.,  156.,  160.,  165.,  168.,  170.]
        col = [-90., -80., -70., -60., -50., -40., -30., -20., -10.,   0.,  10., 20.,  30.,  40.,  50.,  60.,  70.,  80.,  90.]
        masked coords: [(0, 2), (0, 16), 
                        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), 
                        (2, 1), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 14), (2, 15), (2, 17), 
                        (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 15), (3, 16), (3, 17), 
                        (4, 1), (4, 2), (4, 4), (4, 5), (4, 13), (4, 14), (4, 16), (4, 17), 
                        (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 14), (5, 15), (5, 16), (5, 17), 
                        (6, 1), (6, 2), (6, 3), (6, 5), (6, 13), (6, 15), (6, 16), (6, 17), 
                        (7, 1), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 14), (7, 15), (7, 17), 
                        (8, 1), (8, 2), (8, 4), (8, 5), (8, 13), (8, 14), (8, 16), (8, 17), 
                        (9, 1), (9, 2), (9, 3), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 15), (9, 16), (9, 17), 
                        (10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 14), (10, 15), (10, 16), (10, 17), 
                        (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), 
                        (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), 
                        (14, 1), (14, 2), (14, 3), (14, 4), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 14), (14, 15), (14, 16), (14, 17), 
                        (15, 1), (15, 2), (15, 3), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 15), (15, 16), (15, 17), 
                        (16, 1), (16, 2), (16, 4), (16, 5), (16, 13), (16, 14), (16, 16), (16, 17), 
                        (17, 1), (17, 3), (17, 4), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 14), (17, 15), (17, 17), 
                        (18, 1), (18, 2), (18, 3), (18, 5), (18, 13), (18, 15), (18, 16), (18, 17), 
                        (19, 1), (19, 2), (19, 3), (19, 4), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (19, 11), (19, 12), (19, 14), (19, 15), (19, 16), (19, 17), 
                        (20, 1), (20, 2), (20, 4), (20, 5), (20, 13), (20, 14), (20, 16), (20, 17), 
                        (21, 1), (21, 2), (21, 3), (21, 5), (21, 6), (21, 7), (21, 8), (21, 9), (21, 10), (21, 11), (21, 12), (21, 13), (21, 15), (21, 16), (21, 17), 
                        (22, 1), (22, 3), (22, 4), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (22, 12), (22, 14), (22, 15), (22, 17), 
                        (23, 1), (23, 2), (23, 3), (23, 4), (23, 5), (23, 13), (23, 14), (23, 15), (23, 16), (23, 17), 
                        (24, 2), (24, 16), 
                        (25, 1), (25, 2), (25, 3), (25, 4), (25, 5), (25, 13), (25, 14), (25, 15), (25, 16), (25, 17), 
                        (26, 1), (26, 3), (26, 4), (26, 6), (26, 7), (26, 8), (26, 9), (26, 10), (26, 11), (26, 12), (26, 14), (26, 15), (26, 17), 
                        (27, 1), (27, 2), (27, 3), (27, 5), (27, 6), (27, 7), (27, 8), (27, 9), (27, 10), (27, 11), (27, 12), (27, 13), (27, 15), (27, 16), (27, 17), 
                        (28, 1), (28, 2), (28, 4), (28, 5), (28, 13), (28, 14), (28, 16), (28, 17), 
                        (29, 1), (29, 2), (29, 3), (29, 4), (29, 6), (29, 7), (29, 8), (29, 9), (29, 10), (29, 11), (29, 12), (29, 14), (29, 15), (29, 16), (29, 17), 
                        (30, 1), (30, 2), (30, 3), (30, 5), (30, 13), (30, 15), (30, 16), (30, 17), 
                        (31, 1), (31, 3), (31, 4), (31, 6), (31, 7), (31, 8), (31, 9), (31, 10), (31, 11), (31, 12), (31, 14), (31, 15), (31, 17), 
                        (32, 1), (32, 2), (32, 4), (32, 5), (32, 13), (32, 14), (32, 16), (32, 17), 
                        (33, 1), (33, 2), (33, 3), (33, 5), (33, 6), (33, 7), (33, 8), (33, 9), (33, 10), (33, 11), (33, 12), (33, 13), (33, 15), (33, 16), (33, 17), 
                        (34, 1), (34, 2), (34, 3), (34, 4), (34, 6), (34, 7), (34, 8), (34, 9), (34, 10), (34, 11), (34, 12), (34, 14), (34, 15), (34, 16), (34, 17), 
                        (35, 1), (35, 2), (35, 3), (35, 4), (35, 5), (35, 13), (35, 14), (35, 15), (35, 16), (35, 17), 
                        (37, 1), (37, 2), (37, 3), (37, 4), (37, 5), (37, 13), (37, 14), (37, 15), (37, 16), (37, 17), 
                        (38, 1), (38, 2), (38, 3), (38, 4), (38, 6), (38, 7), (38, 8), (38, 9), (38, 10), (38, 11), (38, 12), (38, 14), (38, 15), (38, 16), (38, 17), 
                        (39, 1), (39, 2), (39, 3), (39, 5), (39, 6), (39, 7), (39, 8), (39, 9), (39, 10), (39, 11), (39, 12), (39, 13), (39, 15), (39, 16), (39, 17), 
                        (40, 1), (40, 2), (40, 4), (40, 5), (40, 13), (40, 14), (40, 16), (40, 17), 
                        (41, 1), (41, 3), (41, 4), (41, 6), (41, 7), (41, 8), (41, 9), (41, 10), (41, 11), (41, 12), (41, 14), (41, 15), (41, 17), 
                        (42, 1), (42, 2), (42, 3), (42, 5), (42, 13), (42, 15), (42, 16), (42, 17), 
                        (43, 1), (43, 2), (43, 3), (43, 4), (43, 6), (43, 7), (43, 8), (43, 9), (43, 10), (43, 11), (43, 12), (43, 14), (43, 15), (43, 16), (43, 17), 
                        (44, 1), (44, 2), (44, 4), (44, 5), (44, 13), (44, 14), (44, 16), (44, 17), 
                        (45, 1), (45, 2), (45, 3), (45, 5), (45, 6), (45, 7), (45, 8), (45, 9), (45, 10), (45, 11), (45, 12), (45, 13), (45, 15), (45, 16), (45, 17), 
                        (46, 1), (46, 3), (46, 4), (46, 6), (46, 7), (46, 8), (46, 9), (46, 10), (46, 11), (46, 12), (46, 14), (46, 15), (46, 17), 
                        (47, 1), (47, 2), (47, 3), (47, 4), (47, 5), (47, 13), (47, 14), (47, 15), (47, 16), (47, 17), 
                        (48, 2), (48, 16), 
                        (49, 1), (49, 2), (49, 3), (49, 4), (49, 5), (49, 13), (49, 14), (49, 15), (49, 16), (49, 17), 
                        (50, 1), (50, 3), (50, 4), (50, 6), (50, 7), (50, 8), (50, 9), (50, 10), (50, 11), (50, 12), (50, 14), (50, 15), (50, 17), 
                        (51, 1), (51, 2), (51, 3), (51, 5), (51, 6), (51, 7), (51, 8), (51, 9), (51, 10), (51, 11), (51, 12), (51, 13), (51, 15), (51, 16), (51, 17), 
                        (52, 1), (52, 2), (52, 4), (52, 5), (52, 13), (52, 14), (52, 16), (52, 17), 
                        (53, 1), (53, 2), (53, 3), (53, 4), (53, 6), (53, 7), (53, 8), (53, 9), (53, 10), (53, 11), (53, 12), (53, 14), (53, 15), (53, 16), (53, 17), 
                        (54, 1), (54, 2), (54, 3), (54, 5), (54, 13), (54, 15), (54, 16), (54, 17), 
                        (55, 1), (55, 3), (55, 4), (55, 6), (55, 7), (55, 8), (55, 9), (55, 10), (55, 11), (55, 12), (55, 14), (55, 15), (55, 17), 
                        (56, 1), (56, 2), (56, 4), (56, 5), (56, 13), (56, 14), (56, 16), (56, 17), 
                        (57, 1), (57, 2), (57, 3), (57, 5), (57, 6), (57, 7), (57, 8), (57, 9), (57, 10), (57, 11), (57, 12), (57, 13), (57, 15), (57, 16), (57, 17), 
                        (58, 1), (58, 2), (58, 3), (58, 4), (58, 6), (58, 7), (58, 8), (58, 9), (58, 10), (58, 11), (58, 12), (58, 14), (58, 15), (58, 16), (58, 17), 
                        (59, 1), (59, 2), (59, 3), (59, 4), (59, 5), (59, 13), (59, 14), (59, 15), (59, 16), (59, 17), 
                        (61, 1), (61, 2), (61, 3), (61, 4), (61, 5), (61, 13), (61, 14), (61, 15), (61, 16), (61, 17), 
                        (62, 1), (62, 2), (62, 3), (62, 4), (62, 6), (62, 7), (62, 8), (62, 9), (62, 10), (62, 11), (62, 12), (62, 14), (62, 15), (62, 16), (62, 17), 
                        (63, 1), (63, 2), (63, 3), (63, 5), (63, 6), (63, 7), (63, 8), (63, 9), (63, 10), (63, 11), (63, 12), (63, 13), (63, 15), (63, 16), (63, 17), 
                        (64, 1), (64, 2), (64, 4), (64, 5), (64, 13), (64, 14), (64, 16), (64, 17), 
                        (65, 1), (65, 3), (65, 4), (65, 6), (65, 7), (65, 8), (65, 9), (65, 10), (65, 11), (65, 12), (65, 14), (65, 15), (65, 17), 
                        (66, 1), (66, 2), (66, 3), (66, 5), (66, 13), (66, 15), (66, 16), (66, 17), 
                        (67, 1), (67, 2), (67, 3), (67, 4), (67, 6), (67, 7), (67, 8), (67, 9), (67, 10), (67, 11), (67, 12), (67, 14), (67, 15), (67, 16), (67, 17), 
                        (68, 1), (68, 2), (68, 4), (68, 5), (68, 13), (68, 14), (68, 16), (68, 17), 
                        (69, 1), (69, 2), (69, 3), (69, 5), (69, 6), (69, 7), (69, 8), (69, 9), (69, 10), (69, 11), (69, 12), (69, 13), (69, 15), (69, 16), (69, 17), 
                        (70, 1), (70, 3), (70, 4), (70, 6), (70, 7), (70, 8), (70, 9), (70, 10), (70, 11), (70, 12), (70, 14), (70, 15), (70, 17), 
                        (71, 1), (71, 2), (71, 3), (71, 4), (71, 5), (71, 13), (71, 14), (71, 15), (71, 16), (71, 17)]
    """
    if dataset.lower() == "sonicom":
        if num_initial_points == 100:
            # [-180., -160., -140., -120., -100.,  -80.,  -60.,  -40.,  -20.,   -5., 5.,   20.,   40.,   60.,   80.,  100.,  120.,  140.,  160.,  175.]
            row_idx = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35, 37, 40, 44, 48, 52, 56, 60, 64, 68, 71]
            col_idx = [1, 3, 4, 6, 8] # [-30, -10, 0, 20, 45]
            return [(i, j) for i in row_idx for j in col_idx]
        
        if num_initial_points == 27:
            row_idx = [0, 8, 16, 24, 32, 40, 48, 56, 64] #[-180.0, -140.0, -100.0, -60.0, -20.0, 20.0, 60.0, 100.0, 140.0]
            col_idx = [0, 4, 8]    #[-45.0, 0.0, 45.0]
            return [(i, j) for i in row_idx for j in col_idx]

        if num_initial_points == 19:
            row_idx = [[18, 27, 36, 45, 54], [12, 21, 27, 36, 45, 51, 60], [18, 27, 36, 45, 54], [24, 48]]
            col_idx = [1, 4, 7, 9]
            return [(row, col) for col, rows in zip(col_idx, row_idx) for row in rows]
        
        if num_initial_points == 18:
            row_idx = [0, 12, 24, 36, 48, 60] #[-180, -120, -60, 0, 60, 120]
            col_idx = [1, 4, 8]  # [-30, 0, 45]
            return [(i, j) for i in row_idx for j in col_idx]
        
        if num_initial_points == 8:
            row_idx = [0, 18, 36, 54]   # [-180.0, -90.0, 0.0, 90.0]
            col_idx = [2, 8]   # [-20, 45]
            return [(i, j) for i in row_idx for j in col_idx]
        
        if num_initial_points == 5:
            return [(24, 2), (24, 8), (36, 4), (48, 2), (48, 8)] # (-60,-20), (-60,45), (0,0), (60,-20), (60,45)
        
        if num_initial_points == 3:
            return [(24, 2), (36, 8), (48, 2)] # (-60,-20), (0,45), (60,-20)
        
        raise ValueError(f"dataset {dataset}, num_initial_points {num_initial_points} is not predefined!")
    elif dataset.lower() == "hutubs":
        if num_initial_points == 3:
            return [[24, 4], [36, 9], [48, 4]]
        raise ValueError(f"dataset {dataset}, num_initial_points {num_initial_points} is not predefined!")
    raise NotImplementedError(f"dataset {dataset} not supported yet!")
    
class MergeHRTFDataset(Dataset):
    def __init__(self, hrtf_loader, dataset, left_hrtf, right_hrtf, num_initial_points, max_degree=21, apply_sht=True, transform=None):
        super(MergeHRTFDataset, self).__init__()
        self.left_hrtf = left_hrtf
        self.right_hrtf = right_hrtf
        self.apply_sht = apply_sht
        if apply_sht:
            self.num_initial_points = num_initial_points
            if hrtf_loader == 'hrtfdata':
                self.row_angles, self.column_angles = left_hrtf.row_angles, left_hrtf.column_angles
            elif hrtf_loader == 'hartufo':
                self.row_angles, self.column_angles = left_hrtf.fundamental_angles, left_hrtf.orthogonal_angles
            else:
                raise ValueError(f"unrecognized hrtf loader: {hrtf_loader}")
            self.num_row_angles, self.num_col_angles = len(self.row_angles), len(self.column_angles)
            self.num_radii = len(self.left_hrtf.radii)
            self.degree = max(1, int(np.sqrt(num_initial_points) - 1))
            self.max_degree = max_degree
        self.transform = transform
        self.selected_coords = get_sample_coords(num_initial_points, dataset)

    def __getitem__(self, index: int):
        try:
            left = self.left_hrtf[index]['features'][:, :, :, 1:]
            right = self.right_hrtf[index]['features'][:, :, :, 1:]
            sample_id = self.left_hrtf.subject_ids[index]
            merge = np.ma.concatenate([left, right], axis=3)
            selected_rows = [coord[0] for coord in self.selected_coords]
            selected_cols = [coord[1] for coord in self.selected_coords]
            original_mask = np.all(np.ma.getmaskarray(left), axis=3)
            
            if self.apply_sht:
                # mask = np.ones((self.num_row_angles, self.num_col_angles, self.num_radii), dtype=bool)
                mask = original_mask.copy()
                # mask[selected_rows, selected_cols, :] = original_mask[selected_rows, selected_cols, :]
                mask[selected_rows, selected_cols, :] = False
                lr_SHT = SphericalHarmonicsTransform(self.degree, self.row_angles,
                                                    self.column_angles,
                                                    self.left_hrtf.radii,
                                                    mask)
                lr_coefficient = torch.from_numpy(lr_SHT(merge)) # [num_coefficients, nbins]
                hr_SHT = SphericalHarmonicsTransform(self.max_degree, self.row_angles,
                                                    self.column_angles,
                                                    self.left_hrtf.radii,
                                                    original_mask)
                hr_coefficient = torch.from_numpy(hr_SHT(merge).T)

                if self.transform is not None:
                    mean_lr, mean_full = self.transform[0]
                    std_lr, std_full = self.transform[1]
                    lr_coefficient = (lr_coefficient - mean_lr) / std_lr
                    hr_coefficient = (hr_coefficient - mean_full) / std_full
                
                merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1)  # nbins x r x w x h
                return {"lr_coefficient": lr_coefficient, "hr_coefficient": hr_coefficient,
                        "hrtf": merge, "mask": original_mask, "id": sample_id}
            else:
                merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1)  # nbins x r x w x h
                lr_hrtf = merge[:, :, selected_rows, selected_cols]
                lr_hrtf = lr_hrtf.reshape(lr_hrtf.shape[0], -1).T # [num_points, nbins]
                return {"lr_hrtf": lr_hrtf, "hr_hrtf": merge, "id": sample_id, "mask": original_mask}
        except Exception as e:
            print(f"[ERROR] Index {index} failed in __getitem__: {e}")
            raise
    def __len__(self):
        return len(self.left_hrtf)
    
class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)
    

class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        if self.batch_data is None:
            return None
        
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v) and k not in {'mask', 'id'}:
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)