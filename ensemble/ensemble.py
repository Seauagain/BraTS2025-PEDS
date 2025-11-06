import os
import shutil
import sys
# from glob import glob
import glob
import nibabel as nib
import numpy as np
from pathlib import Path
import time 
import tempfile
import subprocess
from multiprocessing import Pool
import argparse

import numpy as np
from pathlib import Path
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import load_json
from nnunetv2.inference.export_prediction import export_prediction_from_logits




def model_ensemble(input_base_dir, npz_base_dirs):
    
    sub_folders_path = sorted( glob.glob( os.path.join(input_base_dir, "*") ) )
    
    for sub_folder_path in sub_folders_path:
        sub_folder_path = Path(sub_folder_path)
        case_name = sub_folder_path.name
        
        for i , npz_base_dir in enumerate(npz_base_dirs):

            npz_path = Path( npz_base_dir) / f"{case_name}.npz" 
            npz = np.load(npz_path, allow_pickle=True)
            if i==0:
                prob = npz['probabilities']
            else:
                prob += npz['probabilities']
        
        out_list = []
        for i in range(0, prob.shape[0]):
            out = np.swapaxes(prob[i], 0, 2)  # 关键：轴交换
            out_list.append(out)
        prob = np.stack(out_list, axis=0).astype(np.float32)
        
        prob = prob / len(npz_base_dirs)
        seg = np.argmax(prob, axis=0)

        img_path = Path( npz_base_dir) / f"{case_name}.nii.gz" 
        img = nib.load(img_path)

        seg_path = Path( "/output" ) / f"{case_name}.nii.gz"

        nib.save(nib.Nifti1Image(seg.astype(np.int8), img.affine), seg_path)
        print("saved: ", seg_path) 

