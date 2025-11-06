import os
import shutil
import sys
from glob import glob
import nibabel as nib
import numpy as np
from pathlib import Path
import time 
import tempfile
import subprocess
from multiprocessing import Pool
import argparse


from skull_stripping.skull_stripping import skull_stripping
from runner.runner import *
from ensemble.ensemble import * 



INPUTS_BASE_DIR="/input"


def clear_tmp():
    """
    Clear /tmp/inputs and /tmp/outputs directories (simple version)
    """
    directories = ['/tmp/inputs', '/tmp/outputs']
    
    for directory in directories:
        try:
            # If directory exists, delete it completely
            if os.path.exists(directory):
                shutil.rmtree(directory)
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")


def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)


def move_files(inputs_base_dir, outputs_base_dir):
    """
    模态命名为数字后缀，且不同病例位于统一的outputs_base_dir
    """
    nnunetv2_rename_dict = {
    "-t1n.nii.gz": "_0000.nii.gz",
    "-t1c.nii.gz": "_0001.nii.gz",
    "-t2w.nii.gz": "_0002.nii.gz",
    "-t2f.nii.gz": "_0003.nii.gz",
    }

    maybe_make_dir(outputs_base_dir)

    inputs_base_dir = Path(inputs_base_dir)
    outputs_base_dir = Path(outputs_base_dir) 

    # 获取所有输入文件夹
    all_input_paths = sorted(Path(inputs_base_dir).iterdir())
    total_files = len(all_input_paths)

    for single_input_path in all_input_paths:
        case_name = single_input_path.name  
        for suffix, val in nnunetv2_rename_dict.items():
            filename = f"{case_name}{suffix}"
            new_filename = f"{case_name}{val}"
            file_path = single_input_path / filename
            dest_path = outputs_base_dir / new_filename
            shutil.copy2(file_path, dest_path) # 统一复制到outputs_base_dir
        

def prepare_input_files():

    move_files(inputs_base_dir=INPUTS_BASE_DIR, outputs_base_dir="/tmp/inputs/skull")
    skull_stripping(inputs_base_dir=INPUTS_BASE_DIR, outputs_base_dir="/tmp/inputs/noskull")


clear_tmp()

prepare_input_files()

model_runner()

model_ensemble(input_base_dir="/input",npz_base_dirs=["/tmp/outputs/nnunet", "/tmp/outputs/hff", "/tmp/outputs/swin"])