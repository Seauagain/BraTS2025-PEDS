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



def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)

def model_runner():
    nnunet_cmd = nnunet_runner()
    subprocess.run(nnunet_cmd, shell=True, check=True)

    nnunet_cmd = nnunet_runner()
    subprocess.run(nnunet_cmd, shell=True, check=True)

    nnunet_cmd = nnunet_runner()
    subprocess.run(nnunet_cmd, shell=True, check=True)



def fast_model_runner():
    swin_cmd = swin_runner(input_path="/tmp/inputs/noskull", output_path="/output", save_npz=False)
    subprocess.run(swin_cmd, shell=True, check=True)
    os.system("rm /output/*.json")


def nnunet_runner(input_path="/tmp/inputs/skull", output_path="/tmp/outputs/nnunet", save_npz=True):
    """
    """
    datasetid = "20250614001"
    trainer = "nnUNetTrainer_1000epochs"
    fold = "ensemble"
    ckpt = "best"
    init = "0.7"

    input_path = Path(input_path)
    output_path = Path(output_path)

    maybe_make_dir(output_path)

    cmd = f"""nnUNetv2_predict \
         -i {input_path} \
         -o {output_path} \
         -d {datasetid} \
         -tr {trainer} \
         -c 3d_fullres \
         -chk checkpoint_{ckpt}.pth"""
        
    if fold == "ensemble":
        pass 
    else:
        cmd = cmd + f" -f {fold} "

    # 如果init不为空，添加gamma参数
    if init != "":
        cmd = cmd + f" -gamma {init} "

    if save_npz:
        cmd += f" --save_probabilities"
    
    return cmd

