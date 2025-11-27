![alt text](assets/brats2025_logo.png "Title")

<div align="center">

# Frequency-Aware Ensemble Learning for BraTS 2025 Pediatric Brain Tumor Segmentation
[[Paper]](https://arxiv.org/pdf/2509.19353) | [[BraTS2025]](https://www.synapse.org/Synapse:syn64153130/wiki/630130)
</div>


We propose an ensemble approach integrating nnU-Net, Swin UNETR, and HFFNet for the BraTS-PED 2025 challenge. Our method incorporates three
key extensions: 
- Adjustable initialization scales for optimal nnU-Net complexity control.
- Transfer learning from BraTS 2021 pre-trained model
to enhance Swin UNETR’s generalization on pediatric dataset.
- Frequency domain decomposition for HFF-Net to separate low-frequency
tissue contours from high-frequency texture details.

## News
🚩[**2025.11**] The source code of our solution has been released.

🚩[**2025.10**] Our solution achieves **🥇rank 1st** in the
BraTS 2025 Pediatric Brain Tumor Segmentation Challenge.

🚩[**2025.09**] We are invited to give an **oral presentation** during the *MICCAI BraTS 2025 Challenge Workshop* on 23 September 2025.


## Installation


## Acknowledgements
Our work builds upon several excellent prior methods. We thank the authors for open-sourcing their code, including the famous [nnUnet](https://github.com/MIC-DKFZ/nnUNet), [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21), and the recent [HFF-Net](https://github.com/VinyehShaw/HFF). For more implementation details, please refer to their original repositories.

