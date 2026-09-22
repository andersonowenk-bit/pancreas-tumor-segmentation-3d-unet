# 3D U-Net for Pancreas and Tumor Segmentation on Abdominal CT

A 3D U-Net that labels every voxel of an abdominal CT scan as background, pancreas, or pancreatic tumor. Built with PyTorch and MONAI on the Medical Segmentation Decathlon Task07 Pancreas dataset, trained end to end on a free Kaggle GPU in 24 minutes.

## Results

Evaluated on 31 held-out CT volumes the model never saw during training.

| Metric | Score |
|---|---|
| Pancreas Dice | **0.6145** |
| Tumor Dice | 0.0150 |
| Training time | 24 min (Tesla T4) |

Dice measures overlap between the predicted and true segmentation (1.0 is a perfect match).

### Validation over training

| Epoch | Pancreas Dice | Tumor Dice |
|---|---|---|
| 10 | 0.5197 | 0.0111 |
| 15 | 0.5019 | 0.0140 |
| 20 | 0.5847 | 0.0169 |
| 25 | 0.5906 | 0.0144 |
| 30 | 0.6145 | 0.0150 |

Full per-epoch loss and validation history is in `history.json`.

## How it works

**Data.** 281 portal venous phase CT scans with voxel labels (0 background, 1 pancreas, 2 tumor). Split: 250 training, 31 validation.

**Preprocessing.** Reoriented to RAS, resampled to 1.5 x 1.5 x 2.0 mm spacing, HU window clipped to [-96, 215] and scaled to [0, 1], cropped to the body.

**Training.** Random 96 x 96 x 96 patches sampled with a 1:1:3 background:pancreas:tumor ratio so the model sees rare tumor voxels more often. Random flips and intensity shifts for augmentation.

**Model.** MONAI 3D U-Net with residual units, channels (16, 32, 64, 128, 256).

**Optimization.** DiceCE loss, AdamW (lr 3e-4), cosine learning rate schedule, mixed precision, 30 epochs, batch size 2.

**Inference.** Sliding window over the full volume with 25% overlap.

## How to run

1. Open `pancreas_scan_final.ipynb` on [Kaggle](https://www.kaggle.com).
2. Add the Task07 Pancreas dataset as an input.
3. Set Accelerator to **GPU T4**.
4. Run all cells. Cell 1 checks the GPU and installs MONAI, cell 2 verifies the data, cell 3 trains and saves checkpoints to `/kaggle/working`.

## Trained model

Download `best_model.pth` from the [Releases](../../releases) page. Load it with:

```python
import torch
from monai.networks.nets import UNet

net = UNet(spatial_dims=3, in_channels=1, out_channels=3,
           channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
           num_res_units=2)
ckpt = torch.load("best_model.pth", map_location="cpu")
net.load_state_dict(ckpt["state_dict"])
net.eval()
```

## Limitations and next steps

Pancreas segmentation is working. Tumor segmentation is not yet: tumors are small and often low contrast, and 30 epochs is far short of what published models use. Planned improvements:

- Train longer (300+ epochs)
- Use a larger network or nnU-Net style configuration
- Add a per-case tumor detection metric (sensitivity) alongside Dice
- Cross-validation instead of a single split

## Dataset

Medical Segmentation Decathlon, Task07 Pancreas (Memorial Sloan Kettering Cancer Center), licensed CC BY-SA 4.0. The data is not included in this repo.

> Antonelli, M. et al. The Medical Segmentation Decathlon. *Nature Communications* 13, 4128 (2022).

## Disclaimer

This is a learning and portfolio project. It is not a medical device and must not be used for diagnosis or any clinical decision.

## License

Code is released under the MIT License.
