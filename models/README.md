# Model files (not included, not committed)

## SAM 2.1 checkpoint (required when you pass `--points`)
Official source: <https://github.com/facebookresearch/sam2> (the repo's `checkpoints/download_ckpts.sh` lists the URLs).
Check the license in that repository before use (at the time of writing the code and SAM 2.1 weights are Apache-2.0 -
verify, do not rely on this note).

| file | size (approx.) | `--sam2-config` (inferred from the file name if omitted) |
|---|---|---|
| `sam2.1_hiera_tiny.pt` | ~155 MB | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| `sam2.1_hiera_small.pt` | ~185 MB | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| `sam2.1_hiera_base_plus.pt` | ~325 MB | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `sam2.1_hiera_large.pt` | ~900 MB | `configs/sam2.1/sam2.1_hiera_l.yaml` |

Download (PowerShell), e.g. the small model:
```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -OutFile models\sam2.1_hiera_small.pt
```
Larger models track better but need more VRAM per chunk; on a 12 GB GPU `small` or `base_plus` with 120-frame chunks is a sensible start.

## NudeNet
`pip install nudenet==3.4.2` already contains the 320x320 model (`320n.onnx`, 12 MB). A 640 model is not bundled; if you obtain one
from the NudeNet project, pass `--nudenet-model path --nudenet-res 640`.
