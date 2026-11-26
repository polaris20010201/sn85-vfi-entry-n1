# Third-party components

Everything this entry ships besides its own code (MIT, see `LICENSE`). All of it allows
commercial use. No video footage of any kind is shipped.

| Component | Version | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| Practical-RIFE: `IFNet_HDv3.py`, `warplayer.py` (verbatim) and the v4.15 weights the shipped weights are fine-tuned from | v4.15, commit `bbfd2ea90910789a860ea3e2b32a240cd577b75e` | https://github.com/hzwer/Practical-RIFE | MIT (code and model downloads, "under the same MIT license as this project"); licence text in `weights/rife/LICENSE` | `weights/rife/train_log/IFNet_HDv3.py`, `weights/rife/model/warplayer.py` |
| Fine-tuned RIFE v4.15 weights (our derivative of the MIT weights above; trained by us, no other model involved) | 2026-10 | this repository and `sn85-entry-003b` | MIT | `weights/rife.pkl`, `weights/rife-context.pkl` |
| Four-frame context module code (`context.py`, `motion_context.py`, written by us) | 2026-10 | this repository | MIT | `context.py`, `motion_context.py` |
| Trained four-frame context head, paired with `weights/rife-context.pkl` | soup4 | `sn85-entry-003b` | MIT | `weights/context.pt` |
| Python, official `python:3.13-slim-bookworm` image | 3.13, pinned by digest | https://hub.docker.com/_/python | PSF-2.0 (Python); the Debian 12 packages under their own free licences | base image |
| FFmpeg and ffprobe, static build `mwader/static-ffmpeg` | 9.0, pinned by digest | https://github.com/wader/static-ffmpeg | GPL-3.0-or-later (built with `--enable-gpl --enable-version3`), run as a separate program | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe` |
| PyTorch | 2.8.0+cu128, pinned by sha256 | https://pytorch.org | BSD-3-Clause | Python site-packages |
| NVIDIA CUDA 12.8 runtime libraries shipped as PyTorch's wheel dependencies (cuBLAS, cuDNN, cuFFT, cuFile, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, NCCL, NVRTC, nvJitLink, NVTX, CUPTI, CUDA runtime) | as pinned in `requirements.txt` | https://pypi.org/user/nvidia/ | NVIDIA software licences that permit redistribution with applications (the standard PyTorch CUDA build) | Python site-packages (`nvidia/`) |
| Triton | 3.4.0 | https://github.com/triton-lang/triton | MIT | Python site-packages |
| NumPy | 2.2.6 | https://numpy.org | BSD-3-Clause; the wheel bundles OpenBLAS (BSD-3-Clause) and the GCC Fortran runtime (GPL-3.0 with the GCC Runtime Library Exception) | Python site-packages |
| SymPy, mpmath, NetworkX, Jinja2, MarkupSafe, fsspec (PyTorch dependencies) | as pinned in `requirements.txt` | PyPI | BSD-3-Clause | Python site-packages |
| filelock, setuptools, typing-extensions (PyTorch dependencies) | as pinned in `requirements.txt` | PyPI | Unlicense, MIT, PSF-2.0 | Python site-packages |
