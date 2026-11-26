# SN85 frame-interpolation entry (sn85-vfi-entry-n1)

`/bin/sh /app/run.sh <input_dir> <output_dir>`: for every input clip (FFV1, N frames), writes an
MP4 (H.264, lossless) with 2N − 1 frames at twice the frame rate. Frame 2k is input frame k, byte-exact;
frame 2k + 1 is synthesised between input frames k and k + 1.

Four-frame context is enabled by default. The original scored `n2_harder` weights
remain available with `VFI_MODEL=rife`; the historical scores below describe that mode.

## Method

- **Model:** RIFE v4 IFNet (Practical-RIFE, MIT), original network code in `weights/rife/`, unchanged.
  The weights (`weights/rife.pkl`) are our own fine-tune of the MIT v4 weights on licence-checked
  real footage (Wikimedia Commons 1080p60), with the competition's quality formula as the loss.
- **Pipeline (`interp_fast.py`):** all clips in one process, model loaded once and warmed up.
  Decode (ffmpeg, CPU thread) → GPU (I420 → RGB exactly as the scorer converts, RIFE on batches of
  pairs, RGB → I420) → encode (NVENC lossless H.264; libx264 qp 0 when NVENC is not available)
  overlap across clips.
- **`run.sh` always exits 0.** One failing clip never aborts the batch.

## Four-frame context module (`VFI_MODEL=rife_context`, default)

`context.py` / `motion_context.py` (our code, MIT, the same files as `sn85-entry-n1`) add a small module
(`FourWarpRIFE`, ~51k parameters) on top of the frozen RIFE v4 pass. For the new frame between inputs
k and k + 1 it also sees inputs k − 1 and k + 2: cached quarter-resolution features of all four frames and
the neighbouring pairs' RIFE flows. It predicts bounded corrections to the final flow, blend mask and colour,
plus small taps from the outer frames. Clip edges reuse the central frames with validity 0.

In `interp_fast.py` (`RifeContext`) it runs on the same batches of pairs as plain RIFE. Each frame is
converted and encoded once, and each pair's RIFE flow is computed once (N inputs take N − 1 RIFE passes);
only frames k − 1 … j + 1 and flows for pairs k − 1 … j stay cached per batch. The batched path matches
a synthetic per-pair reference to 4e-6 at batch sizes 1, 2, 3 and 8 (earlier CPU check).
With the shipped trained pair, the current CPU tests cover batch sizes 1, 2 and 8,
including two-frame clips; random inputs differ by at most 0.00013 due to batched
convolution rounding before the image warps.

**Weights:** four-frame mode ships the matched pair from `sn85-entry-n1`:
`weights/context.pt` (sha256 `29a491b5618128232e0b5368d35d6c60a8b015cb5808c93841a9ac5cddec36e0`)
and `weights/rife-context.pkl` (sha256 `1705c532810d815bc5034d02cd847816e7b786b1943519bdd02e9e86de34eca6`).
This uses the n1 backbone, not the Commons `n2_harder` backbone in `weights/rife.pkl`.
The context head's `base_sha256` must match `CONTEXT_BASE_WEIGHTS`; the check remains enabled.
`CONTEXT_ANY_BASE=1` bypasses it only for deliberate experiments.

To use the Commons backbone with four-frame context, train a matched head with
`sn85-entry-n1/train/train_head.py --init <dir>`, where `<dir>` contains the
RIFE code and a byte-for-byte copy of `weights/rife.pkl` at `train_log/flownet.pkl`.
Then set both `CONTEXT_BASE_WEIGHTS` and `CONTEXT_WEIGHTS` to that matched pair.
If context loading fails, the entry logs `context_fallback` and runs the original
plain RIFE weights. The load log reports both requested and active model modes.
Quality and GPU runtime of this newly enabled configuration have not been measured.
Run `python -m unittest test_four_frame` for CPU edge, batching and base-match checks.

No footage, frames or reference data are shipped or looked up. Every output frame is computed from the input clip.

## Weights

| | |
| --- | --- |
| File | `weights/rife.pkl` |
| sha256 | `2e95abe48a46438170d621e635a94e60c1215b789c71f6f4e8b5d81dfa968ff6` |
| Origin | `n2_harder` best checkpoint (Commons night + aerial phase). It did not improve on its starting point, so the weights are the same as `c2_harder` / `c1_initial` best @ step 250. |

## Options (environment)

`VFI_MODEL` (`rife_context` default, `rife`), `VFI_BATCH` (pairs per GPU batch, default 2, set in `run.sh`), `VFI_THREADS` (ffmpeg threads, default 4), `MAIN_SECONDS` (overall timeout, default 3300),
`CONTEXT_WEIGHTS` (default `weights/context.pt`), `CONTEXT_BASE_WEIGHTS` (default `weights/rife-context.pkl`), `CONTEXT_ANY_BASE` (default 0).

## Background and history

**The method it builds on (2020 →).** This entry is built on RIFE, *Real-Time Intermediate Flow Estimation for
Video Frame Interpolation* (Huang, Zhang, Heng, Shi, Zhou), first published on arXiv in November 2020
(arXiv:2011.06294) and later at ECCV 2022. RIFE belongs to the flow-based line of frame interpolation that includes
Super SloMo (2018) and DAIN (2019), but estimates the intermediate flow directly with a compact network (IFNet)
instead of computing bidirectional flow and reversing it, which makes it fast enough for 1080p video. Its author
continued the model in Practical-RIFE (v4.x); this entry uses the v4.15 network code verbatim
(commit `bbfd2ea`, MIT) and starts from its weights.

**This entry (2026).** Everything specific to this repository was made in October 2026 for Bittensor subnet 85's
frame-interpolation task:

| When | What |
| --- | --- |
| Oct 2026 | Practical-RIFE v4.15 weights fine-tuned on licence-checked Wikimedia Commons 1080p60 footage, with the competition's quality formula as the loss (`c1_initial` → `c2_harder` → `n2_harder`; the last phase did not improve, so `weights/rife.pkl` is the `c1_initial` / `c2_harder` best at step 250) |
| Oct 2026 | Four-frame context module (`context.py`, `motion_context.py`, `FourWarpRIFE`, ~51k parameters) written and trained in `sn85-entry-n1` / `sn85-entry-003b`; its head (`weights/context.pt`) is paired with the n1 backbone (`weights/rife-context.pkl`) |
| 16 Oct 2026 | This repository created: the pipelined entry (`interp_fast.py`), `run.sh`, the pinned `Dockerfile` and licence notes |

The git history of this repository starts on 16 Oct 2026; the earlier dates above are those of the published work
it builds on, not of this code.

## Licences

Our code and fine-tuned weights are MIT (`LICENSE`). Third-party components are listed in `LICENSES.md`.
All downloads in the `Dockerfile` are pinned by digest or sha256.
# sn85-vfi-entry-n1
