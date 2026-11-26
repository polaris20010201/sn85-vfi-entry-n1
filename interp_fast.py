"""Pipelined single-model interpolation entry: RIFE 4.15 or VFIMamba-S on every pair, all clips in one process.

    python3 interp_fast.py INPUT_DIR OUTPUT_DIR          (VFI_MODEL=rife_context (default) | rife | vfimamba)

Three stages overlapping across clips:
- decode (thread): ffmpeg decodes the next clip to raw I420 on the CPU (the inputs are FFV1, which NVDEC
  does not decode), threads capped;
- GPU (main): the clip goes to the GPU once through pinned memory; I420 -> RGB exactly as the scorer
  converts (BT.601 limited range, integer maths); the model on batches of pairs; RGB -> I420 on the GPU;
- encode (thread): NVENC lossless H.264 (bit-exact for yuv420p; libx264 qp 0 if NVENC is missing) while
  the GPU works on the next clip.
The model is loaded once and warmed with a dummy 1080p batch. Frame 2k is input frame k byte for byte.
Weights: weights/rife.pkl (RIFE 4.15, code in weights/rife/) or weights/vfimamba.pkl (code in vfimamba/).
rife_context adds the four-frame context module (motion_context.FourWarpRIFE, weights/context.pt): the new
frame between inputs k and k+1 also sees inputs k-1 and k+2 and the neighbouring pairs' flows, and corrects
the final flow, mask and colour of the frozen RIFE pass. Each frame is encoded and each pair's flow computed once.
If the context module or its weights cannot be loaded, the entry falls back to plain RIFE.
Always exits 0.
"""
from __future__ import annotations

import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
T0 = time.perf_counter()
CY, CUB, CUG, CVG, CVR, SHIFT = 1220542, 2116026, -409993, -852492, 1673527, 20
MODEL = os.environ.get("VFI_MODEL", "rife_context")
BATCH = int(os.environ.get("VFI_BATCH", "4" if MODEL == "vfimamba" else "8"))
CONTEXT = os.environ.get("CONTEXT_WEIGHTS", os.path.join(HERE, "weights", "context.pt"))
CONTEXT_BASE = os.environ.get("CONTEXT_BASE_WEIGHTS", os.path.join(HERE, "weights", "rife-context.pkl"))
# The context head is trained for one set of RIFE weights (its base_sha256); 1 runs it on others anyway (benchmarks).
CONTEXT_ANY_BASE = os.environ.get("CONTEXT_ANY_BASE", "0").strip() not in ("", "0", "false", "no")
THREADS = int(os.environ.get("VFI_THREADS", "4"))


def log(**kw):
    kw["t"] = round(time.perf_counter() - T0, 2)
    print(json.dumps(kw), file=sys.stderr, flush=True)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,color_range,color_space,color_primaries,color_transfer",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return s, int(s["width"]), int(s["height"]), Fraction(s["r_frame_rate"])


def colour_args(s):
    out = []
    for key, opt in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                     ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")):
        v = s.get(key)
        if v and v not in ("unknown", "unspecified", "reserved"):
            out += [opt, v]
    return out


def yuv_to_rgb(b, w, h):
    """uint8 I420 rows (n, w*h*3/2) on the GPU -> RGB in [0, 1] (n, 3, h, w), as the scorer converts."""
    b = b.to(torch.int32)
    n, cw, ch = b.shape[0], (w + 1) // 2, (h + 1) // 2
    y = b[:, :w * h].view(n, h, w)
    u = (b[:, w * h:w * h + cw * ch].view(n, ch, cw) - 128).repeat_interleave(2, 1).repeat_interleave(2, 2)[:, :h, :w]
    v = (b[:, w * h + cw * ch:].view(n, ch, cw) - 128).repeat_interleave(2, 1).repeat_interleave(2, 2)[:, :h, :w]
    yy = (y - 16).clamp_min(0) * CY
    half = 1 << (SHIFT - 1)
    rgb = torch.stack(((yy + CVR * v + half) >> SHIFT, (yy + CVG * v + CUG * u + half) >> SHIFT,
                       (yy + CUB * u + half) >> SHIFT), 1)
    return rgb.clamp_(0, 255).float().div_(255.0)


INV = torch.tensor(np.linalg.inv(np.array([[CY, 0, CVR], [CY, CUG, CVG], [CY, CUB, 0]], dtype=np.float64) / (1 << SHIFT)),
                   dtype=torch.float32)


def rgb_to_i420(rgb, w, h):
    """RGB (n, 3, h, w) in [0, 1] -> uint8 I420 rows (n, w*h*3/2), the exact inverse of the conversion above."""
    x = rgb[:, :, :h, :w].float().clamp(0, 1) * 255.0
    yuv = torch.einsum("ij,njhw->nihw", INV.to(x.device), x)
    y, uv = yuv[:, 0] + 16.0, yuv[:, 1:]
    if h % 2 or w % 2:
        uv = F.pad(uv, (0, w % 2, 0, h % 2), mode="replicate")
    uv = F.avg_pool2d(uv, 2) + 128.0
    n = rgb.shape[0]
    return torch.floor(torch.cat((y.reshape(n, -1), uv.reshape(n, -1)), 1) + 0.5).clamp_(0, 255).to(torch.uint8)


class Rife:
    """RIFE 4.15 (hzwer/Practical-RIFE, MIT): the release's IFNet in weights/rife/, scales [8, 4, 2, 1], padded to 64."""

    def __init__(self, dev, weights=None):
        self.ensemble = False       # IFNet's temporal ensemble: p01-p10 final 0.6939 vs 0.7161 plain, so always off
        sys.path.insert(0, os.path.join(HERE, "weights", "rife"))
        self.net = importlib.import_module("train_log.IFNet_HDv3").IFNet()
        sd = torch.load(weights or os.path.join(HERE, "weights", "rife.pkl"), map_location="cpu", weights_only=True)
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()
                                  if not k.replace("module.", "").startswith(("teacher.", "caltime."))})
        self.net.eval().to(dev)

    def __call__(self, a, b):
        n, _, h, w = a.shape
        x = F.pad(torch.cat((a, b), 1), (0, (-w) % 64, 0, (-h) % 64))
        return self.net(x, 0.5, [8, 4, 2, 1], ensemble=self.ensemble)[2][3][:, :, :h, :w].float()

    def reset(self):
        pass

    def pairs(self, gpu, k, j, w, h):
        """New frames for pairs k .. j-1 of the clip's I420 rows on the GPU."""
        return self(yuv_to_rgb(gpu[k:j], w, h), yuv_to_rgb(gpu[k + 1:j + 1], w, h))


class RifeContext(Rife):
    """RIFE 4.15 plus the four-frame context module (motion_context.FourWarpRIFE, our code and weights, MIT).

    Per clip it caches RGB frames, their context features and each pair's primary RIFE flow and mask, so a
    batch of pairs k .. j-1 computes only what the previous batch has not: frames up to j+1, flows up to pair j.
    Clip edges reuse the central frames with validity 0, as in training.
    """

    def __init__(self, dev):
        super().__init__(dev, weights=CONTEXT_BASE)
        import hashlib
        sys.path.insert(0, HERE)
        from motion_context import FourWarpRIFE
        ck = torch.load(CONTEXT, map_location="cpu", weights_only=True)
        if ck.get("variant") != "fourwarp":
            raise ValueError(f"unsupported context variant {ck.get('variant')!r}")
        with open(CONTEXT_BASE, "rb") as f:
            base = hashlib.sha256(f.read()).hexdigest()
        if base != ck["base_sha256"] and not CONTEXT_ANY_BASE:
            raise ValueError(f"context head was trained for RIFE weights {ck['base_sha256'][:12]}, not {base[:12]} "
                             "(CONTEXT_ANY_BASE=1 runs it anyway)")
        self.model = FourWarpRIFE(self.net, **ck["config"])
        self.model.context.load_state_dict(ck["context"], strict=True)
        self.model.eval().to(dev)
        self.step, self.matched = ck.get("step"), base == ck["base_sha256"]
        self.reset()

    def reset(self):
        self.rgb, self.feat, self.fields, self.quarter = {}, {}, {}, {}

    def pairs(self, gpu, k, j, w, h):
        n, ctx = gpu.shape[0], self.model.context
        new = [i for i in range(max(0, k - 1), min(n - 1, j + 1) + 1) if i not in self.rgb]
        if new:
            rgb = F.pad(yuv_to_rgb(gpu[new], w, h), (0, (-w) % 64, 0, (-h) % 64))
            feat = ctx.encode(rgb)
            for t, i in enumerate(new):
                self.rgb[i] = rgb[t:t + 1]
                self.feat[i] = (tuple(f[t:t + 1] for f in feat)
                                if isinstance(feat, tuple) else feat[t:t + 1])
        new = [p for p in range(max(0, k - 1), min(n - 2, j) + 1) if p not in self.fields]
        if new:
            x = torch.cat((torch.cat([self.rgb[p] for p in new]), torch.cat([self.rgb[p + 1] for p in new])), 1)
            flow, mask = self.model.primary(x, [8., 4., 2., 1.])
            quarter = self.model.quarter_flow(flow)
            for t, p in enumerate(new):
                self.fields[p] = (flow[t:t + 1], mask[t:t + 1])
                self.quarter[p] = quarter[t:t + 1]
        qs = range(k, j)
        cat = torch.cat
        left, right = [max(0, q - 1) for q in qs], [min(n - 1, q + 2) for q in qs]
        x = cat((cat([self.rgb[q] for q in qs]), cat([self.rgb[q + 1] for q in qs])), 1)
        fields = (cat([self.fields[q][0] for q in qs]), cat([self.fields[q][1] for q in qs]))
        def gather_features(indices):
            values = [self.feat[i] for i in indices]
            if isinstance(values[0], tuple):
                return tuple(cat(level) for level in zip(*values))
            return cat(values)

        features = [gather_features(idx) for idx in (left, list(qs), [q + 1 for q in qs], right)]
        outer = (cat([self.rgb[i] for i in left]), cat([self.rgb[i] for i in right]))
        valid = (torch.tensor([float(q > 0) for q in qs], device=x.device).view(-1, 1, 1, 1),
                 torch.tensor([float(q + 2 < n) for q in qs], device=x.device).view(-1, 1, 1, 1))
        zero = torch.zeros_like(self.quarter[k])
        neighbors = (cat([self.quarter[q - 1] if q > 0 else zero for q in qs]),
                     cat([self.quarter[q + 1] if q + 2 < n else zero for q in qs]))
        pred = self.model.synthesize(x, fields, features, valid, neighbors, outer)[2][-1]
        for d in (self.rgb, self.feat, self.fields, self.quarter):  # the next batch starts at frame / pair j-1
            for i in [i for i in d if i < j - 1]:
                del d[i]
        return pred[:, :, :h, :w].float()


class VFIMamba:
    """VFIMamba-S (MCG-NJU, Apache-2.0): the repository's model code in vfimamba/, padded to 32."""

    def __init__(self, dev):
        sys.path.insert(0, os.path.join(HERE, "vfimamba"))
        cfg = importlib.import_module("config")
        backbone, multiscale = cfg.MODEL_CONFIG["MODEL_TYPE"]
        bcfg, mcfg = cfg.MODEL_CONFIG["MODEL_ARCH"]
        self.net = multiscale(backbone(**bcfg), **mcfg)
        sd = torch.load(os.path.join(HERE, "weights", "vfimamba.pkl"), map_location="cpu", weights_only=True)
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items() if "attn_mask" not in k and "HW" not in k})
        self.net.eval().to(dev)
        self.local = cfg.LOCAL

    reset, pairs = Rife.reset, Rife.pairs

    def __call__(self, a, b):
        n, _, h, w = a.shape
        x = F.pad(torch.cat((a, b), 1), (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        return self.net(x, local=self.local, timestep=0.5)[3][:, :, :h, :w].float()


def main(input_dir, output_dir):
    torch.set_grad_enabled(False)
    torch.set_num_threads(THREADS)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    names = sorted(n for n in os.listdir(input_dir) if not n.startswith("."))
    try:
        model = {"vfimamba": VFIMamba, "rife_context": RifeContext}.get(MODEL, Rife)(dev)
    except Exception as e:  # noqa: BLE001  a missing or mismatched context head must not cost the batch
        if MODEL != "rife_context":
            raise
        log(phase="context_fallback", error=str(e)[:300])
        model = Rife(dev)
    warm = torch.randint(0, 256, (BATCH + 1, 1920 * 1080 * 3 // 2), dtype=torch.uint8, device=dev)
    model.pairs(warm, 0, BATCH, 1920, 1080)                         # warm-up: kernels and autotuning, once
    model.reset()
    del warm
    torch.cuda.synchronize()
    active_model = "rife_context" if isinstance(model, RifeContext) else ("rife" if isinstance(model, Rife) else MODEL)
    log(phase="load", model=active_model, requested_model=MODEL, batch=BATCH,
        **({"context": model.step, "context_base_matched": model.matched} if isinstance(model, RifeContext) else {}))
    dec_q: queue.Queue = queue.Queue(maxsize=2)
    enc_q: queue.Queue = queue.Queue(maxsize=2)

    def decoder():
        for name in names:
            src = os.path.join(input_dir, name)
            try:
                s, w, h, rate = probe(src)
                raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", str(THREADS), "-i", src, "-map", "0:v:0",
                                      "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
                fb = w * h * 3 // 2
                dec_q.put((name, s, w, h, rate, np.frombuffer(raw, np.uint8)[: len(raw) // fb * fb].reshape(-1, fb)))
            except Exception as e:  # noqa: BLE001
                log(item=name[:12], status="FAILED", stage="decode", error=str(e)[:200])
        dec_q.put(None)

    def encode(cmd, frames, mids):
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            p.stdin.write(frames[0].tobytes())
            for k in range(len(mids)):
                p.stdin.write(mids[k].tobytes())
                p.stdin.write(frames[k + 1].tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        return p.wait() == 0

    def encoder():
        while (job := enc_q.get()) is not None:
            name, s, w, h, rate, frames, mids, t_start = job
            partial = os.path.join(output_dir, f".partial-{name}")
            rate2, tags = rate * 2, colour_args(s)
            base = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                    "-r", f"{rate2.numerator}/{rate2.denominator}", *tags, "-i", "pipe:0"]
            codec = "h264_nvenc lossless"
            ok = encode(base + ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "lossless", "-pix_fmt", "yuv420p", *tags,
                                "-f", "mp4", partial], frames, mids)
            if not ok:
                codec = "libx264 qp0"
                ok = encode(base + ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-threads", str(THREADS),
                                    "-pix_fmt", "yuv420p", *tags, "-f", "mp4", partial], frames, mids)
            if ok:
                os.chmod(partial, 0o644)
                os.replace(partial, os.path.join(output_dir, name))
                log(item=name[:12], status="done", frames=2 * len(frames) - 1, codec=codec, s=round(time.perf_counter() - t_start, 2))
            else:
                log(item=name[:12], status="FAILED", stage="encode")

    td, te = threading.Thread(target=decoder, daemon=True), threading.Thread(target=encoder, daemon=True)
    td.start()
    te.start()
    batch = BATCH
    while (job := dec_q.get()) is not None:
        name, s, w, h, rate, frames = job
        t_start = time.perf_counter()
        try:
            n = len(frames)
            gpu = torch.from_numpy(frames).pin_memory().to(dev, non_blocking=True)
            out = torch.empty((n - 1, frames.shape[1]), dtype=torch.uint8, device=dev)
            model.reset()
            k = 0
            while k < n - 1:
                j = min(n - 1, k + batch)
                try:
                    mids = model.pairs(gpu, k, j, w, h)
                except torch.OutOfMemoryError:
                    if batch == 1:
                        raise
                    # Drop partial caches before retrying: their tensor views
                    # can retain an entire failed batch's GPU allocations.
                    model.reset()
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    log(item=name[:12], phase="oom_retry", batch=batch)
                    continue
                out[k:j] = rgb_to_i420(mids, w, h)
                k = j
            host = torch.empty(out.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(out)
            enc_q.put((name, s, w, h, rate, frames, host.numpy(), t_start))
        except Exception as e:  # noqa: BLE001  one bad item never aborts the batch
            torch.cuda.empty_cache()
            model.reset()
            log(item=name[:12], status="FAILED", stage="gpu", error=str(e)[:300])
    enc_q.put(None)
    te.join()


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:  # noqa: BLE001
        log(phase="fatal", error=str(e)[:300])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
