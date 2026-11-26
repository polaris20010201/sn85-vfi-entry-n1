# SN85 frame-interpolation entry (sn85-vfi-entry-n1): fine-tuned RIFE v4.15 on the GPU (n2_harder entry).
# The sandbox builder takes only plain FROM / COPY / RUN / ENV / WORKDIR, and every image
# reference carries a digest only. All downloads are pinned: base images by digest, Python
# wheels by sha256 (requirements.txt, --require-hashes).
#
# python:3.13-slim-bookworm, the base of the VidAIO images (same digest as the official example).
FROM python@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1

# ffmpeg + ffprobe 9.0, static build with libx264 and FFV1 (mwader/static-ffmpeg:9.0),
# the same build the VidAIO scorer decodes with.
COPY --from=mwader/static-ffmpeg@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# PyTorch 2.8.0 (CUDA 12.8 wheels, which bundle the CUDA runtime) and NumPy, every wheel by sha256.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r /app/requirements.txt

COPY run.sh interp_fast.py context.py motion_context.py /app/
COPY weights /app/weights
RUN chmod 0755 /app/run.sh \
 && find /app -type f ! -name run.sh -exec chmod 0644 {} + \
 && echo "2e95abe48a46438170d621e635a94e60c1215b789c71f6f4e8b5d81dfa968ff6  /app/weights/rife.pkl" | sha256sum -c - \
 && echo "1705c532810d815bc5034d02cd847816e7b786b1943519bdd02e9e86de34eca6  /app/weights/rife-context.pkl" | sha256sum -c - \
 && echo "90b03e4118fd503fd6726cc9bfbd5a1ff7f414763678ee91ed4e9565ddc30431  /app/weights/context.pt" | sha256sum -c -

# Fail the build, not the evaluation, when something the run relies on is missing.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
 && ffmpeg -hide_banner -decoders 2>/dev/null | grep -q ffv1 \
 && ffprobe -version >/dev/null \
 && command -v timeout >/dev/null \
 && python3 -c "import torch, numpy; assert torch.__version__.startswith('2.8.0'), torch.__version__; assert numpy.__version__ == '2.2.6'" \
 && python3 -I -c "import sys, torch, importlib; sys.path.insert(0, '/app/weights/rife'); \
net = importlib.import_module('train_log.IFNet_HDv3').IFNet(); \
sd = torch.load('/app/weights/rife.pkl', map_location='cpu', weights_only=True); \
net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items() if not k.replace('module.', '').startswith(('teacher.', 'caltime.'))}); \
sys.path.insert(0, '/app'); from interp_fast import RifeContext; \
assert RifeContext(torch.device('cpu')).matched"
