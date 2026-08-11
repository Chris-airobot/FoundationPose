# FoundationPose env setup (conda)

What we actually used on this machine (RTX 4090, CUDA toolkit at `/home/chris/cuda-12.8-min`).

```bash
cd /home/chris/Projects/internship/Samsung/FoundationPose

# 1) Create env
conda env create -f environment.yml
conda activate foundationpose

# 2) PyTorch (CUDA 12.8 wheels)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3) CUDA paths needed to compile extensions
export CUDA_HOME=/home/chris/cuda-12.8-min
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"

# 4) GPU extensions
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# 5) Remaining Python deps + mycpp
python -m pip install -r requirements.txt
bash build_all_conda.sh

# 6) Check
python check_env.py
```

## Weights layout

```text
weights/
  2023-10-28-18-33-37/   # refiner
    model_best.pth
    config.yml
  2024-01-11-20-02-45/   # scorer
    model_best.pth
    config.yml
```

If Google Drive unzip creates an extra nested folder name, move the inner timestamp folder up so it sits directly under `weights/`.

## Demo data

Extract under `demo_data/` (e.g. `demo_data/mustard0/`).

## Capture a RealSense box sequence (no ROS)

If company policy permits installing the official RealSense Python wrapper inside this Conda environment:

```bash
conda activate foundationpose
python -m pip install pyrealsense2

# Show connected cameras and their serial numbers.
python capture_realsense_box.py --list-devices

# Select one camera explicitly when multiple RealSense devices are connected.
python capture_realsense_box.py \
  --serial REPLACE_WITH_SERIAL \
  --output-dir demo_data/box_test
```

Press `S`, draw a tight rectangle around the box, and press Enter to start recording. The script aligns depth to color, converts depth to 16-bit millimetres, saves the first-frame mask, and writes the color-camera intrinsics to `cam_K.txt`. It records 300 frames by default; press `Q` to stop earlier. It refuses to write into a non-empty output directory.

Run the captured sequence with a dedicated debug directory (the demo clears that directory at startup):

```bash
python run_demo.py \
  --mesh_file box.obj \
  --test_scene_dir demo_data/box_test \
  --debug 2 \
  --debug_dir debug_box_test
```

## Run demo

```bash
conda activate foundationpose
cd /home/chris/Projects/internship/Samsung/FoundationPose
python run_demo.py
```
