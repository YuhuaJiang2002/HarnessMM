# Adroit dexterous-hand pilot

The pilot runs the Shadow Dexterous Hand in three Gymnasium-Robotics environments. Each task uses a published DAPG checkpoint as its closed-loop motor controller. [Adroit benchmark](https://robotics.farama.org/envs/adroit_hand/) · [DAPG checkpoints](https://github.com/aravindr93/hand_dapg)

| Environment | Task | Actuators |
| --- | --- | ---: |
| `AdroitHandPen-v1` | Rotate a pen in the hand to its target pose | 24 |
| `AdroitHandDoor-v1` | Operate the latch and swing the door open | 28 |
| `AdroitHandRelocate-v1` | Pick up a ball and move it to a target position | 30 |

The [frozen protocol](protocol.json) assigns ten independently seeded resets per task to two training, two validation, and six test cases. This runner opens the training cases. It uses the registered episode limit and logs the native success signal at each step. A pilot episode succeeds if any step meets the native criterion. The report also records the first successful step, the number of successful steps, and the final success flag.

- `motor` runs the benchmark's corresponding DAPG controller directly.
- `agent` sends the task instruction and the simulator's initial RGB image to GPT 6.2 Sol once. The reply selects a motor skill or abstains. A compatible skill drives the actuators using native observations at every step. An incompatible skill or abstention ends the episode before actuation.

## Prepare the controller weights

From `harnessMM-R/`, download the checkpoint release and the two MJRL modules needed to read it. The exporter accepts the checkpoint hashes in the protocol and writes NumPy weights with reference inputs and outputs from the original PyTorch network.

```sh
mkdir -p outputs/adroit/deps
deps="$PWD/outputs/adroit/deps"
git clone git@github.com:aravindr93/hand_dapg.git "$deps/hand_dapg"
git -C "$deps/hand_dapg" checkout --detach 9a1de23df6574e1b94e1da742e3326eb1f61c013
git clone git@github.com:aravindr93/mjrl.git "$deps/mjrl-source"
git -C "$deps/mjrl-source" checkout --detach 3871d93763d3b49c4741e6daeaebbc605fe140dc
mkdir -p "$deps/policy-code/mjrl/policies" "$deps/policy-code/mjrl/utils"
cp "$deps/mjrl-source/mjrl/policies/gaussian_mlp.py" "$deps/policy-code/mjrl/policies/"
cp "$deps/mjrl-source/mjrl/utils/fc_network.py" "$deps/policy-code/mjrl/utils/"

python3.11 -m venv outputs/adroit/export-runtime
export_python=outputs/adroit/export-runtime/bin/python
"$export_python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
"$export_python" -m pip install numpy==1.26.4
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$deps/policy-code" "$export_python" adroit/export_dapg.py \
  "$deps/hand_dapg/dapg/policies" "$deps/weights"
```

## Run the training pilot

Use a separate Python 3.12 environment for the simulator. The runner verifies the simulator versions and checks the NumPy controller against the saved PyTorch reference before resetting the simulator.

```sh
python3.12 -m venv outputs/adroit/runtime
outputs/adroit/runtime/bin/python -m pip install -r adroit/requirements.txt
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
outputs/adroit/runtime/bin/python adroit/run.py --mode motor \
  --weights "$deps/weights" --out outputs/adroit/motor
```

Agent mode also needs a renderer, curl, and a working Responses credential. Set `HMMR_CREDENTIAL_FILE` to an owner-only file containing the token and use `HMMR_PROVIDER_URL` if your endpoint differs from the default. For Mesa EGL software rendering on Linux, set `MESA_SOFTWARE_DEVICE` to the index of the EGL device advertising `EGL_MESA_device_software`.

```sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1 MUJOCO_EGL_DEVICE_ID="$MESA_SOFTWARE_DEVICE"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
outputs/adroit/runtime/bin/python adroit/run.py --mode agent \
  --weights "$deps/weights" --out outputs/adroit/agent
```

Use a fresh output directory for each run. It holds the protocol, runtime versions, weight hashes, full action traces and native scores. Agent mode also saves initial frames and provider receipts. If a request fails, the runner retains the available evidence and exits without replaying it.
