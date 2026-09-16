# Troubleshooting

Field notes from getting this pipeline running on a CUDA container with no working GPU
OpenGL. Most of these cost real time, and two of them are diagnostic traps that look like
answers but are not.

---

## The core problem

MuJoCo's default rendering path needs OpenGL. On a headless GPU container the usual
answer is EGL, selected with `MUJOCO_GL=egl`. When EGL cannot enumerate a display device,
MuJoCo silently falls back to Mesa's software rasteriser and everything still "works" —
just at roughly **157 ms/frame** instead of ~2 ms.

That gap is the whole story. At 157 ms/frame, four cameras × ~2 400 frames/episode is
about 25 minutes of pure rasterisation per episode. Nothing about parallelism helps,
because llvmpipe already saturates every core you give it.

**The fix used here: render on the GPU through MuJoCo Warp** (`mujoco_warp`), which uses
CUDA ray tracing and never touches EGL or OpenGL. See `docs/ARCHITECTURE.md`.

---

## Diagnostics that lie

### `hasattr(egl, "eglQueryDevicesEXT")` does not detect GLVND

`libEGL.so.1.1.0` **is** the GLVND dispatcher. libglvnd hands out extension entry points
through `eglGetProcAddress`, not as exported ELF symbols — so a symbol probe reports "not
GLVND" about the GLVND dispatcher itself. This produced a false negative that sent a whole
debugging session after an `apt-get install libegl1 libglvnd0` that was never needed.

If you want to know whether GLVND is in play, read the vendor JSONs under
`/usr/share/glvnd/egl_vendor.d/`. Do not probe symbols.

### Scanning `/proc/self/maps` for `swrast` or `llvmpipe` proves nothing

Modern Mesa merged both into `libgallium`. Their absence from the mapped-library list is
consistent with CPU rendering. Conversely, `libegl_nvidia` being mapped is consistent with
CPU rendering too — GLVND loads every vendor library it can find and still falls back to
Mesa when the NVIDIA one cannot open a display.

**The only trustworthy signal is frame time.** If a full scene renders at >100 ms/frame,
you are on the CPU, whatever the library list says. The collector's benchmark ranks
configurations by measured scene time for exactly this reason.

---

## `Cannot initialize a EGL device display`

Forcing the NVIDIA vendor library:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
python -c "import mujoco; mujoco.Renderer(...)"
# ImportError: Cannot initialize a EGL device display
```

Check the device nodes:

```bash
ls /dev/nvidia*
# nvidia-modeset  nvidia-uvm  nvidia-uvm-tools  nvidiactl  nvidia2
```

If the per-GPU nodes do not include `nvidia0` — only `nvidiactl` and `nvidia-modeset` —
NVIDIA's EGL cannot open a device display. This is a container configuration issue, not
something you can fix from inside the process. There is no userspace package that repairs
it; `apt-get install libnvidia-gl-<branch>` will fail with `Unable to locate package` in
an image with no NVIDIA apt repo.

Also check:

```bash
cat /proc/driver/nvidia/version     # often absent in containers
echo $NVIDIA_VISIBLE_DEVICES        # 'void' means "no GPUs" — contradictory if one is present
echo $NVIDIA_DRIVER_CAPABILITIES    # must include 'graphics', not just 'compute'/'utility'
```

`NVIDIA_VISIBLE_DEVICES=void` alongside a visible GPU is internally inconsistent and worth
raising with the platform, but it is not what blocks EGL — the missing `nvidia0` is.

---

## MuJoCo Warp specifics

### `forward` fails on some Blackwell + gVisor runtimes

Calling full forward dynamics compiles collision and solver kernels, which can fail during
compilation on these runtimes. Offline replay does not need them:

```python
# don't
mjw.forward(m, d)

# do — transforms only, no collision detection
mjw.fwd_kinematics(m, d)
mjw.refit_bvh(m, d, rc)
mjw.render(m, d, rc)
```

`fwd_kinematics` computes body, geom, **camera** and light transforms. That last pair is
what makes it sufficient: if a future version computes only body transforms, every episode
will contain four frozen camera views while the arms still move. Re-run the step-3 pilot
check after any upgrade.

### Renders come out white or yellow

MuJoCo Warp adds the model headlight on top of the explicit scene lights. With the v21
two-light rig this clips most pixels. The collector compensates:

```python
self.mjm.vis.headlight.active = 0
self.mjm.light_diffuse[:] *= 0.55
self.mjm.light_specular[:] *= 0.10
```

If your frames are blown out, this is the first thing to check.

### `noslip` is unsupported

MuJoCo Warp does not implement `noslip` contact iterations, and the v21 grasp depends on
them. Do not try to move physics to the GPU. Run contacts on the CPU and rendering on the
GPU; that is why the pipeline is split.

---

## Environment

### `labmaze` fails to build on Python 3.13

`dm_control` depends on `labmaze`, which has no 3.13 wheel. The dinner-table environment
uses only `dm_control.mjcf` and `dm_control.mujoco`, never the locomotion stack, so:

```bash
pip install dm-control --no-deps
```

after installing its real dependencies explicitly. Or use Python 3.12 and skip this
entirely.

### `KeyError: 'cfg'` running `task_demo.py`

The notebook parse failed. `task_demo.py` splits
`notebooks/01_environment_cells.py` on the literal strings `"# CELL 8 of 13"` and
`"# CELL 2 of 13"`. If those marker comments were edited, renamed, or reflowed, the split
produces a body that does not define `cfg`. Keep the markers byte-identical.

### Rendering is slow and the GPU looks idle

You are on llvmpipe. Measure a frame time — if it is >100 ms, EGL is not working and you
are on the CPU path. Either fix EGL or switch to `record_molab_mjwarp.py`.

### The GPU is idle and frames are not the problem

Check the dataloader, not the collector. Four 320×320 video streams is a heavy decode
load; a starved encoder looks identical to a slow renderer from the outside.
