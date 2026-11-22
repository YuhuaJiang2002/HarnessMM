# Render recorded tabletop trajectories with Cycles

Each episode from schema 3 includes the recipe, camera calibration and rigid poses needed to render it independently. The provided scene uses the measured positions, orientations, lens geometry and frame times. Cycles adds beveled polymer parts, brushed fixtures, fasteners, a Cartesian slide, a vacuum cup and tubing. The slide and housing are visual geometry. Collision, contact, reward and tool annotations come from MuJoCo. The workbench maps and image-based lighting are included. [Asset sources and license](assets/README.md)

Use Blender 5.0 with FFmpeg on the path. From `harnessMM-R`, set `EPISODE` to an extracted episode folder and run:

```bash
for view in overhead wrist; do
  blender -b --python-exit-code 1 --python rendering/render_blender.py -- \
    "$EPISODE" --view "$view" --out rendered --samples 32 --width 640
done
```

A Python environment with `bpy==5.0.1` can use `python rendering/render_blender.py` with the same arguments. CPU rendering is the default. Select `--device OPTIX` or `--device CUDA` in a compatible GPU allocation. Each view is a separate job, so episodes and cameras can be distributed across workers. Use a fresh destination per episode. Finished filenames are protected against accidental replacement.

- A normal run writes one H.264 stream per view at the recorded frame rate.
- `--still 48` renders recorded frame 48 as a PNG.
- `--keyframes --width 480 --samples 24` renders a JPEG at reset and every distinct tool-call boundary. Filenames are the zero-based frame indices in `tool_calls.jsonl`. Rejected calls share a boundary image with the preceding state.

The output keeps the native 4:3 field of view. For a new width, scale both focal lengths by the width ratio. Pixel-center principal points scale as `(c + 0.5) * ratio - 0.5`. The per-frame camera-to-world matrices apply directly in Blender’s camera coordinates. Rendering reads fixed trajectory data and makes no model calls.
