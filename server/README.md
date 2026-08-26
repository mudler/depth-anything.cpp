# DA3 studio — demo server

A browser app (in the style of free-splatter.cpp's demo) that turns **videos or
photos into a coherent 3D point cloud** — or **3D gaussians** with DA3-Giant —
using the depth-anything.cpp (DA3) engine, rendered live in a WebGL splat viewer.

* **Browse sample scenes** — a gallery of baked scenes with frame-by-frame
  "build-up" playback (`acc_2 … acc_N`).
* **Create** — upload a video (→ one cross-view DA3 pass → fused cloud) or photos
  (→ multi-view cloud, or single-image gaussians), pick the **model** and the
  **output type** (point cloud / voxels / gaussians), and watch it appear.
* **Merged voxel playback** — fused voxel scenes can include a compact DTVM
  exposed-face timeline. Each face is stored once with the frame interval in
  which it is visible, so every build-up step is a merged surface without storing
  a complete accumulated mesh per frame.

The viewer core (EWA splatting, depth-sort worker, camera) is the same proven
renderer as free-splatter.cpp; it consumes antimatter15 `.splat` (32 B/record).
Voxel scenes retain that file as a compatible cube fallback and may additionally
carry `temporal_faces.dtvm`, generated from the TSDF's exact integer grid.

## How it works

```
video ──ffmpeg──▶ frames ──┐
photos ────────────────────┴─▶ da_capi_points_multi  (ONE cross-view pass)
                               → per-view depth+pose in one shared world frame
                               → back-project → fused cloud → .splat  (coherent)

photo  ─▶ da_capi_gaussians (DA3-Giant, 224²) → anisotropic gaussians → .splat
```

Inference is in-process via **purego** FFI to `libdepthanything.so` (no cgo).
Models are loaded lazily and LRU-evicted (`-max-live`, default 1) so the big
~5 GB checkpoints don't all sit in RAM at once.

## Build & run

```bash
# 1. shared library (from the repo root)
cmake -B build -DDA_SHARED=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2. curated model set (~13 GB, one weight per capability)
bash scripts/fetch_models.sh

# 3. the server (CGO not needed — purego)
cd server
CGO_ENABLED=0 go build -o da3-server .
./da3-server -lib ../build/libdepthanything.so -models-dir ../models

# open http://localhost:8794
```

### Flags

| flag | default | meaning |
|------|---------|---------|
| `-addr` | `:8794` | listen address |
| `-lib` | `../build/libdepthanything.so` | shared library path |
| `-models-dir` | `../models` | directory of `.gguf` weights |
| `-scenes-dir` | `../.cache/da3-scenes` | baked/uploaded scenes |
| `-work-dir` | `/tmp/da3-demo` | scratch for uploads/frames |
| `-threads` | `12` | inference threads |
| `-max-live` | `1` | max resident model contexts (LRU) |
| `-max-splats` | `1500000` | cap splats per output |

## Models

Catalogued in `models.go`; only those present on disk appear in the picker.

| model | output | notes |
|-------|--------|-------|
| `da3-small` / `da3-base` / `da3-large` | point cloud | relative depth + pose, increasing quality |
| `da3-giant` | point cloud **or gaussians** | adds the GS head |
| `da3-nested-metric` | point cloud | anyview + metric branches (real-metre scale, single-image) |

## API

| method · path | purpose |
|---|---|
| `GET /api/models` | available models + capabilities + default |
| `POST /api/reconstruct` | photos (+`model`,`mode`) → `{id,n,size,seconds}` |
| `GET /api/splat/{id}` | the `.splat` bytes |
| `GET /api/scenes` | list baked/uploaded scenes |
| `POST /api/scene/from-video` | video (+`model`,`mode`,`max_frames`,`conf_pct`) → `{job}` (async) |
| `GET /api/scene/status/{job}` | `{state,total,done,kept,scene}` |
| `GET /scenes-assets/...` | a scene's `manifest.json` + `.splat`/`.dtvm` + thumbnails |

## Known limitations

* **Point-cloud build-up** slices ONE coherent cross-view pass by frame; it is
  bounded by cross-view attention (`-max-frames`, default 12 per pass). Long
  time-lapses need windowing + chunk alignment (future work).
* **Gaussians** use the engine's GS head, which is fixed to a **224×224** input
  (16×16 patches); the server resizes accordingly. Single-image feed-forward
  gaussians are a per-pixel depth surface — sharp from the original viewpoint,
  thin edge-on. Multi-view gaussian fusion is future work.
* The `da3-nested-metric` model gives metric scale on the single-image path; the
  multi-view fuse currently runs its anyview branch (relative, still coherent).
* Temporal merged meshes currently use one quad per exposed voxel face. Internal
  faces are culled at every playback step; greedy coplanar rectangle merging is a
  future size/triangle-count optimization.
