#pragma once
// Normal-space point TSDF fusion (task A, de-ghosting variant of fuse_voxel).
//
// The streaming stitcher deposits one point per pixel per emitted frame, so
// overlapping observations of a surface pile up as several near-coincident and,
// under residual misalignment, DOUBLED sheets — which read as ghosting. Plain
// voxel downsampling (fuse_voxel) cannot fix this: two sheets separated by more
// than a cell fall into different voxels and BOTH survive, just sparser.
//
// fuse_tsdf() instead builds a truncated signed-distance FIELD. Each point splats
// a confidence-weighted signed distance along its (camera-oriented) surface normal
// into the voxels its truncation band covers; a voxel straddling the surface holds
// the weighted-mean SDF of every sheet whose band reaches it, so its zero-crossing
// lands at the confidence-weighted MIDPLANE of the sheets — collapsing a doubled
// surface into one. Sheets farther apart than 2*trunc never share a voxel and stay
// distinct (they may be genuinely different surfaces). The extracted surface is one
// point per zero-crossing voxel, moved onto the level set along the averaged normal.
#include <cstdint>
#include <vector>

namespace da {

struct TsdfParams {
    // Voxel edge (world units). <=0 => voxel_frac * bbox-diagonal, because monocular
    // reconstructions live at an arbitrary metric scale (an absolute cm-voxel would
    // either no-op or collapse the whole cloud).
    float voxel = 0.f;
    // Voxel edge as a FRACTION of the bbox diagonal (used only when voxel<=0). This
    // is the scene-relative "fusion detail" knob. <=0 => 0.004 (0.4% of extent).
    float voxel_frac = 0.f;
    // Truncation half-width of the SDF band (world units). Two sheets merge only
    // where their bands overlap, i.e. gaps up to 2*trunc collapse. <=0 => trunc_mult*voxel.
    float trunc = 0.f;
    // Truncation as a MULTIPLE of the voxel edge (used only when trunc<=0). This is
    // the scene-relative "merge range" knob (max merged gap = 2*trunc_mult voxels).
    // <=0 => 4.
    float trunc_mult = 0.f;
    // PCA normal-estimation radius (world units). <=0 => 2.5*voxel.
    float normal_radius = 0.f;
    // Drop zero-crossing voxels backed by fewer than this many point deposits
    // (speckle/floater cull). <1 => 1.
    int   min_hits = 2;
};

// Exact integer-grid surface occupancy extracted by fuse_tsdf.  This is kept
// alongside the displaced point representation so downstream exporters never
// have to guess voxel coordinates from floating-point positions.
struct TsdfVoxel {
    int32_t x = 0, y = 0, z = 0;
    uint8_t r = 0, g = 0, b = 0;
    int first_frame = 0;
};

struct TsdfSurface {
    float voxel_size = 0.f;
    std::vector<TsdfVoxel> voxels;
};

// In-place normal-space TSDF fusion of a parallel point cloud (xyz: 3N, rgb: 3N,
// radius: N). Replaces the buffers with the extracted single-sheet surface and
// returns its point count. `weights` (length N) is per-point confidence used to
// weight the SDF average; if null, 1/(radius+eps) is used (near/high-confidence
// points, which carry the smaller radius, dominate). `cameras` (3*ncam camera
// centres, world frame) orients each normal toward its nearest camera so the SDF
// sign is globally consistent — REQUIRED for correct fusion; if empty, normals are
// left with PCA's arbitrary sign and fusion degrades to a denoise.
//
// `in_frame` (length N, optional): the input capture frame of each point. When
// given, each output surface voxel inherits the MINIMUM (first-observing) frame of
// the points that fed it, the output is sorted frame-major by that frame, and (if
// `out_frame` is non-null) the per-output-point frame is written there. This keeps
// the progressive/flythrough reveal — which reveals a frame-major prefix — valid
// after fusion; without it the output is voxel-sorted and reveal order is spatial.
int fuse_tsdf(std::vector<float>& xyz, std::vector<uint8_t>& rgb,
              std::vector<float>& radius, const TsdfParams& p,
              const std::vector<float>* weights = nullptr,
              const std::vector<float>* cameras = nullptr,
              const std::vector<int>* in_frame = nullptr,
              std::vector<int>* out_frame = nullptr,
              TsdfSurface* out_surface = nullptr);

} // namespace da
