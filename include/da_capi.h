#ifndef DA_CAPI_H
#define DA_CAPI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct da_ctx da_ctx;
/* ABI version. 3: added da_capi_depth_dense, da_capi_points, da_capi_free_bytes.
   4: added da_capi_load_nested (two-branch metric model).
   5: added da_capi_points_multi (fused multi-view cloud) + da_capi_gaussians.
   6: added da_capi_points_stream (sliding-window streaming cloud); removed the
      unused da_capi_depth_pose_multi.
   7: da_capi_gaussians also returns the input camera intrinsics + processed size
      (out_intr[9], out_w, out_h) so the viewer can render from the input view.
   8: da_capi_points_stream gained a `metric` flag (rescale the streamed cloud to
      metres via the nested metric branch; requires a da_capi_load_nested ctx).
   9: added da_capi_stream_last_poses — per-input-frame camera poses from the last
      da_capi_points_stream call (for the viewer flythrough).
   10: added da_capi_set_fuse_params — scene-relative TSDF voxel/truncation knobs for
      the next streamed fuse (exposed as viewer sliders).
   11: added temporal exposed-face voxel mesh generation and retrieval for streamed
      fused scenes. */
int         da_capi_abi_version(void);
/* Set scene-relative TSDF surface-fusion knobs consumed by the NEXT
   da_capi_points_stream whose fuse flag is set. voxel_frac: voxel edge as a fraction
   of the reconstruction's bbox diagonal (fusion detail; <=0 => 0.004). trunc_mult:
   truncation as a multiple of the voxel edge (merge range = 2*trunc_mult voxels;
   <=0 => 4). Split out because da_capi_points_stream is at the purego arg ceiling. */
void        da_capi_set_fuse_params(da_ctx* ctx, double voxel_frac, double trunc_mult);
/* Enable/disable a DTVM temporal exposed-face mesh for the NEXT streamed fuse.
   The mesh retains exact TSDF integer voxel keys and face lifetimes derived from
   each voxel's first-observing frame. Disabled by default. */
void        da_capi_set_temporal_voxel_mesh(da_ctx* ctx, int enabled);
da_ctx*     da_capi_load(const char* gguf_path, int n_threads);  /* NULL on failure */
/* Load a NESTED metric model from its two branches: the anyview (GIANT) GGUF and
   the metric (ViT-L + DPT/sky) GGUF. The returned ctx runs the nested metric
   alignment: da_capi_depth_dense / da_capi_depth_path / da_capi_pose_path all
   produce the final metric-scale depth + scaled extrinsics (is_metric=1, conf/sky
   = NULL). NULL on failure. */
da_ctx*     da_capi_load_nested(const char* anyview_gguf, const char* metric_gguf, int n_threads);
void        da_capi_free(da_ctx* ctx);                           /* safe on NULL */
/* malloc'd JSON describing model config; free via da_capi_free_string. */
char*       da_capi_info_json(da_ctx* ctx);
void        da_capi_free_string(char* s);
const char* da_capi_last_error(da_ctx* ctx);                     /* owned by ctx, "" if none */
/* Run depth on an image file. On success writes *out_h,*out_w and returns a malloc'd
   float[H*W] depth map (row-major); caller frees via da_capi_free_floats. NULL on error. */
float* da_capi_depth_path(da_ctx* ctx, const char* image_path, int* out_h, int* out_w);
void   da_capi_free_floats(float* p);
/* Run pose; fills ext[12] (3x4 row-major) and intr[9] (3x3). Returns 0 ok, -1 error. */
int da_capi_pose_path(da_ctx* ctx, const char* image_path, float out_ext[12], float out_intr[9]);
/* Single-image 3D export. Runs the native depth+pose pipeline, captures the
   processed-resolution RGB colors, and writes a glTF-2.0 binary point cloud to
   out_glb. Returns 0 ok, -1 error (see da_capi_last_error). */
int da_capi_export_glb(da_ctx* ctx, const char* image_path, const char* out_glb);
/* Single-image 3D export to a COLMAP sparse model (cameras/images/points3D) in
   directory out_dir. binary != 0 => .bin (default); 0 => .txt. Returns 0 ok, -1 error. */
int da_capi_export_colmap(da_ctx* ctx, const char* image_path, const char* out_dir, int binary);

/* Dense per-pixel output for a single image. Returns 0 ok, -1 error.
   Writes processed dims to *out_h,*out_w. Each non-NULL out_* float buffer is
   malloc'd [H*W] row-major and must be freed via da_capi_free_floats; buffers
   not produced by the model are set to NULL.
     - DualDPT model (camera-pose capable): *out_depth + *out_conf are filled,
       *out_sky = NULL, out_ext[12] (3x4 row-major) + out_intr[9] (3x3) filled.
     - mono model (DA3MONO): *out_depth + *out_sky are filled, *out_conf = NULL,
       out_ext/out_intr zeroed (mono has no camera pose).
     - nested model (da_capi_load_nested): *out_depth = final metric-scale depth,
       *out_conf = *out_sky = NULL, out_ext/out_intr = scaled extrinsics/intrinsics.
   *out_is_metric = 1 for metric/nested/mono variants (best-effort from config),
   else 0. Any of out_h/out_w/out_depth/out_conf/out_sky/out_is_metric may be NULL;
   out_ext/out_intr must point to 12/9 floats respectively. */
int da_capi_depth_dense(da_ctx* ctx, const char* image_path, int* out_h, int* out_w,
                        float** out_depth, float** out_conf, float** out_sky,
                        float out_ext[12], float out_intr[9], int* out_is_metric);

/* Single-image 3D point cloud (DualDPT/pose-capable models only; returns -1 for
   mono models with a clear last_error). Runs depth+pose+processed-RGB, back-projects
   to world space keeping pixels with conf >= conf_thresh. On success sets *out_n
   and writes a malloc'd *out_xyz[3*N float] + *out_rgb[3*N uint8]; free xyz via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points(da_ctx* ctx, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb);
/* Free a uint8 buffer returned by da_capi_points (out_rgb). */
void da_capi_free_bytes(unsigned char* p);

/* Multi-view FUSED colored point cloud (DualDPT/pose-capable DA3 models; returns
   -1 for mono/DA2). Runs ONE cross-view depth+pose pass over n_images and back-
   projects every view into a single shared world frame, so the cloud is coherent
   across frames (no per-frame scale drift). Recovers per-pixel processed RGB and a
   perspective-correct per-point radius (~depth/focal * point_size). Keeps pixels
   with confidence >= percentile(conf, conf_pct) (conf_pct in [0,100]). Points are
   emitted grouped by source view (frames outer); if out_counts != NULL it must
   point to int[n_images] and receives the per-view point count (for progressive
   build-up rendering). On success sets *out_n and mallocs *out_xyz[3N] (world xyz,
   OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free xyz/radius via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points_multi(da_ctx* ctx, const char** image_paths, int n_images,
                         double conf_pct, float point_size,
                         int* out_n, int* out_counts,
                         float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Sliding-window STREAMING colored point cloud (DualDPT/pose-capable DA3 models;
   returns -1 for mono/DA2). For long ordered frame sequences (time-lapse video):
   runs the fused cross-view pass on overlapping windows of chunk_size frames
   (sharing `overlap` frames) and stitches each window into ONE global frame via a
   weighted-Umeyama Sim3 solved on the overlap, so hundreds of frames fuse into a
   single coherent cloud at bounded per-pass memory. conf_pct in [0,100];
   global_budget caps total emitted points (<=0 = unlimited, subsampled per window).
   out_counts (if non-NULL) is int[n_images] receiving per-INPUT-frame point counts
   (frame-major, for progressive build-up). On success sets *out_n and mallocs
   *out_xyz[3N] (world xyz, OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free
   xyz/radius via da_capi_free_floats and rgb via da_capi_free_bytes. 0 ok, -1 err.

   De-ghosting toggles (0=off, non-zero=on):
     icp_refine  : point-to-plane ICP refinement of each window seam;
     loop_close  : loop-closure detection + Sim3 pose-graph drift removal;
     fuse        : final voxel/surface fusion (collapses doubled sheets, de-densifies)
                   with cell fuse_voxel_m metres (<=0 => 0.03 m default). After fusion
                   the point order is spatial, so out_counts is rescaled to sum to N.
     metric      : rescale each window to absolute METRES via the nested metric branch
                   (per-window robust-median scale_factor applied to depth + pose).
                   REQUIRES a da_capi_load_nested ctx; returns -1 otherwise. Without it
                   the cloud is at an arbitrary relative scale.

   Per-frame camera poses (for the viewer flythrough) are stashed on the ctx and
   retrieved separately via da_capi_stream_last_poses — this function is already at the
   FFI argument-count ceiling, so they can't be added as out-params here. */
int da_capi_points_stream(da_ctx* ctx, const char** image_paths, int n_images,
                          int chunk_size, int overlap, double conf_pct,
                          float point_size, int global_budget,
                          int icp_refine, int loop_close, int fuse, int metric,
                          double fuse_voxel_m,
                          int* out_n, int* out_counts,
                          float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Per-input-frame camera poses from the most recent da_capi_points_stream call.
   Mallocs *out_pos[3F] (camera centre) and *out_fwd[3F] (unit view direction), OpenCV
   world axes; sets *out_nframes = F (== that call's n_images). Free via
   da_capi_free_floats. Returns 0 ok, -1 if none (no stream run / all windows failed). */
int da_capi_stream_last_poses(da_ctx* ctx, float** out_pos, float** out_fwd, int* out_nframes);

/* Retrieve the DTVM v1 temporal voxel mesh generated by the most recent fused
   da_capi_points_stream call. Mallocs *out_data[*out_size]; free with
   da_capi_free_bytes. Returns 0 on success, -1 when generation was disabled or
   no valid fused surface was produced. */
int da_capi_stream_last_voxel_mesh(da_ctx* ctx, unsigned char** out_data, size_t* out_size);

/* Single-image 3D GAUSSIANS (DA3-GIANT / GS-head models only; returns -1 with a
   clear last_error otherwise). Returns world-frame (OpenCV) gaussians as parallel
   arrays: *out_xyz[3N] means, *out_scale[3N], *out_quat[4N] (wxyz), *out_rgb[3N]
   linear colour in [0,1] (input-photo colour; see note), *out_opacity[N]. Sets
   *out_n. Also, when non-NULL, writes the input camera intrinsics to out_intr[9]
   (K 3x3 row-major, pixel units) and the processed resolution to *out_w,*out_h --
   the pose is canonical (identity) so a viewer placed at the origin looking down
   +z with this K reproduces the input view. Free every returned float buffer via
   da_capi_free_floats. Returns 0 ok, -1 error. */
int da_capi_gaussians(da_ctx* ctx, const char* image_path, int* out_n,
                      float** out_xyz, float** out_scale, float** out_quat,
                      float** out_rgb, float** out_opacity,
                      float* out_intr, int* out_w, int* out_h);
#ifdef __cplusplus
}
#endif
#endif
