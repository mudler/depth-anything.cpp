#include "da_capi.h"
#include "engine.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "glb_export.hpp"
#include "colmap_export.hpp"
#include "reconstruct.hpp"
#include "stream.hpp"
#include "fuse.hpp"
#include "tsdf.hpp"
#include "voxel_mesh.hpp"
#include "common.hpp"
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <utility>
#include <vector>

struct da_ctx {
    std::unique_ptr<da::Engine> engine;
    std::string last_error;
    // Per-frame capture poses from the most recent da_capi_points_stream call (OpenCV
    // axes, 3F each). da_capi_points_stream is at the purego arg-count ceiling, so these
    // are retrieved via da_capi_stream_last_poses instead of as out-params.
    std::vector<float> frame_pos, frame_fwd;
    // Scene-relative TSDF fusion knobs for the next da_capi_points_stream (set via
    // da_capi_set_fuse_params, which dodges the stream's arg-count ceiling). 0 => the
    // fuse_tsdf defaults (voxel = 0.4% of bbox diag, truncation = 4 voxels).
    double fuse_voxel_frac = 0.0, fuse_trunc_mult = 0.0;
    // Optional temporal exposed-face mesh from the most recent streamed fuse.
    // Configuration is separate because points_stream is at the PureGo arg ceiling.
    bool temporal_voxel_mesh_enabled = false;
    std::vector<uint8_t> temporal_voxel_mesh;
};

static char* dup_cstr(const std::string& s){
    char* p = (char*)std::malloc(s.size()+1);
    if (p) std::memcpy(p, s.c_str(), s.size()+1);
    return p;
}
// Minimal JSON string escaping for interpolated values (quotes, backslash, controls).
static std::string json_escape(const std::string& s){
    std::string o; o.reserve(s.size()+2);
    for (char ch : s){
        switch (ch){
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)ch < 0x20){ char b[8]; std::snprintf(b,sizeof(b),"\\u%04x",ch); o += b; }
                else o += ch;
        }
    }
    return o;
}
// Best-effort metric detection from the checkpoint name: metric/nested/mono
// variants produce metric-scale depth; relative DualDPT (base/giant/large/small)
// do not. Unknown -> 0.
static bool capi_is_metric(const da::Config& cfg){
    // DA2 metric models carry a positive head_max_depth (20/80); authoritative.
    if (cfg.head_max_depth > 0.f) return true;
    std::string n = cfg.checkpoint_name;
    for (char& ch : n) ch = (char)std::tolower((unsigned char)ch);
    return n.find("metric") != std::string::npos ||
           n.find("nested") != std::string::npos ||
           n.find("mono")   != std::string::npos;
}

// Run the nested metric pipeline (anyview GIANT + metric ViT-L branches ->
// alignment) for a single image. Fills depth + scaled ext/intr + processed dims.
// Returns false with c->last_error set on failure.
static bool capi_run_nested(da_ctx* c, const char* image_path,
                            std::vector<float>& depth,
                            std::array<float,12>& ext, std::array<float,9>& intr,
                            int& H, int& W){
    da::NestedOut out;
    if (!c->engine->depth_metric_path(image_path, out, H, W)){
        c->last_error = "nested: depth_metric failed"; return false; }
    depth = std::move(out.depth);
    ext = out.extrinsics;
    intr = out.intrinsics;
    return true;
}

extern "C" {
int da_capi_abi_version(void){ return 11; }

// Scene-relative TSDF fusion knobs applied by the NEXT da_capi_points_stream when
// its fuse flag is set. voxel_frac = voxel edge as a fraction of the bbox diagonal
// ("fusion detail"); trunc_mult = truncation as a multiple of the voxel ("merge
// range", max merged gap = 2*trunc_mult voxels). Either <=0 keeps the built-in
// default. Split out from da_capi_points_stream, which is at the purego arg ceiling.
void da_capi_set_fuse_params(da_ctx* c, double voxel_frac, double trunc_mult){
    if (!c) return;
    c->fuse_voxel_frac = voxel_frac;
    c->fuse_trunc_mult = trunc_mult;
}
void da_capi_set_temporal_voxel_mesh(da_ctx* c, int enabled){
    if (!c) return;
    c->temporal_voxel_mesh_enabled = (enabled != 0);
}
da_ctx* da_capi_load(const char* path, int n_threads){
    if (!path) return nullptr;
    auto e = da::Engine::load(path, n_threads);
    if (!e) return nullptr;
    auto* c = new da_ctx(); c->engine = std::move(e); return c;
}
da_ctx* da_capi_load_nested(const char* anyview, const char* metric, int n_threads){
    if (!anyview || !metric) return nullptr;
    auto e = da::Engine::load_nested(anyview, metric, n_threads);
    if (!e) return nullptr;
    auto* c = new da_ctx(); c->engine = std::move(e); return c;
}
void da_capi_free(da_ctx* c){ delete c; }
char* da_capi_info_json(da_ctx* c){
    if (!c || !c->engine) return nullptr;
    const auto& cfg = c->engine->config();
    std::string j = "{\"checkpoint\":\"" + json_escape(cfg.checkpoint_name) + "\",\"embed_dim\":" +
        std::to_string(cfg.embed_dim) + ",\"depth\":" + std::to_string(cfg.depth) +
        ",\"num_heads\":" + std::to_string(cfg.num_heads) + "}";
    return dup_cstr(j);
}
void da_capi_free_string(char* s){ std::free(s); }
const char* da_capi_last_error(da_ctx* c){ return c ? c->last_error.c_str() : ""; }
float* da_capi_depth_path(da_ctx* c, const char* image_path, int* out_h, int* out_w){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "depth: bad args"; return nullptr; }
    std::vector<float> depth, conf; int H = 0, W = 0;
    if (c->engine->is_nested()){
        std::array<float,12> ext; std::array<float,9> intr;
        if (!capi_run_nested(c, image_path, depth, ext, intr, H, W)) return nullptr;
    } else if (c->engine->is_da2()){
        if (!c->engine->depth_relative_path(image_path, depth, H, W)){ c->last_error = "depth: da2 failed"; return nullptr; }
    } else if (!c->engine->depth(image_path, depth, conf, H, W)){ c->last_error = "depth: failed"; return nullptr; }
    float* p = (float*)std::malloc(depth.size() * sizeof(float));
    if (!p){ c->last_error = "depth: oom"; return nullptr; }
    std::memcpy(p, depth.data(), depth.size() * sizeof(float));
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    return p;
}
void da_capi_free_floats(float* p){ std::free(p); }
int da_capi_pose_path(da_ctx* c, const char* image_path, float out_ext[12], float out_intr[9]){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "pose: bad args"; return -1; }
    if (c->engine->is_da2()){ c->last_error = "pose: da2 model has no camera pose"; return -1; }
    std::vector<float> depth, conf; std::array<float,12> ext; std::array<float,9> intr; int H = 0, W = 0;
    if (c->engine->is_nested()){
        if (!capi_run_nested(c, image_path, depth, ext, intr, H, W)) return -1;
    } else if (!c->engine->depth_pose_path(image_path, depth, conf, ext, intr, H, W)){ c->last_error = "pose: failed"; return -1; }
    if (out_ext)  std::memcpy(out_ext,  ext.data(),  12 * sizeof(float));
    if (out_intr) std::memcpy(out_intr, intr.data(),  9 * sizeof(float));
    return 0;
}
// Shared single-image export prep: run native depth+pose, capture processed RGB,
// build N=1 exporter inputs. Returns false (with c->last_error set) on failure.
static bool capi_export_prep(da_ctx* c, const char* image_path,
                             std::vector<float>& depth, std::vector<float>& conf,
                             std::vector<std::array<float,9>>& K,
                             std::vector<std::array<float,16>>& E,
                             std::vector<uint8_t>& rgb_u8,
                             da::Image& img, int& H, int& W){
    if (!da::load_image_rgb(image_path, img)){ c->last_error = "export: load image failed"; return false; }
    std::array<float,12> ext; std::array<float,9> intr;
    if (!c->engine->depth_pose_native(img, depth, conf, ext, intr, H, W)){
        c->last_error = "export: depth+pose failed"; return false; }
    da::Preprocessed pp;
    if (!da::preprocess_real(img, c->engine->config(), pp, &rgb_u8) || pp.H != H || pp.W != W){
        c->last_error = "export: capture processed colors failed"; return false; }
    std::array<float,16> ext4{};
    for (int i = 0; i < 12; ++i) ext4[i] = ext[i];
    ext4[12] = 0.f; ext4[13] = 0.f; ext4[14] = 0.f; ext4[15] = 1.f;
    K = { intr }; E = { ext4 };
    return true;
}
int da_capi_export_glb(da_ctx* c, const char* image_path, const char* out_glb){
    if (!c || !c->engine || !image_path || !out_glb){ if (c) c->last_error = "export_glb: bad args"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    if (!da::write_glb(out_glb, depth, conf, K, E, imgs_u8, H, W, 1, da::GlbOptions{})){
        c->last_error = "export_glb: write failed"; return -1; }
    return 0;
}
int da_capi_export_colmap(da_ctx* c, const char* image_path, const char* out_dir, int binary){
    if (!c || !c->engine || !image_path || !out_dir){ if (c) c->last_error = "export_colmap: bad args"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    std::string path(image_path);
    size_t s = path.find_last_of("/\\");
    std::vector<std::string> names { s == std::string::npos ? path : path.substr(s + 1) };
    std::vector<std::pair<int,int>> orig_wh { { img.w, img.h } };
    if (!da::write_colmap(out_dir, depth, conf, K, E, imgs_u8, names, orig_wh, H, W, 1, binary != 0)){
        c->last_error = "export_colmap: write failed"; return -1; }
    return 0;
}
int da_capi_depth_dense(da_ctx* c, const char* image_path, int* out_h, int* out_w,
                        float** out_depth, float** out_conf, float** out_sky,
                        float out_ext[12], float out_intr[9], int* out_is_metric){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "depth_dense: bad args"; return -1; }
    // Default outputs to a clean state.
    if (out_depth) *out_depth = nullptr;
    if (out_conf)  *out_conf  = nullptr;
    if (out_sky)   *out_sky   = nullptr;
    if (out_ext)  std::memset(out_ext,  0, 12 * sizeof(float));
    if (out_intr) std::memset(out_intr, 0,  9 * sizeof(float));
    int H = 0, W = 0;
    // Nested metric model: run both branches + alignment -> metric-scale depth +
    // scaled pose. No conf/sky surface (sky is already folded into depth).
    if (c->engine->is_nested()){
        std::vector<float> ndepth; std::array<float,12> next; std::array<float,9> nintr;
        if (!capi_run_nested(c, image_path, ndepth, next, nintr, H, W)) return -1;
        const size_t hw = (size_t)H * W;
        if (hw == 0 || ndepth.size() != hw){ c->last_error = "depth_dense: nested empty/size mismatch"; return -1; }
        if (out_depth){
            float* dptr = (float*)std::malloc(hw * sizeof(float));
            if (!dptr){ c->last_error = "depth_dense: oom"; return -1; }
            std::memcpy(dptr, ndepth.data(), hw * sizeof(float));
            *out_depth = dptr;
        }
        if (out_ext)  std::memcpy(out_ext,  next.data(),  12 * sizeof(float));
        if (out_intr) std::memcpy(out_intr, nintr.data(),  9 * sizeof(float));
        if (out_h) *out_h = H;
        if (out_w) *out_w = W;
        if (out_is_metric) *out_is_metric = 1;
        return 0;
    }
    // Depth Anything V2: depth only. No conf/sky surface, no camera pose. ext/intr
    // stay zeroed (memset above); metric iff head_max_depth > 0 (sigmoid x max_depth).
    if (c->engine->is_da2()){
        std::vector<float> d2;
        if (!c->engine->depth_relative_path(image_path, d2, H, W)){
            c->last_error = "depth_dense: da2 failed"; return -1; }
        const size_t hw = (size_t)H * W;
        if (hw == 0 || d2.size() != hw){ c->last_error = "depth_dense: da2 empty/size mismatch"; return -1; }
        if (out_depth){
            float* dptr = (float*)std::malloc(hw * sizeof(float));
            if (!dptr){ c->last_error = "depth_dense: oom"; return -1; }
            std::memcpy(dptr, d2.data(), hw * sizeof(float));
            *out_depth = dptr;
        }
        if (out_h) *out_h = H;
        if (out_w) *out_w = W;
        if (out_is_metric) *out_is_metric = (c->engine->config().head_max_depth > 0.f) ? 1 : 0;
        return 0;
    }
    da::Image img;
    if (!da::load_image_rgb(image_path, img)){ c->last_error = "depth_dense: load image failed"; return -1; }
    const bool mono = c->engine->is_mono();
    std::vector<float> depth, second; // second = conf (DualDPT) or sky (mono)
    if (mono){
        if (!c->engine->depth_mono(img, depth, second, H, W)){
            c->last_error = "depth_dense: depth_mono failed"; return -1; }
    } else {
        std::array<float,12> ext; std::array<float,9> intr;
        if (!c->engine->depth_pose_native(img, depth, second, ext, intr, H, W)){
            c->last_error = "depth_dense: depth+pose failed"; return -1; }
        if (out_ext)  std::memcpy(out_ext,  ext.data(),  12 * sizeof(float));
        if (out_intr) std::memcpy(out_intr, intr.data(),  9 * sizeof(float));
    }
    const size_t hw = (size_t)H * W;
    if (hw == 0 || depth.size() != hw){ c->last_error = "depth_dense: empty/size mismatch"; return -1; }
    float* dptr = (float*)std::malloc(hw * sizeof(float));
    float* sptr = (float*)std::malloc(hw * sizeof(float));
    if (!dptr || !sptr){ std::free(dptr); std::free(sptr); c->last_error = "depth_dense: oom"; return -1; }
    std::memcpy(dptr, depth.data(), hw * sizeof(float));
    std::memcpy(sptr, second.data(), std::min(second.size(), hw) * sizeof(float));
    if (out_depth) *out_depth = dptr; else std::free(dptr);
    if (mono){
        if (out_sky)  *out_sky  = sptr; else std::free(sptr);
    } else {
        if (out_conf) *out_conf = sptr; else std::free(sptr);
    }
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    if (out_is_metric) *out_is_metric = capi_is_metric(c->engine->config()) ? 1 : 0;
    return 0;
}
int da_capi_points(da_ctx* c, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "points: bad args"; return -1; }
    if (out_xyz) *out_xyz = nullptr;
    if (out_rgb) *out_rgb = nullptr;
    if (c->engine->is_mono() || c->engine->is_da2()){ c->last_error = "points: this model has no camera pose; use a DualDPT model"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    // back_project expects world-to-camera extrinsics; capi_export_prep yields the
    // same 4x4 ext used by glb/colmap export (mirrors examples/cli cmd_depth_export).
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    da::WorldPoints wp = da::back_project(depth, conf, K, E, imgs_u8, H, W, 1, conf_thresh);
    const size_t n = wp.xyz.size() / 3;
    float* xyz = (float*)std::malloc(wp.xyz.size() * sizeof(float));
    unsigned char* rgb = (unsigned char*)std::malloc(wp.rgb.size() * sizeof(unsigned char));
    if ((wp.xyz.size() && !xyz) || (wp.rgb.size() && !rgb)){
        std::free(xyz); std::free(rgb); c->last_error = "points: oom"; return -1; }
    if (wp.xyz.size()) std::memcpy(xyz, wp.xyz.data(), wp.xyz.size() * sizeof(float));
    if (wp.rgb.size()) std::memcpy(rgb, wp.rgb.data(), wp.rgb.size() * sizeof(unsigned char));
    if (out_xyz) *out_xyz = xyz; else std::free(xyz);
    if (out_rgb) *out_rgb = rgb; else std::free(rgb);
    if (out_n) *out_n = (int)n;
    return 0;
}
void da_capi_free_bytes(unsigned char* p){ std::free(p); }

int da_capi_points_multi(da_ctx* c, const char** paths, int n_images,
                         double conf_pct, float point_size,
                         int* out_n, int* out_counts,
                         float** out_xyz, unsigned char** out_rgb, float** out_radius){
    if (out_xyz) *out_xyz = nullptr;
    if (out_rgb) *out_rgb = nullptr;
    if (out_radius) *out_radius = nullptr;
    if (!c || !c->engine || !paths || n_images <= 0){ if (c) c->last_error = "points_multi: bad args"; return -1; }
    if (c->engine->is_mono() || c->engine->is_da2()){
        c->last_error = "points_multi: model has no camera pose; use a DualDPT DA3 model"; return -1; }
    if (!(point_size > 0.f)) point_size = 1.f;

    std::vector<da::Image> imgs(n_images);
    for (int i = 0; i < n_images; ++i){
        if (!paths[i] || !da::load_image_rgb(paths[i], imgs[i])){ c->last_error = "points_multi: load image failed"; return -1; }
    }
    std::vector<da::ViewResult> views; int H = 0, W = 0;
    if (!c->engine->depth_pose_multi(imgs, views, H, W)){ c->last_error = "points_multi: depth_pose_multi failed"; return -1; }
    const int N = (int)views.size();
    const size_t plane = (size_t)H * (size_t)W;

    std::vector<float> depth_all; depth_all.reserve((size_t)N * plane);
    std::vector<float> conf_all;  conf_all.reserve((size_t)N * plane);
    std::vector<std::array<float,9>>  K(N);
    std::vector<std::array<float,16>> E(N);
    std::vector<std::vector<uint8_t>> rgb_store(N);
    std::vector<const uint8_t*>       images_u8(N);
    bool have_conf = true;
    for (int i = 0; i < N; ++i){
        depth_all.insert(depth_all.end(), views[i].depth.begin(), views[i].depth.end());
        if (views[i].conf.size() == plane) conf_all.insert(conf_all.end(), views[i].conf.begin(), views[i].conf.end());
        else have_conf = false;
        K[i] = views[i].intr;
        std::array<float,16> e4{}; for (int k = 0; k < 12; ++k) e4[k] = views[i].ext[k];
        e4[12] = 0.f; e4[13] = 0.f; e4[14] = 0.f; e4[15] = 1.f; E[i] = e4;
        // Capture the processed RGB (same resize policy depth_pose_multi used) for
        // per-point colour; preprocess_real fills rgb_store directly.
        da::Preprocessed pp;
        if (!da::preprocess_real(imgs[i], c->engine->config(), pp, &rgb_store[i]) || pp.H != H || pp.W != W){
            c->last_error = "points_multi: preprocess color mismatch"; return -1; }
        images_u8[i] = rgb_store[i].data();
    }
    if (!have_conf) conf_all.clear();
    float conf_thr = -1e30f;
    if (have_conf){
        if (conf_pct < 0) conf_pct = 0; if (conf_pct > 100) conf_pct = 100;
        conf_thr = (float)da::percentile_linear(conf_all, conf_pct);
    }

    da::WorldPoints wp = da::back_project(depth_all, conf_all, K, E, images_u8, H, W, N, conf_thr);
    const size_t np = wp.xyz.size() / 3;
    if (np == 0){ c->last_error = "points_multi: no points survived (raise conf_pct or check parallax)"; return -1; }

    float* xyz = (float*)std::malloc(np * 3 * sizeof(float));
    unsigned char* rgb = (unsigned char*)std::malloc(np * 3);
    float* rad = (float*)std::malloc(np * sizeof(float));
    if (!xyz || !rgb || !rad){ std::free(xyz); std::free(rgb); std::free(rad); c->last_error = "points_multi: oom"; return -1; }
    if (out_counts) for (int i = 0; i < n_images; ++i) out_counts[i] = 0;
    for (size_t j = 0; j < np; ++j){
        xyz[3*j+0] = wp.xyz[3*j+0]; xyz[3*j+1] = wp.xyz[3*j+1]; xyz[3*j+2] = wp.xyz[3*j+2];
        rgb[3*j+0] = wp.rgb[3*j+0]; rgb[3*j+1] = wp.rgb[3*j+1]; rgb[3*j+2] = wp.rgb[3*j+2];
        int f = wp.frame[j], u = wp.u[j], v = wp.v[j];
        float d = depth_all[(size_t)f * plane + (size_t)v * W + (size_t)u];
        float fx = K[f][0], fy = K[f][4];
        float rr = 0.5f * (d / fx + d / fy) * point_size;
        if (!(rr > 0.f) || !std::isfinite(rr)) rr = 1e-4f;
        rad[j] = rr;
        if (out_counts && f >= 0 && f < n_images) out_counts[f]++;
    }
    if (out_xyz) *out_xyz = xyz; else std::free(xyz);
    if (out_rgb) *out_rgb = rgb; else std::free(rgb);
    if (out_radius) *out_radius = rad; else std::free(rad);
    if (out_n) *out_n = (int)np;
    return 0;
}

int da_capi_points_stream(da_ctx* c, const char** image_paths, int n_images,
                          int chunk_size, int overlap, double conf_pct,
                          float point_size, int global_budget,
                          int icp_refine, int loop_close, int fuse, int metric,
                          double fuse_voxel_m,
                          int* out_n, int* out_counts,
                          float** out_xyz, unsigned char** out_rgb, float** out_radius){
    if (out_xyz) *out_xyz = nullptr;
    if (out_rgb) *out_rgb = nullptr;
    if (out_radius) *out_radius = nullptr;
    if (c){ c->frame_pos.clear(); c->frame_fwd.clear(); c->temporal_voxel_mesh.clear(); }
    if (!c || !c->engine || !image_paths || n_images <= 0){ if (c) c->last_error = "points_stream: bad args"; return -1; }
    if (c->engine->is_mono() || c->engine->is_da2()){
        c->last_error = "points_stream: model has no camera pose; use a DualDPT DA3 model"; return -1; }

    std::vector<std::string> paths(n_images);
    for (int i = 0; i < n_images; ++i){
        if (!image_paths[i]){ c->last_error = "points_stream: null path"; return -1; }
        paths[i] = image_paths[i];
    }
    da::StreamParams sp;
    if (chunk_size > 0) sp.chunk_size = chunk_size;
    sp.overlap = overlap;             // clamped inside stream_points
    sp.conf_pct = conf_pct;
    if (point_size > 0.f) sp.point_size = point_size;
    sp.global_budget = global_budget;
    sp.icp_refine = (icp_refine != 0);   // task B
    sp.loop_close = (loop_close != 0);   // task C
    sp.metric     = (metric != 0);       // absolute-metres rescale (nested engine only)
    if (sp.metric && !c->engine->is_nested()){
        c->last_error = "points_stream: metric requires a nested (anyview+metric) model; load via da_capi_load_nested";
        return -1; }

    da::StreamCloud sc;
    if (!da::stream_points(*c->engine, paths, c->engine->config(), sp, sc, c->last_error)) return -1;

    // Task A: optional final surface fusion. A normal-space TSDF (src/tsdf.cpp)
    // collapses doubled/misaligned sheets into a single zero-crossing surface —
    // unlike a plain voxel downsample, which keeps both sheets when their gap
    // exceeds a cell. fuse_voxel_m sets the voxel edge (0 => scene-relative default);
    // the truncation band (max merge gap = 2*trunc) defaults to 4 voxels inside.
    if (fuse != 0){
        da::TsdfParams tp;
        tp.voxel = (float)fuse_voxel_m;   // absolute override (harness); <=0 => use frac
        tp.voxel_frac = (float)c->fuse_voxel_frac;   // scene-relative "detail" knob
        tp.trunc_mult = (float)c->fuse_trunc_mult;   // scene-relative "merge range" knob
        const int F = (int)sc.counts.size();
        const size_t pre_pts = sc.radius.size();
        // Per-input-point capture frame. The emitted cloud is frame-major, so the
        // per-frame counts define the frame of each point index. Handing this to
        // fuse_tsdf lets it tag each output voxel with its first-observing frame and
        // emit FRAME-MAJOR, so the progressive/flythrough reveal (which reveals a
        // frame prefix) still lines up after fusion.
        std::vector<int> in_frame; in_frame.reserve(pre_pts);
        for (int f = 0; f < F; ++f) for (int k = 0; k < sc.counts[f]; ++k) in_frame.push_back(f);
        if (in_frame.size() != pre_pts) in_frame.clear();   // only use if it lines up
        std::vector<int> out_frame;
        da::TsdfSurface temporal_surface;
        auto t_fuse0 = std::chrono::steady_clock::now();
        // frame_pos orients normals toward the observing camera (correct SDF sign);
        // weights default to 1/radius inside (near/high-confidence points dominate).
        int nf = da::fuse_tsdf(sc.xyz, sc.rgb, sc.radius, tp, nullptr, &sc.frame_pos,
                               in_frame.empty() ? nullptr : &in_frame, &out_frame,
                               c->temporal_voxel_mesh_enabled ? &temporal_surface : nullptr);
        DA_LOG("stream timing: tsdf(cpu)=%.2fs  %zu->%d pts",
               std::chrono::duration<double>(std::chrono::steady_clock::now()-t_fuse0).count(),
               pre_pts, nf);
        // Rebuild real per-frame counts from the fused points' first-observing frame
        // (the cloud is now frame-major), so the reveal shows each surface as its
        // capture frame is reached. Fall back to a proportional rescale if unavailable.
        if ((int)out_frame.size() == nf && F > 0){
            std::vector<int> cc(F, 0);
            for (int f : out_frame) if (f >= 0 && f < F) cc[f]++;
            sc.counts.swap(cc);
        } else {
            long pre = 0; for (int cc : sc.counts) pre += cc;
            if (pre > 0) for (int& cc : sc.counts) cc = (int)((long long)cc * nf / pre);
        }
        if (c->temporal_voxel_mesh_enabled && !temporal_surface.voxels.empty()) {
            c->temporal_voxel_mesh = da::encode_temporal_voxel_mesh(temporal_surface, (uint32_t)F);
            DA_LOG("temporal voxel mesh: %zu voxels, %.2f MiB",
                   temporal_surface.voxels.size(),
                   (double)c->temporal_voxel_mesh.size() / (1024.0 * 1024.0));
        }
    }

    const size_t np = sc.radius.size();
    if (np == 0){ c->last_error = "points_stream: no points"; return -1; }

    float* xyz = (float*)std::malloc(np * 3 * sizeof(float));
    unsigned char* rgb = (unsigned char*)std::malloc(np * 3);
    float* rad = (float*)std::malloc(np * sizeof(float));
    if (!xyz || !rgb || !rad){ std::free(xyz); std::free(rgb); std::free(rad); c->last_error = "points_stream: oom"; return -1; }
    std::memcpy(xyz, sc.xyz.data(), np * 3 * sizeof(float));
    std::memcpy(rgb, sc.rgb.data(), np * 3);
    std::memcpy(rad, sc.radius.data(), np * sizeof(float));
    if (out_counts) for (int i = 0; i < n_images; ++i) out_counts[i] = (i < (int)sc.counts.size()) ? sc.counts[i] : 0;
    if (out_xyz) *out_xyz = xyz; else std::free(xyz);
    if (out_rgb) *out_rgb = rgb; else std::free(rgb);
    if (out_radius) *out_radius = rad; else std::free(rad);
    if (out_n) *out_n = (int)np;

    // Stash per-frame camera poses for da_capi_stream_last_poses (flythrough). These are
    // unaffected by fusion (it only touches points).
    if ((int)sc.frame_pos.size() == 3 * n_images && (int)sc.frame_fwd.size() == 3 * n_images){
        c->frame_pos = std::move(sc.frame_pos);
        c->frame_fwd = std::move(sc.frame_fwd);
    }
    return 0;
}

// Retrieve the per-input-frame camera poses stashed by the most recent
// da_capi_points_stream call (kept separate because that function is at the FFI
// arg-count ceiling). Mallocs *out_pos[3F] (camera centre) and *out_fwd[3F] (unit view
// direction), OpenCV world axes; sets *out_nframes = F. Free via da_capi_free_floats.
// Returns 0 ok, -1 if no poses are available (e.g. no stream run, or all windows failed).
int da_capi_stream_last_poses(da_ctx* c, float** out_pos, float** out_fwd, int* out_nframes){
    if (out_pos) *out_pos = nullptr;
    if (out_fwd) *out_fwd = nullptr;
    if (out_nframes) *out_nframes = 0;
    if (!c || c->frame_pos.empty() || c->frame_pos.size() != c->frame_fwd.size()){
        if (c) c->last_error = "stream_last_poses: no poses available"; return -1; }
    const size_t sz = c->frame_pos.size();
    float* pos = (float*)std::malloc(sz * sizeof(float));
    float* fwd = (float*)std::malloc(sz * sizeof(float));
    if (!pos || !fwd){ std::free(pos); std::free(fwd); c->last_error = "stream_last_poses: oom"; return -1; }
    std::memcpy(pos, c->frame_pos.data(), sz * sizeof(float));
    std::memcpy(fwd, c->frame_fwd.data(), sz * sizeof(float));
    if (out_pos) *out_pos = pos; else std::free(pos);
    if (out_fwd) *out_fwd = fwd; else std::free(fwd);
    if (out_nframes) *out_nframes = (int)(sz / 3);
    return 0;
}

int da_capi_stream_last_voxel_mesh(da_ctx* c, unsigned char** out_data, size_t* out_size){
    if (out_data) *out_data = nullptr;
    if (out_size) *out_size = 0;
    if (!c || c->temporal_voxel_mesh.empty()){
        if (c) c->last_error = "stream_last_voxel_mesh: no temporal mesh available";
        return -1;
    }
    const size_t n = c->temporal_voxel_mesh.size();
    unsigned char* data = (unsigned char*)std::malloc(n);
    if (!data){ c->last_error = "stream_last_voxel_mesh: oom"; return -1; }
    std::memcpy(data, c->temporal_voxel_mesh.data(), n);
    if (out_data) *out_data = data; else std::free(data);
    if (out_size) *out_size = n;
    return 0;
}

int da_capi_gaussians(da_ctx* c, const char* path, int* out_n,
                      float** out_xyz, float** out_scale, float** out_quat,
                      float** out_rgb, float** out_opacity,
                      float* out_intr, int* out_w, int* out_h){
    if (out_xyz) *out_xyz = nullptr;
    if (out_scale) *out_scale = nullptr;
    if (out_quat) *out_quat = nullptr;
    if (out_rgb) *out_rgb = nullptr;
    if (out_opacity) *out_opacity = nullptr;
    if (!c || !c->engine || !path){ if (c) c->last_error = "gaussians: bad args"; return -1; }
    da::Image img;
    if (!da::load_image_rgb(path, img)){ c->last_error = "gaussians: load image failed"; return -1; }
    da::Gaussians g; int H = 0, W = 0;
    if (!c->engine->reconstruct(img, g, H, W)){
        c->last_error = "gaussians: reconstruct failed (needs a GS model, e.g. DA3-GIANT)"; return -1; }
    if (out_intr) for (int i = 0; i < 9; ++i) out_intr[i] = g.intr[i];
    if (out_w) *out_w = W;
    if (out_h) *out_h = H;
    const int N = g.N;
    if (N <= 0 || (int)g.means.size() < N*3 || (int)g.scales.size() < N*3 ||
        (int)g.rotations.size() < N*4 || (int)g.opacities.size() < N || (int)g.harmonics.size() < N*3*9){
        c->last_error = "gaussians: empty/short arrays"; return -1; }
    const double SH_C0 = 0.28209479177387814;
    float* xyz = (float*)std::malloc((size_t)N * 3 * sizeof(float));
    float* scl = (float*)std::malloc((size_t)N * 3 * sizeof(float));
    float* quat = (float*)std::malloc((size_t)N * 4 * sizeof(float));
    float* rgb = (float*)std::malloc((size_t)N * 3 * sizeof(float));
    float* op  = (float*)std::malloc((size_t)N * sizeof(float));
    if (!xyz || !scl || !quat || !rgb || !op){
        std::free(xyz); std::free(scl); std::free(quat); std::free(rgb); std::free(op);
        c->last_error = "gaussians: oom"; return -1; }
    // Prefer the input-photo colour baked by Engine::reconstruct; SH-DC is a
    // near-grey flat base (see gs_adapter.hpp). Fall back to SH-DC if absent.
    const bool have_col = (int)g.colors.size() >= N*3;
    for (int i = 0; i < N; ++i){
        for (int k = 0; k < 3; ++k){ xyz[3*i+k] = g.means[3*i+k]; scl[3*i+k] = g.scales[3*i+k]; }
        for (int k = 0; k < 4; ++k) quat[4*i+k] = g.rotations[4*i+k];
        for (int ch = 0; ch < 3; ++ch){
            double col = have_col ? (double)g.colors[3*i+ch]
                                  : 0.5 + SH_C0 * (double)g.harmonics[((size_t)i*3 + ch)*9 + 0];
            rgb[3*i+ch] = (float)(col < 0 ? 0 : (col > 1 ? 1 : col));
        }
        op[i] = g.opacities[i];
    }
    if (out_xyz) *out_xyz = xyz; else std::free(xyz);
    if (out_scale) *out_scale = scl; else std::free(scl);
    if (out_quat) *out_quat = quat; else std::free(quat);
    if (out_rgb) *out_rgb = rgb; else std::free(rgb);
    if (out_opacity) *out_opacity = op; else std::free(op);
    if (out_n) *out_n = N;
    return 0;
}
}
