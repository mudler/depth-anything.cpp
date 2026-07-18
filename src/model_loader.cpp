#include "model_loader.hpp"
#include "da_gguf_keys.h"
#include "common.hpp"
#include "compute_mode.hpp"
#include "backend.hpp"

#include "ggml-backend.h"

#include <utility>

namespace da {

// --- Global GPU compute-mode flag (see compute_mode.hpp) -------------------
static bool g_gpu_mode = false;
void set_gpu_mode(bool on){ g_gpu_mode = on; }
bool gpu_mode(){ return g_gpu_mode; }

// Weights read directly on the HOST via ->data during graph build feed
// host-computed graph INPUTS (not graph nodes), so they MUST stay in
// host-accessible memory and are never mirrored to the device:
//   - "vit.pos_embed"   : host bicubic interp in DinoBackbone::interp_pos_embed
//   - "vit.camera_token": host camera-token inject in DinoBackbone::forward*
// (The metric branch aliases m_vit.* -> vit.* so the same names apply.)
static bool is_host_only_tensor(const std::string& name) {
    return name == "vit.pos_embed" || name == "vit.camera_token";
}

// Weights that are DUAL-USE: real ggml graph operands in the fused path
// (DinoBackbone::build_feats_graph -> layernorm(x, vit.norm.{w,b})), AND read
// directly on the host in the unfused / multi-view path (layernorm_host). These
// must be mirrored to the device so the graph has a valid backend buffer (else
// the Metal encoder derefs a NULL buffer and segfaults), while their host
// originals are preserved for the host readers via ModelLoader::host_tensor().
static bool is_host_also_tensor(const std::string& name) {
    return name == "vit.norm.weight" || name == "vit.norm.bias";
}

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_u32(g,id);
}
static int32_t kv_i32(gguf_context* g, const char* k, int32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::vector<float> kv_f32_arr(gguf_context* g, const char* k){
    std::vector<float> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_FLOAT32){
        size_t n = gguf_get_arr_n(g,id);
        const float* a = (const float*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}

ModelLoader::~ModelLoader(){
    // Free the device buffer + its metadata ctx BEFORE the host ctx_ (which holds
    // the host-read tensors and the source bytes the device weights were copied from).
    if (gpu_buf_)    ggml_backend_buffer_free(gpu_buf_);
    if (device_ctx_) ggml_free(device_ctx_);
    if (gguf_) gguf_free(gguf_);
    if (ctx_)  ggml_free(ctx_);
}

bool ModelLoader::load(const std::string& path){
    // Guard against re-entry: a second load() would otherwise leak the prior
    // gguf/ctx and accumulate stale tensor-map entries.
    if (gguf_) { gguf_free(gguf_); gguf_ = nullptr; }
    if (ctx_)  { ggml_free(ctx_);  ctx_  = nullptr; }
    tensors_.clear();
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_){ DA_LOG("gguf_init_from_file failed: %s", path.c_str()); return false; }
    // Nested metric branch: config + tensors live under m_vit.*/m_head.* prefixes.
    // Detect it and read the m_* KV; tensors are aliased to vit.*/head.* below so the
    // existing backbone/DPT graph code (which hard-codes vit./head.) works unchanged.
    const bool metric = gguf_find_key(gguf_, DA_KV_M_VIT_EMBED_DIM) >= 0;
    cfg_.patch_size      = kv_u32(gguf_, DA_KV_PATCH_SIZE, 14);
    if (metric){
        cfg_.embed_dim       = kv_u32(gguf_, DA_KV_M_VIT_EMBED_DIM);
        cfg_.depth           = kv_u32(gguf_, DA_KV_M_VIT_DEPTH);
        cfg_.num_heads       = kv_u32(gguf_, DA_KV_M_VIT_NUM_HEADS);
        cfg_.head_dim        = kv_u32(gguf_, DA_KV_M_VIT_HEAD_DIM);
        cfg_.mlp_hidden      = kv_u32(gguf_, DA_KV_M_VIT_MLP_HIDDEN);
        cfg_.num_register    = kv_u32(gguf_, DA_KV_M_VIT_NUM_REGISTER);
        cfg_.pos_embed_grid  = kv_u32(gguf_, DA_KV_M_VIT_POS_EMBED_GRID);
        cfg_.alt_start       = kv_i32(gguf_, DA_KV_M_VIT_ALT_START, -1);
        cfg_.rope_start      = kv_i32(gguf_, DA_KV_M_VIT_ROPE_START, -1);
        cfg_.qknorm_start    = kv_i32(gguf_, DA_KV_M_VIT_QKNORM_START, -1);
        cfg_.init_values     = kv_f32(gguf_, DA_KV_M_VIT_INIT_VALUES, 0.f);
        cfg_.rope_freq       = kv_f32(gguf_, DA_KV_M_VIT_ROPE_FREQ, 100.f);
        cfg_.ln_eps          = kv_f32(gguf_, DA_KV_M_VIT_LN_EPS, 1e-6f);
        cfg_.interp_offset   = kv_f32(gguf_, DA_KV_M_VIT_INTERP_OFFSET, 0.1f);
        cfg_.cat_token       = kv_bool(gguf_, DA_KV_M_VIT_CAT_TOKEN, true);
        cfg_.qkv_bias        = kv_bool(gguf_, DA_KV_M_VIT_QKV_BIAS, true);
        cfg_.interp_antialias= kv_bool(gguf_, DA_KV_M_VIT_INTERP_ANTIALIAS, false);
        cfg_.ffn_type        = kv_str(gguf_, DA_KV_M_VIT_FFN_TYPE, "mlp");
        cfg_.head_features   = kv_u32(gguf_, DA_KV_M_HEAD_FEATURES, 0);
        cfg_.out_layers      = kv_i32_arr(gguf_, DA_KV_M_VIT_OUT_LAYERS);
        cfg_.head_out_channels = kv_i32_arr(gguf_, DA_KV_M_HEAD_OUT_CHANNELS);
        // Metric DPT pos_embed defaults to False (DPT.__init__ default); the nested
        // converter omits the KV, so it is absent here -> no UV pos-embed in the head.
        cfg_.head_pos_embed  = kv_bool(gguf_, "depthanything3.m_head.pos_embed", false);
    } else {
        cfg_.embed_dim       = kv_u32(gguf_, DA_KV_VIT_EMBED_DIM);
        cfg_.depth           = kv_u32(gguf_, DA_KV_VIT_DEPTH);
        cfg_.num_heads       = kv_u32(gguf_, DA_KV_VIT_NUM_HEADS);
        cfg_.head_dim        = kv_u32(gguf_, DA_KV_VIT_HEAD_DIM);
        cfg_.mlp_hidden      = kv_u32(gguf_, DA_KV_VIT_MLP_HIDDEN);
        cfg_.num_register    = kv_u32(gguf_, DA_KV_VIT_NUM_REGISTER);
        cfg_.pos_embed_grid  = kv_u32(gguf_, DA_KV_VIT_POS_EMBED_GRID);
        cfg_.alt_start       = kv_i32(gguf_, DA_KV_VIT_ALT_START, -1);
        cfg_.rope_start      = kv_i32(gguf_, DA_KV_VIT_ROPE_START, -1);
        cfg_.qknorm_start    = kv_i32(gguf_, DA_KV_VIT_QKNORM_START, -1);
        cfg_.init_values     = kv_f32(gguf_, DA_KV_VIT_INIT_VALUES, 0.f);
        cfg_.rope_freq       = kv_f32(gguf_, DA_KV_VIT_ROPE_FREQ, 100.f);
        cfg_.ln_eps          = kv_f32(gguf_, DA_KV_VIT_LN_EPS, 1e-6f);
        cfg_.interp_offset   = kv_f32(gguf_, DA_KV_VIT_INTERP_OFFSET, 0.1f);
        cfg_.cat_token       = kv_bool(gguf_, DA_KV_VIT_CAT_TOKEN, true);
        cfg_.qkv_bias        = kv_bool(gguf_, DA_KV_VIT_QKV_BIAS, true);
        cfg_.interp_antialias= kv_bool(gguf_, DA_KV_VIT_INTERP_ANTIALIAS, false);
        cfg_.ffn_type        = kv_str(gguf_, DA_KV_VIT_FFN_TYPE, "mlp");
        cfg_.head_features   = kv_u32(gguf_, DA_KV_HEAD_FEATURES, 0);
        cfg_.out_layers      = kv_i32_arr(gguf_, DA_KV_VIT_OUT_LAYERS);
        cfg_.head_out_channels = kv_i32_arr(gguf_, DA_KV_HEAD_OUT_CHANNELS);
        cfg_.head_pos_embed  = kv_bool(gguf_, DA_KV_HEAD_POS_EMBED, true);
    }
    cfg_.img_mean        = kv_f32_arr(gguf_, DA_KV_IMG_MEAN);
    cfg_.img_std         = kv_f32_arr(gguf_, DA_KV_IMG_STD);
    cfg_.img_resize_target = kv_u32(gguf_, DA_KV_IMG_RESIZE_TARGET, 504);
    cfg_.img_resize_mode = kv_str(gguf_, DA_KV_IMG_RESIZE_MODE, "upper_bound");
    cfg_.checkpoint_name = kv_str(gguf_, DA_KV_CHECKPOINT_NAME);
    cfg_.head_max_depth  = kv_f32(gguf_, DA_KV_HEAD_MAX_DEPTH, 0.f);
    cfg_.arch            = kv_str(gguf_, DA_KV_ARCH, "depthanything3");

    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (!t) continue;
        tensors_[nm] = t;
        // Alias metric-branch tensors so vit./head. lookups resolve to m_vit./m_head.
        std::string s(nm);
        if (s.rfind("m_vit.", 0) == 0)  tensors_["vit."  + s.substr(6)] = t;
        else if (s.rfind("m_head.", 0) == 0) tensors_["head." + s.substr(7)] = t;
    }
    // All four structural dims are converter-written and required by the graph
    // (head_dim/num_heads feed attention reshapes). Gate on all of them so a
    // malformed external GGUF fails here rather than dividing by zero later.
    return cfg_.embed_dim>0 && cfg_.depth>0 && cfg_.num_heads>0 && cfg_.head_dim>0;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it==tensors_.end() ? nullptr : it->second;
}

bool ModelLoader::offload_weights(Backend& be){
    // CPU backend: no-op. Graphs keep referencing the gguf host tensors in ctx_
    // directly (zero-copy; the CPU path is byte-identical to before this change).
    if (!be.is_offloading()) return true;
    if (device_ctx_) return true;                       // idempotent
    ggml_backend_t backend = be.handle();
    if (!backend || !ctx_){ DA_LOG("offload_weights: null backend/ctx"); return false; }

    // Mirror every gguf tensor EXCEPT the host-read tensors into a no_alloc
    // device context, allocate it on the backend, upload the bytes, and repoint
    // tensors_[name] at the device tensor. The host-read tensors stay pointing at
    // the original ctx_ (host) tensors so DinoBackbone's host reads keep working.
    // ctx_ remains alive as the host source of both bytes and host-read tensors.
    //
    // The tensors_ map also holds metric-branch aliases (m_vit.* AND vit.* point
    // at the SAME ggml_tensor*). De-dup by pointer so each source tensor is only
    // mirrored/uploaded once, then repoint every alias at the shared device copy.
    const size_t n = tensors_.size();
    ggml_init_params dp{};
    dp.mem_size  = ggml_tensor_overhead() * (n + 8);
    dp.no_alloc  = true;
    device_ctx_ = ggml_init(dp);
    if (!device_ctx_){ DA_LOG("offload_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<ggml_tensor*, ggml_tensor*> src2dev; src2dev.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> newmap; newmap.reserve(n);
    size_t n_dev = 0;
    for (auto& kv : tensors_) {
        if (is_host_only_tensor(kv.first)) {
            newmap.emplace(kv.first, kv.second);        // keep host tensor as-is
            continue;
        }
        // Dual-use weights: preserve the host original for host_tensor() reads,
        // then fall through to mirror them onto the device for the graph.
        if (is_host_also_tensor(kv.first))
            host_tensors_.emplace(kv.first, kv.second);
        ggml_tensor* s = kv.second;
        auto it = src2dev.find(s);
        if (it != src2dev.end()) {                      // alias of an already-mirrored tensor
            newmap.emplace(kv.first, it->second);
            continue;
        }
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, s->name);
        src2dev.emplace(s, d);
        newmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);                   // host source bytes in ctx_
        ++n_dev;
    }
    gpu_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if (!gpu_buf_){ DA_LOG("offload_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(newmap);   // graphs now reference the device-resident weights
    DA_LOG("offload_weights: %zu weights -> %s (%zu host-only tensors kept on CPU, "
           "%zu dual-use weights mirrored + host-preserved)",
           n_dev, be.device_name().c_str(), (size_t)2, host_tensors_.size());
    return true;
}

ggml_tensor* ModelLoader::host_tensor(const std::string& name) const {
    auto it = host_tensors_.find(name);
    if (it != host_tensors_.end()) return it->second;  // preserved host original
    return tensor(name);                               // CPU path: never offloaded
}
} // namespace da
