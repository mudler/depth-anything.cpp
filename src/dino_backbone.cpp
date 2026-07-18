#include <cstdint>
#include <tuple>
#include <mutex>
#include <map>
#include "dino_backbone.hpp"
#include "ggml_extend.hpp"
#include "vit_block.hpp"
#include "common.hpp"
#include <cmath>
#include <algorithm>
namespace da {
static float cubic(float x){ // Catmull-Rom, a=-0.75 (PyTorch bicubic)
    const float a=-0.75f; x=std::fabs(x);
    if (x<1) return ((a+2)*x - (a+3))*x*x + 1;
    if (x<2) return (((x-5)*x+8)*x-4)*a;
    return 0;
}
std::vector<float> DinoBackbone::interp_pos_embed(int gh, int gw) const {
    const auto& c = ml_.config();
    ggml_tensor* pe = ml_.tensor("vit.pos_embed");   // ggml ne0=embed (768), ne1=rows (1370)
    const int embed = (int)c.embed_dim, M = (int)c.pos_embed_grid;
    const float* p = (const float*)pe->data;         // row r, channel ch at p[r*embed + ch]
    // Input-INDEPENDENT (depends only on the model's pos_embed + grid gh,gw), so it
    // is identical on every forward at a given resolution. Cache it (keyed by the
    // pos_embed pointer + grid) to avoid the scalar bicubic on every forward.
    static std::mutex pe_mu;
    static std::map<std::tuple<uintptr_t,int,int>, std::vector<float>> pe_cache;
    const auto pe_key = std::make_tuple((uintptr_t)p, gh, gw);
    {
        std::lock_guard<std::mutex> lk(pe_mu);
        auto it = pe_cache.find(pe_key);
        if (it != pe_cache.end()) return it->second;
    }
    auto src = [&](int r, int cc, int ch)->float{
        int row = 1 + r*M + cc; return p[(size_t)row*embed + ch];
    };
    std::vector<float> out((size_t)(1+gh*gw)*embed);
    for (int ch=0; ch<embed; ++ch) out[ch] = p[ch];   // cls pos-embed (row 0)
    const float sx = (float)(gw + c.interp_offset)/M, sy = (float)(gh + c.interp_offset)/M;
    for (int oy=0; oy<gh; ++oy){
        float iy = (oy+0.5f)/sy - 0.5f; int y0=(int)std::floor(iy); float fy=iy-y0;
        for (int ox=0; ox<gw; ++ox){
            float ix = (ox+0.5f)/sx - 0.5f; int x0=(int)std::floor(ix); float fx=ix-x0;
            int orow = 1 + oy*gw + ox;
            for (int ch=0; ch<embed; ++ch){
                float acc=0;
                for (int m=-1;m<=2;++m){ float wyv=cubic(fy-m); int yy=std::min(std::max(y0+m,0),M-1);
                    for (int n=-1;n<=2;++n){ float wxv=cubic(fx-n); int xx=std::min(std::max(x0+n,0),M-1);
                        acc += wyv*wxv*src(yy,xx,ch); }}
                out[(size_t)orow*embed + ch] = acc;
            }
        }
    }
    std::lock_guard<std::mutex> lk(pe_mu);
    return pe_cache.emplace(pe_key, std::move(out)).first->second;
}
bool DinoBackbone::prepare_tokens(const std::vector<float>& input_chw, int H, int W, std::vector<float>& out_tokens){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch, embed=(int)c.embed_dim;
    std::vector<float> pos = interp_pos_embed(gh, gw);
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ine[4] = { W, H, 3, 1 };
        ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, input_chw.data(), ine, 4);
        ggml_tensor* pw = ml_.tensor("vit.patch_embed.weight");   // conv weight
        ggml_tensor* pb = ml_.tensor("vit.patch_embed.bias");
        ggml_tensor* x = ggml_conv_2d(ctx, pw, img, patch, patch, 0,0,1,1); // -> [gw,gh,embed,1]
        x = ggml_reshape_2d(ctx, x, (int64_t)gw*gh, embed);                  // [N_patch, embed]
        x = ggml_cont(ctx, ggml_transpose(ctx, x));                         // [embed, N_patch]
        x = ggml_add(ctx, x, pb);                                           // bias broadcast over tokens
        ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
        x = ggml_concat(ctx, cls, x, 1);                                    // [embed, 1+N_patch]
        const int64_t pne[2] = { embed, 1 + (int64_t)gh*gw };
        ggml_tensor* pe = be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2);
        x = ggml_add(ctx, x, pe);
        return x;
    }, out_tokens);
}

bool DinoBackbone::forward(const std::vector<float>& input_chw, int H, int W,
                           std::vector<std::vector<float>>& feats,
                           std::vector<std::vector<float>>& cam_tokens){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch;
    const int embed=(int)c.embed_dim, heads=(int)c.num_heads, hd=(int)c.head_dim;
    const int Npatch=gh*gw, Ntok=1+Npatch;
    const float eps=c.ln_eps;

    // Two RoPE position sets over the 257 tokens. Token 0 (cls/cam) is a "special"
    // row at (0,0) for both. Patch token t (1..256): pos_local = (row+1, col+1);
    // pos_nodiff = (1,1) for every patch (zeros_like + 1).
    std::vector<float> pos_local(2*Ntok, 0.f), pos_nodiff(2*Ntok, 0.f);
    for (int t=1;t<Ntok;++t){ int idx=t-1; int row=idx/gw, col=idx%gw;
        pos_local[2*t+0]=(float)(row+1); pos_local[2*t+1]=(float)(col+1);
        pos_nodiff[2*t+0]=1.f;           pos_nodiff[2*t+1]=1.f; }
    RopeTables rt_local  = build_rope_tables(pos_local,  Ntok, hd, c.rope_freq);
    RopeTables rt_nodiff = build_rope_tables(pos_nodiff, Ntok, hd, c.rope_freq);

    std::vector<float> pos = interp_pos_embed(gh, gw);

    // Camera token (N=1): reference slot only -> camera_token row 0 (ne1 index 0).
    ggml_tensor* camt = ml_.tensor("vit.camera_token");          // ggml ne0=embed, ne1=2
    std::vector<float> cam0(embed, 0.f);
    if (camt){ const float* cp=(const float*)camt->data; for (int e=0;e<embed;++e) cam0[e]=cp[e]; }

    const std::vector<int32_t>& outL = c.out_layers;
    const size_t NL = outL.size();
    feats.assign(NL, {}); cam_tokens.assign(NL, {});

    // Per-out-layer raw captures: local_x (last LOCAL output) and x (post-block).
    std::vector<std::vector<float>> raw_local(NL), raw_x(NL);

    GraphInputPool pool;
    std::vector<float> throwaway;
    bool ok = be_.forward_capture([&](ggml_context* ctx) -> ggml_tensor* {
        // --- prepare tokens (same graph as prepare_tokens) ---
        const int64_t ine[4]={W,H,3,1};
        ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, input_chw.data(), ine, 4);
        ggml_tensor* x = ggml_conv_2d(ctx, ml_.tensor("vit.patch_embed.weight"), img, patch,patch,0,0,1,1);
        x = ggml_reshape_2d(ctx, x, (int64_t)Npatch, embed);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_add(ctx, x, ml_.tensor("vit.patch_embed.bias"));
        ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
        x = ggml_concat(ctx, cls, x, 1);
        const int64_t pne[2]={embed, Ntok};
        x = ggml_add(ctx, x, be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2));

        // --- rope inputs (both position sets) + camera token input ---
        // Register only inputs that some block actually consumes: an unconnected graph
        // input has no allocated buffer at upload time (asserts). Metric ViT-L has
        // rope_start=-1 and alt_start=-1, so neither RoPE nor the cam token is used.
        ggml_tensor *clb=nullptr,*slb=nullptr,*cnb=nullptr,*snb=nullptr;
        if (c.rope_start>=0){
            build_rope_inputs(ctx, be_, pool, rt_local,  clb, slb);
            build_rope_inputs(ctx, be_, pool, rt_nodiff, cnb, snb);
        }
        ggml_tensor* cam_in = nullptr;
        if (c.alt_start>=0){
            const int64_t camne[2]={embed,1};
            cam_in = be_.add_graph_input_nd(ctx, pool, cam0.data(), camne, 2);
        }

        ggml_tensor* local_x = x;          // last LOCAL-attention output
        for (int i=0;i<(int)c.depth;++i){
            // Cam-token overwrite BEFORE block i==alt_start: x[:,token0] = cam_token.
            if (c.alt_start>=0 && i==c.alt_start){
                ggml_tensor* rest = ggml_cont(ctx, ggml_view_2d(ctx, x, embed, Ntok-1,
                                                                x->nb[1], x->nb[1]));
                x = ggml_concat(ctx, cam_in, rest, 1);
            }
            const bool global   = (c.alt_start>=0 && i>=c.alt_start && (i%2==1));
            const bool use_rope  = (c.rope_start>=0 && i>=c.rope_start);
            ggml_tensor* cb = use_rope ? (global? cnb: clb) : nullptr;
            ggml_tensor* sb = use_rope ? (global? snb: slb) : nullptr;
            BlockWeights bw = load_block(ml_, i);
            x = vit_block(ctx, x, bw, heads, hd, eps, cb, sb);
            if (!global) local_x = x;       // track most recent LOCAL output only
            for (size_t o=0;o<NL;++o) if (outL[o]==i){
                be_.capture(local_x, &raw_local[o]);
                be_.capture(x,       &raw_x[o]);
            }
        }
        return x;                            // final readback discarded
    }, throwaway);
    if (!ok) return false;

    // --- host post-process matching get_intermediate_layers (width == 2*embed) ---
    // feat   = cat([local_x, vit_norm(x)]) over channels, patches 1..256  -> [256,1536]
    // cam    = cat([local_x[token0], x[token0]]) RAW (no norm)            -> [1536]
    ggml_tensor* nw = ml_.host_tensor("vit.norm.weight");
    ggml_tensor* nb = ml_.host_tensor("vit.norm.bias");
    const float* nwp=(const float*)nw->data;
    const float* nbp=(const float*)nb->data;
    auto layernorm_host=[&](const float* row)->std::vector<float>{
        double mean=0; for(int e=0;e<embed;++e) mean+=row[e]; mean/=embed;
        double var=0; for(int e=0;e<embed;++e){ double d=row[e]-mean; var+=d*d; } var/=embed;
        double inv=1.0/std::sqrt(var+(double)eps); std::vector<float> o(embed);
        for(int e=0;e<embed;++e) o[e]=(float)((row[e]-mean)*inv)*nwp[e]+nbp[e];
        return o;
    };
    for (size_t o=0;o<NL;++o){
        const auto& lx=raw_local[o]; const auto& xx=raw_x[o];  // both [embed, Ntok], ne0=embed fastest
        if (!c.cat_token){
            // cat_token FALSE (metric ViT-L): out_x = x; feat = full vit.norm(x),
            // token-0 stripped -> [Npatch, embed]. cam = x[token0] RAW (unused on the
            // metric branch, but kept consistent in shape [embed]).
            std::vector<float> camraw(embed);
            for(int e=0;e<embed;++e) camraw[e]=xx[e];
            cam_tokens[o]=std::move(camraw);
            std::vector<float> f((size_t)Npatch*embed);
            for(int t=1;t<Ntok;++t){
                std::vector<float> no=layernorm_host(&xx[(size_t)t*embed]);
                std::copy(no.begin(), no.end(), f.begin()+(size_t)(t-1)*embed);
            }
            feats[o]=std::move(f);
            continue;
        }
        // camera token: raw cat of token-0 halves (second half UN-normed).
        std::vector<float> camcat((size_t)2*embed);
        for(int e=0;e<embed;++e){ camcat[e]=lx[e]; camcat[embed+e]=xx[e]; }
        cam_tokens[o]=std::move(camcat);
        // features for patches 1..256, channel = cat([local_x_raw, norm(x)]).
        std::vector<float> f((size_t)Npatch*2*embed);
        for(int t=1;t<Ntok;++t){
            const float* lrow=&lx[(size_t)t*embed];
            const float* xrow=&xx[(size_t)t*embed];
            std::vector<float> no=layernorm_host(xrow);
            float* dst=&f[(size_t)(t-1)*2*embed];
            for(int e=0;e<embed;++e){ dst[e]=lrow[e]; dst[embed+e]=no[e]; }
        }
        feats[o]=std::move(f);
    }
    return true;
}

bool DinoBackbone::build_feats_graph(ggml_context* ctx, const std::vector<float>& input_chw,
                                     int H, int W, GraphInputPool& pool, ggml_tensor* out_feat[4]){
    const auto& c = ml_.config();
    if (!c.cat_token) return false;   // fused path is the cat_token=true (BASE/giant) case only
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch;
    const int embed=(int)c.embed_dim, heads=(int)c.num_heads, hd=(int)c.head_dim;
    const int Npatch=gh*gw, Ntok=1+Npatch;
    const float eps=c.ln_eps;

    // RoPE position sets (identical to forward()).
    std::vector<float> pos_local(2*Ntok, 0.f), pos_nodiff(2*Ntok, 0.f);
    for (int t=1;t<Ntok;++t){ int idx=t-1; int row=idx/gw, col=idx%gw;
        pos_local[2*t+0]=(float)(row+1); pos_local[2*t+1]=(float)(col+1);
        pos_nodiff[2*t+0]=1.f;           pos_nodiff[2*t+1]=1.f; }
    RopeTables rt_local  = build_rope_tables(pos_local,  Ntok, hd, c.rope_freq);
    RopeTables rt_nodiff = build_rope_tables(pos_nodiff, Ntok, hd, c.rope_freq);

    std::vector<float> pos = interp_pos_embed(gh, gw);

    ggml_tensor* camt = ml_.tensor("vit.camera_token");
    std::vector<float> cam0(embed, 0.f);
    if (camt){ const float* cp=(const float*)camt->data; for (int e=0;e<embed;++e) cam0[e]=cp[e]; }

    const std::vector<int32_t>& outL = c.out_layers;
    const size_t NL = outL.size();

    ggml_tensor* nw = ml_.tensor("vit.norm.weight");
    ggml_tensor* nb = ml_.tensor("vit.norm.bias");

    // --- prepare tokens (same graph as forward()) ---
    const int64_t ine[4]={W,H,3,1};
    ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, input_chw.data(), ine, 4);
    ggml_tensor* x = ggml_conv_2d(ctx, ml_.tensor("vit.patch_embed.weight"), img, patch,patch,0,0,1,1);
    x = ggml_reshape_2d(ctx, x, (int64_t)Npatch, embed);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_add(ctx, x, ml_.tensor("vit.patch_embed.bias"));
    ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
    x = ggml_concat(ctx, cls, x, 1);
    const int64_t pne[2]={embed, Ntok};
    x = ggml_add(ctx, x, be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2));

    ggml_tensor *clb=nullptr,*slb=nullptr,*cnb=nullptr,*snb=nullptr;
    if (c.rope_start>=0){
        build_rope_inputs(ctx, be_, pool, rt_local,  clb, slb);
        build_rope_inputs(ctx, be_, pool, rt_nodiff, cnb, snb);
    }
    ggml_tensor* cam_in = nullptr;
    if (c.alt_start>=0){
        const int64_t camne[2]={embed,1};
        cam_in = be_.add_graph_input_nd(ctx, pool, cam0.data(), camne, 2);
    }

    ggml_tensor* local_x = x;
    for (int i=0;i<(int)c.depth;++i){
        if (c.alt_start>=0 && i==c.alt_start){
            ggml_tensor* rest = ggml_cont(ctx, ggml_view_2d(ctx, x, embed, Ntok-1,
                                                            x->nb[1], x->nb[1]));
            x = ggml_concat(ctx, cam_in, rest, 1);
        }
        const bool global   = (c.alt_start>=0 && i>=c.alt_start && (i%2==1));
        const bool use_rope  = (c.rope_start>=0 && i>=c.rope_start);
        ggml_tensor* cb = use_rope ? (global? cnb: clb) : nullptr;
        ggml_tensor* sb = use_rope ? (global? snb: slb) : nullptr;
        BlockWeights bw = load_block(ml_, i);
        x = vit_block(ctx, x, bw, heads, hd, eps, cb, sb);
        if (!global) local_x = x;
        for (size_t o=0;o<NL && o<4;++o) if (outL[o]==i){
            // feat = cat([local_x_raw, layernorm(x)], dim0), then strip token-0.
            // Layout matches forward()'s host post-process exactly: channel 0..embed-1
            // = local_x RAW, embed..2*embed-1 = vit.norm(x); token-major channel-minor.
            ggml_tensor* normed = layernorm(ctx, x, nw, nb, eps);
            ggml_tensor* fcat   = ggml_concat(ctx, local_x, normed, 0);   // [2*embed, Ntok]
            ggml_tensor* fstr   = ggml_cont(ctx, ggml_view_2d(ctx, fcat, 2*(int64_t)embed,
                                                Npatch, fcat->nb[1], fcat->nb[1]));
            out_feat[o] = fstr;                                            // [2*embed, Npatch]
        }
    }
    return true;
}

// Pass-A: run the per-view LOCAL blocks [0,upto) (no cross-view mixing, since
// global attention starts at alt_start > upto here) and capture token-0 (cls) of
// each view's output. Used to drive reference-view selection at layer alt_start-1.
bool DinoBackbone::capture_local_cls(const std::vector<std::vector<float>>& views_chw, int H, int W,
                                     int upto, std::vector<std::vector<float>>& cls_out){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch;
    const int embed=(int)c.embed_dim, heads=(int)c.num_heads, hd=(int)c.head_dim;
    const int Npatch=gh*gw, Ntok=1+Npatch;
    const int S=(int)views_chw.size();
    const float eps=c.ln_eps;

    std::vector<float> pos_local(2*Ntok, 0.f);
    for (int t=1;t<Ntok;++t){ int idx=t-1; int row=idx/gw, col=idx%gw;
        pos_local[2*t+0]=(float)(row+1); pos_local[2*t+1]=(float)(col+1); }
    RopeTables rt_local = build_rope_tables(pos_local, Ntok, hd, c.rope_freq);
    std::vector<float> pos = interp_pos_embed(gh, gw);

    std::vector<std::vector<float>> cap_view(S);   // each [embed,Ntok]
    GraphInputPool pool; std::vector<float> throwaway;
    bool ok = be_.forward_capture([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t pne[2]={embed, Ntok};
        ggml_tensor* pe = be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2);
        // Only register RoPE inputs if a local layer in [0,upto) actually uses them
        // (otherwise the unconnected graph input has no buffer at upload time).
        ggml_tensor *clb=nullptr,*slb=nullptr;
        const bool any_rope = (c.rope_start>=0 && c.rope_start<upto);
        if (any_rope) build_rope_inputs(ctx, be_, pool, rt_local, clb, slb);
        ggml_tensor* last = nullptr;
        for (int s=0;s<S;++s){
            const int64_t ine[4]={W,H,3,1};
            ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, views_chw[s].data(), ine, 4);
            ggml_tensor* xv = ggml_conv_2d(ctx, ml_.tensor("vit.patch_embed.weight"), img, patch,patch,0,0,1,1);
            xv = ggml_reshape_2d(ctx, xv, (int64_t)Npatch, embed);
            xv = ggml_cont(ctx, ggml_transpose(ctx, xv));
            xv = ggml_add(ctx, xv, ml_.tensor("vit.patch_embed.bias"));
            ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
            xv = ggml_concat(ctx, cls, xv, 1);
            xv = ggml_add(ctx, xv, pe);
            for (int i=0;i<upto;++i){
                const bool use_rope = (c.rope_start>=0 && i>=c.rope_start);
                ggml_tensor* cb = use_rope? clb : nullptr;
                ggml_tensor* sb = use_rope? slb : nullptr;
                BlockWeights bw = load_block(ml_, i);
                xv = vit_block(ctx, xv, bw, heads, hd, eps, cb, sb);
            }
            be_.capture(xv, &cap_view[s]);
            last = xv;
        }
        return last;
    }, throwaway);
    if (!ok) return false;
    cls_out.assign(S, {});
    for (int s=0;s<S;++s){
        cls_out[s].assign(cap_view[s].begin(), cap_view[s].begin()+embed); // token-0 = first row
    }
    return true;
}

// saddle_balanced: select the view whose {avg cosine-sim to others, cls L2-norm,
// variance of normalized cls} are jointly closest to the per-metric median (0.5).
int DinoBackbone::select_reference_view_saddle(const std::vector<std::vector<float>>& cls, int embed) const {
    const int S=(int)cls.size();
    if (S<=1) return 0;
    std::vector<double> norm(S, 0.0);
    std::vector<std::vector<double>> cn(S, std::vector<double>(embed, 0.0)); // normalized cls
    for (int v=0;v<S;++v){
        double n=0; for (int e=0;e<embed;++e){ double d=cls[v][e]; n+=d*d; }
        n=std::sqrt(n); norm[v]=n;
        double inv = (n>0)? 1.0/n : 0.0;
        for (int e=0;e<embed;++e) cn[v][e]=cls[v][e]*inv;
    }
    // sim_score[v] = mean over w!=v of cos(cn[v],cn[w])  (diag removed; sum/(S-1))
    std::vector<double> sim_score(S,0.0), feat_norm(S,0.0), feat_var(S,0.0);
    for (int v=0;v<S;++v){
        double s=0;
        for (int w=0;w<S;++w){
            if (w==v) continue;
            double dot=0;
            for (int e=0;e<embed;++e) dot+=cn[v][e]*cn[w][e];
            s+=dot;
        }
        sim_score[v]=s/(double)(S-1);
        feat_norm[v]=norm[v];
        // variance of NORMALIZED cls over channels (torch .var default: unbiased, /(C-1))
        double mean=0; for (int e=0;e<embed;++e) mean+=cn[v][e]; mean/=embed;
        double var=0; for (int e=0;e<embed;++e){ double d=cn[v][e]-mean; var+=d*d; }
        feat_var[v]=var/(double)(embed-1);
    }
    auto norm01=[&](std::vector<double>& m){
        double mn=m[0], mx=m[0];
        for (double v: m){ mn=std::min(mn,v); mx=std::max(mx,v); }
        for (double& v: m) v=(v-mn)/(mx-mn+1e-8);
    };
    norm01(sim_score); norm01(feat_norm); norm01(feat_var);
    int best=0; double bestbal=1e300;
    for (int v=0;v<S;++v){
        double bal = std::fabs(sim_score[v]-0.5)+std::fabs(feat_norm[v]-0.5)+std::fabs(feat_var[v]-0.5);
        if (bal<bestbal){ bestbal=bal; best=v; }
    }
    return best;
}

bool DinoBackbone::forward_mv(const std::vector<std::vector<float>>& views_chw, int H, int W,
                              std::vector<std::vector<std::vector<float>>>& feats,
                              std::vector<std::vector<std::vector<float>>>& cam_tokens,
                              int* out_b_idx){
    const auto& c = ml_.config();
    const int S=(int)views_chw.size();
    const int THRESH_FOR_REF_SELECTION = 3;
    // S < threshold (or no alt path): no reference-view selection -> direct forward.
    if (S < THRESH_FOR_REF_SELECTION || c.alt_start < 0){
        if (out_b_idx) *out_b_idx = 0;
        return forward_mv_ordered(views_chw, H, W, feats, cam_tokens);
    }

    // --- Pass A: select reference view from layer-(alt_start-1) input = output of
    //     block (alt_start-2). Run per-view LOCAL blocks [0, alt_start-1). ---
    const int embed=(int)c.embed_dim;
    std::vector<std::vector<float>> cls;
    if (!capture_local_cls(views_chw, H, W, c.alt_start-1, cls)) return false;
    int b_idx = select_reference_view_saddle(cls, embed);
    if (out_b_idx) *out_b_idx = b_idx;

    // reorder_by_reference: position 0 <- b_idx; positions 1..b_idx <- 0..b_idx-1;
    // positions >b_idx unchanged.
    std::vector<int> reorder(S);
    reorder[0]=b_idx;
    for (int p=1;p<S;++p) reorder[p] = (p<=b_idx)? (p-1) : p;
    std::vector<std::vector<float>> rv(S);
    for (int p=0;p<S;++p) rv[p]=views_chw[reorder[p]];

    // --- Pass B: full forward on reordered views (ref at position 0). ---
    std::vector<std::vector<std::vector<float>>> rfeats, rcams;
    if (!forward_mv_ordered(rv, H, W, rfeats, rcams)) return false;

    // restore_original_order: target t<b_idx <- current t+1; target b_idx <- current 0;
    // target t>b_idx <- current t.
    std::vector<int> restore(S);
    for (int t=0;t<S;++t) restore[t] = (t<b_idx)? (t+1) : t;
    restore[b_idx]=0;
    const size_t NL = rfeats.size();
    feats.assign(NL, std::vector<std::vector<float>>(S));
    cam_tokens.assign(NL, std::vector<std::vector<float>>(S));
    for (size_t o=0;o<NL;++o) for (int t=0;t<S;++t){
        feats[o][t]      = rfeats[o][restore[t]];
        cam_tokens[o][t] = rcams[o][restore[t]];
    }
    return true;
}

bool DinoBackbone::forward_mv_ordered(const std::vector<std::vector<float>>& views_chw, int H, int W,
                              std::vector<std::vector<std::vector<float>>>& feats,
                              std::vector<std::vector<std::vector<float>>>& cam_tokens){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch;
    const int embed=(int)c.embed_dim, heads=(int)c.num_heads, hd=(int)c.head_dim;
    const int Npatch=gh*gw, Ntok=1+Npatch;
    const int S=(int)views_chw.size();
    const float eps=c.ln_eps;

    // Per-view RoPE position sets (identical across views: same patch grid).
    std::vector<float> pos_local(2*Ntok, 0.f), pos_nodiff(2*Ntok, 0.f);
    for (int t=1;t<Ntok;++t){ int idx=t-1; int row=idx/gw, col=idx%gw;
        pos_local[2*t+0]=(float)(row+1); pos_local[2*t+1]=(float)(col+1);
        pos_nodiff[2*t+0]=1.f;           pos_nodiff[2*t+1]=1.f; }
    RopeTables rt_local = build_rope_tables(pos_local, Ntok, hd, c.rope_freq);
    // Global attention sees all S*Ntok tokens (view-major). RoPE = pos_nodiff tiled S times.
    std::vector<float> pos_nodiff_g(2*(size_t)Ntok*S, 0.f);
    for (int s=0;s<S;++s) for (int t=0;t<Ntok;++t){
        pos_nodiff_g[2*((size_t)s*Ntok+t)+0]=pos_nodiff[2*t+0];
        pos_nodiff_g[2*((size_t)s*Ntok+t)+1]=pos_nodiff[2*t+1]; }
    RopeTables rt_nodiff_g = build_rope_tables(pos_nodiff_g, Ntok*S, hd, c.rope_freq);

    std::vector<float> pos = interp_pos_embed(gh, gw);

    // Camera token slots: ref = slot 0, src = slot 1 (camera_token ne0=embed, ne1=2).
    ggml_tensor* camt = ml_.tensor("vit.camera_token");
    std::vector<float> cam_ref(embed, 0.f), cam_src(embed, 0.f);
    if (camt){ const float* cp=(const float*)camt->data;
        for (int e=0;e<embed;++e){ cam_ref[e]=cp[e]; cam_src[e]=cp[(size_t)embed+e]; } }

    const std::vector<int32_t>& outL = c.out_layers;
    const size_t NL = outL.size();
    feats.assign(NL, std::vector<std::vector<float>>(S));
    cam_tokens.assign(NL, std::vector<std::vector<float>>(S));

    // Per-out-layer captures of the FULL [embed,Ntok,S] local_x and x tensors.
    std::vector<std::vector<float>> cap_local(NL), cap_x(NL);

    GraphInputPool pool;
    std::vector<float> throwaway;
    bool ok = be_.forward_capture([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t pne[2]={embed, Ntok};
        ggml_tensor* pe = be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2);
        // Build per-view tokens, concat along dim 2 -> [embed, Ntok, S] (view-major).
        ggml_tensor* x = nullptr;
        for (int s=0;s<S;++s){
            const int64_t ine[4]={W,H,3,1};
            ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, views_chw[s].data(), ine, 4);
            ggml_tensor* xv = ggml_conv_2d(ctx, ml_.tensor("vit.patch_embed.weight"), img, patch,patch,0,0,1,1);
            xv = ggml_reshape_2d(ctx, xv, (int64_t)Npatch, embed);
            xv = ggml_cont(ctx, ggml_transpose(ctx, xv));
            xv = ggml_add(ctx, xv, ml_.tensor("vit.patch_embed.bias"));
            ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
            xv = ggml_concat(ctx, cls, xv, 1);            // [embed, Ntok]
            xv = ggml_add(ctx, xv, pe);
            xv = ggml_reshape_3d(ctx, xv, embed, Ntok, 1);
            x = (s==0)? xv : ggml_concat(ctx, x, xv, 2);
        }

        // RoPE inputs (local + global nodiff) and camera-token slots.
        ggml_tensor *clb,*slb,*cng,*sng;
        build_rope_inputs(ctx, be_, pool, rt_local,    clb, slb);
        build_rope_inputs(ctx, be_, pool, rt_nodiff_g, cng, sng);
        const int64_t camne[2]={embed,1};
        ggml_tensor* cam_ref_in = be_.add_graph_input_nd(ctx, pool, cam_ref.data(), camne, 2);
        ggml_tensor* cam_src_in = be_.add_graph_input_nd(ctx, pool, cam_src.data(), camne, 2);

        ggml_tensor* local_x = x;
        for (int i=0;i<(int)c.depth;++i){
            // Cam-token overwrite BEFORE block i==alt_start: view0 t0<-ref, views>=1 t0<-src.
            if (c.alt_start>=0 && i==c.alt_start){
                ggml_tensor* newx=nullptr;
                for (int s=0;s<S;++s){
                    ggml_tensor* vs = ggml_cont(ctx, ggml_view_2d(ctx, x, embed, Ntok,
                                                    x->nb[1], (size_t)s*x->nb[2]));
                    ggml_tensor* rest = ggml_cont(ctx, ggml_view_2d(ctx, vs, embed, Ntok-1,
                                                    vs->nb[1], vs->nb[1]));
                    ggml_tensor* cam = (s==0)? cam_ref_in : cam_src_in;
                    ggml_tensor* nv = ggml_concat(ctx, cam, rest, 1);   // [embed,Ntok]
                    nv = ggml_reshape_3d(ctx, nv, embed, Ntok, 1);
                    newx = (s==0)? nv : ggml_concat(ctx, newx, nv, 2);
                }
                x = newx;
            }
            const bool global  = (c.alt_start>=0 && i>=c.alt_start && (i%2==1));
            const bool use_rope = (c.rope_start>=0 && i>=c.rope_start);
            BlockWeights bw = load_block(ml_, i);
            if (global){
                // Cross-view: flatten [embed,Ntok,S] -> [embed,Ntok*S] (view-major), one attention.
                ggml_tensor* cb = use_rope? cng : nullptr;
                ggml_tensor* sb = use_rope? sng : nullptr;
                ggml_tensor* xf = ggml_reshape_2d(ctx, ggml_cont(ctx, x), embed, (int64_t)Ntok*S);
                xf = vit_block(ctx, xf, bw, heads, hd, eps, cb, sb);
                // Materialize as a real (non-view) tensor: a bare reshape view's backing
                // buffer can be reused by the allocator, corrupting intermediate captures.
                x  = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, xf), embed, Ntok, S));
            } else {
                // Local: per-view independent attention.
                ggml_tensor* cb = use_rope? clb : nullptr;
                ggml_tensor* sb = use_rope? slb : nullptr;
                ggml_tensor* nx=nullptr;
                for (int s=0;s<S;++s){
                    ggml_tensor* vs = ggml_cont(ctx, ggml_view_2d(ctx, x, embed, Ntok,
                                                    x->nb[1], (size_t)s*x->nb[2]));
                    vs = vit_block(ctx, vs, bw, heads, hd, eps, cb, sb);
                    vs = ggml_reshape_3d(ctx, vs, embed, Ntok, 1);
                    nx = (s==0)? vs : ggml_concat(ctx, nx, vs, 2);
                }
                x = nx;
                local_x = x;
            }
            for (size_t o=0;o<NL;++o) if (outL[o]==i){
                be_.capture(local_x, &cap_local[o]);
                be_.capture(x,       &cap_x[o]);
            }
        }
        return x;
    }, throwaway);
    if (!ok) return false;

    // Host post-process per view (matches the S=1 path, repeated per view-slice).
    ggml_tensor* nw = ml_.host_tensor("vit.norm.weight");
    ggml_tensor* nb = ml_.host_tensor("vit.norm.bias");
    const float* nwp=(const float*)nw->data;
    const float* nbp=(const float*)nb->data;
    auto layernorm_host=[&](const float* row)->std::vector<float>{
        double mean=0; for(int e=0;e<embed;++e) mean+=row[e]; mean/=embed;
        double var=0; for(int e=0;e<embed;++e){ double d=row[e]-mean; var+=d*d; } var/=embed;
        double inv=1.0/std::sqrt(var+(double)eps); std::vector<float> o(embed);
        for(int e=0;e<embed;++e) o[e]=(float)((row[e]-mean)*inv)*nwp[e]+nbp[e];
        return o;
    };
    const size_t view_stride=(size_t)Ntok*embed;
    for (size_t o=0;o<NL;++o){
        for (int s=0;s<S;++s){
            const float* lx=&cap_local[o][(size_t)s*view_stride]; // [embed,Ntok]
            const float* xx=&cap_x[o][(size_t)s*view_stride];
            std::vector<float> camcat((size_t)2*embed);
            for(int e=0;e<embed;++e){ camcat[e]=lx[e]; camcat[embed+e]=xx[e]; }
            cam_tokens[o][s]=std::move(camcat);
            std::vector<float> f((size_t)Npatch*2*embed);
            for(int t=1;t<Ntok;++t){
                const float* lrow=&lx[(size_t)t*embed];
                const float* xrow=&xx[(size_t)t*embed];
                std::vector<float> no=layernorm_host(xrow);
                float* dst=&f[(size_t)(t-1)*2*embed];
                for(int e=0;e<embed;++e){ dst[e]=lrow[e]; dst[embed+e]=no[e]; }
            }
            feats[o][s]=std::move(f);
        }
    }
    return true;
}
}
