#include "tsdf.hpp"
#include "icp.hpp"   // estimate_normals

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace da {

// Integer voxel coordinate (fixed origin at world 0), matching fuse.cpp's scheme.
struct VKey {
    int32_t x, y, z;
    bool operator==(const VKey& o) const { return x==o.x && y==o.y && z==o.z; }
};
struct VKeyHash {
    size_t operator()(const VKey& k) const {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B185EBCA87ULL;
        h ^= (uint64_t)(uint32_t)k.y * 0xC2B2AE3D27D4EB4FULL + (h<<6) + (h>>2);
        h ^= (uint64_t)(uint32_t)k.z * 0x165667B19E3779F9ULL + (h<<6) + (h>>2);
        return (size_t)h;
    }
};
static inline VKey key_at(double x, double y, double z, double cell) {
    return { (int32_t)std::floor(x/cell), (int32_t)std::floor(y/cell), (int32_t)std::floor(z/cell) };
}

// Colours must be averaged in LINEAR light, not in gamma-encoded sRGB: averaging
// sRGB values pulls the mean toward black (the transfer curve is concave), so a
// voxel merging any colour contrast comes out visibly darker. Convert sRGB u8 ->
// linear before accumulating and back on output so merged colour keeps its brightness.
static inline double srgb_to_lin(double v) {
    double c = v / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
static inline double lin_to_srgb(double c) {
    double s = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return s * 255.0;
}

// Per-voxel signed-distance accumulator. sdf/nrm/col are confidence-weighted sums;
// divide by w at the end. hits counts contributing deposits (speckle cull).
struct Cell {
    double w = 0, sdf = 0;                 // Σ wi , Σ wi*si
    double nx = 0, ny = 0, nz = 0;         // Σ wi*ni (averaged surface normal)
    double cr = 0, cg = 0, cb = 0;         // Σ wi*rgb
    double rad = 0;                        // Σ wi*radius
    int    hits = 0;
    int    fmin = INT32_MAX;               // first (min) input frame that fed this voxel
};

int fuse_tsdf(std::vector<float>& xyz, std::vector<uint8_t>& rgb,
              std::vector<float>& radius, const TsdfParams& p,
              const std::vector<float>* weights,
              const std::vector<float>* cameras,
              const std::vector<int>* in_frame,
              std::vector<int>* out_frame,
              TsdfSurface* out_surface) {
    if (out_surface) { out_surface->voxel_size = 0.f; out_surface->voxels.clear(); }
    const int N = (int)(xyz.size()/3);
    if (N < 8) return N;
    const bool have_rgb = ((int)rgb.size() == 3*N);
    const bool have_rad = ((int)radius.size() == N);
    const bool have_w   = (weights && (int)weights->size() == N);
    const bool have_fr  = (in_frame && (int)in_frame->size() == N);

    // --- scene-relative defaults (arbitrary monocular scale) ---
    float lo[3] = {xyz[0],xyz[1],xyz[2]}, hi[3] = {xyz[0],xyz[1],xyz[2]};
    for (int i = 0; i < N; ++i) for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], xyz[3*i+k]); hi[k] = std::max(hi[k], xyz[3*i+k]); }
    const double diag = std::sqrt((double)(hi[0]-lo[0])*(hi[0]-lo[0])
                                + (double)(hi[1]-lo[1])*(hi[1]-lo[1])
                                + (double)(hi[2]-lo[2])*(hi[2]-lo[2]));
    const double vfrac = (p.voxel_frac > 0.f) ? p.voxel_frac : 0.004;
    double vox = (p.voxel > 0.f) ? p.voxel : (diag > 0 ? vfrac*diag : 0.03);
    if (!(vox > 0)) vox = 0.03;
    const double tmult = (p.trunc_mult > 0.f) ? p.trunc_mult : 4.0;
    double trunc = (p.trunc > 0.f) ? p.trunc : tmult*vox;
    double nr    = (p.normal_radius > 0.f) ? p.normal_radius : 2.5*vox;
    const int min_hits = std::max(1, p.min_hits);

    // --- per-point normals (PCA, unoriented) ---
    std::vector<double> pd((size_t)3*N);
    for (int i = 0; i < 3*N; ++i) pd[i] = xyz[i];
    std::vector<double> nrm;
    estimate_normals(pd, nr, nullptr, nrm);   // fills 3N; zero where too few neighbours

    // --- orient each normal toward its nearest camera so the SDF sign is globally
    // consistent. PCA's eigenvector sign is arbitrary per point; without this,
    // half a sheet's deposits cancel the other half and fusion collapses to noise. ---
    const int ncam = cameras ? (int)(cameras->size()/3) : 0;
    if (ncam > 0) {
        const float* cam = cameras->data();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double nx = nrm[3*i], ny = nrm[3*i+1], nz = nrm[3*i+2];
            if (nx==0 && ny==0 && nz==0) continue;
            const double px = pd[3*i], py = pd[3*i+1], pz = pd[3*i+2];
            int best = 0; double best2 = 1e300;
            for (int c = 0; c < ncam; ++c) {
                double dx = cam[3*c]-px, dy = cam[3*c+1]-py, dz = cam[3*c+2]-pz;
                double d2 = dx*dx+dy*dy+dz*dz;
                if (d2 < best2) { best2 = d2; best = c; }
            }
            double vx = cam[3*best]-px, vy = cam[3*best+1]-py, vz = cam[3*best+2]-pz;
            if (nx*vx + ny*vy + nz*vz < 0) { nrm[3*i]=-nx; nrm[3*i+1]=-ny; nrm[3*i+2]=-nz; }
        }
    }

    // --- splat each point's truncated SDF along its normal into the field ---
    std::unordered_map<VKey, Cell, VKeyHash> field;
    field.reserve((size_t)N);
    const int K = (int)std::ceil(trunc / vox);           // voxels of band per side
    const double eps = 1e-12;
    for (int i = 0; i < N; ++i) {
        double nx = nrm[3*i], ny = nrm[3*i+1], nz = nrm[3*i+2];
        if (nx==0 && ny==0 && nz==0) continue;            // no reliable normal -> skip
        const double px = pd[3*i], py = pd[3*i+1], pz = pd[3*i+2];
        double w = have_w ? (double)(*weights)[i]
                          : (have_rad ? 1.0/((double)radius[i] + eps) : 1.0);
        if (!(w > 0)) w = eps;
        const double cr = srgb_to_lin(have_rgb ? rgb[3*i]   : 200.0);   // accumulate in
        const double cg = srgb_to_lin(have_rgb ? rgb[3*i+1] : 200.0);   // linear light so
        const double cb = srgb_to_lin(have_rgb ? rgb[3*i+2] : 200.0);   // merging keeps brightness
        const double ri = have_rad ? radius[i]  : (float)vox;
        // March along ±normal one voxel at a time; deposit into each distinct voxel
        // the SIGNED distance from the surface (point) to that voxel's centre.
        VKey last = { INT32_MIN, INT32_MIN, INT32_MIN };
        for (int k = -K; k <= K; ++k) {
            double qx = px + (k*vox)*nx, qy = py + (k*vox)*ny, qz = pz + (k*vox)*nz;
            VKey key = key_at(qx, qy, qz, vox);
            if (key == last) continue;                    // same voxel as previous step
            last = key;
            double cx = (key.x + 0.5)*vox, cy = (key.y + 0.5)*vox, cz = (key.z + 0.5)*vox;
            double s = nx*(cx-px) + ny*(cy-py) + nz*(cz-pz);   // signed dist along normal
            if (s > trunc || s < -trunc) continue;
            Cell& c = field[key];
            c.w += w; c.sdf += w*s;
            c.nx += w*nx; c.ny += w*ny; c.nz += w*nz;
            c.cr += w*cr; c.cg += w*cg; c.cb += w*cb;
            c.rad += w*ri; c.hits++;
            if (have_fr) { int f = (*in_frame)[i]; if (f < c.fmin) c.fmin = f; }
        }
    }

    // --- extract the zero-crossing shell: voxels whose centre is within half a cell
    // of the surface. Move each onto the level set along its averaged normal. ---
    struct Out { VKey k; int frame; float x,y,z; uint8_t r,g,b; float rad; };
    std::vector<Out> outs;
    outs.reserve(field.size()/2 + 16);
    const double half = 0.5*vox;
    for (const auto& kv : field) {
        const Cell& c = kv.second;
        if (c.hits < min_hits || !(c.w > 0)) continue;
        double D = c.sdf / c.w;
        if (D > half || D < -half) continue;              // not a surface voxel
        double nx = c.nx, ny = c.ny, nz = c.nz;
        double ln = std::sqrt(nx*nx+ny*ny+nz*nz);
        if (ln > 0) { nx/=ln; ny/=ln; nz/=ln; } else { nx=ny=0; nz=1; }
        double cx = (kv.first.x + 0.5)*vox, cy = (kv.first.y + 0.5)*vox, cz = (kv.first.z + 0.5)*vox;
        // Surface point = voxel centre pushed back onto the zero level set.
        Out o;
        o.k = kv.first;
        o.x = (float)(cx - D*nx); o.y = (float)(cy - D*ny); o.z = (float)(cz - D*nz);
        // Reveal frame = the FIRST input frame that observed this voxel, so the
        // additive flythrough reveals each surface as soon as any camera has seen it
        // and never hides it again (nearest-camera instead back-loads the reveal).
        o.frame = (c.fmin == INT32_MAX) ? 0 : c.fmin;
        double iw = 1.0/c.w;
        auto c8 = [](double v){ double x=std::max(0.0,std::min(255.0,v)); return (uint8_t)std::lround(x); };
        o.r = c8(lin_to_srgb(c.cr*iw)); o.g = c8(lin_to_srgb(c.cg*iw)); o.b = c8(lin_to_srgb(c.cb*iw));
        // Size each splat to its voxel so the one-point-per-cell surface tiles solidly.
        // Keeping the (much smaller) averaged input radius leaves dark gaps between the
        // now voxel-spaced points, which reads as a dark, sparse surface.
        o.rad = (float)(0.6 * vox);
        outs.push_back(o);
    }
    // Order FRAME-MAJOR (so the flythrough/build-up reveal, which reveals a frame
    // prefix, shows points as their first-observing camera is reached), with the
    // voxel key as a deterministic tiebreak (hash iteration is unspecified).
    std::sort(outs.begin(), outs.end(), [](const Out& a, const Out& b){
        if (a.frame != b.frame) return a.frame < b.frame;
        if (a.k.x != b.k.x) return a.k.x < b.k.x;
        if (a.k.y != b.k.y) return a.k.y < b.k.y;
        return a.k.z < b.k.z;
    });

    const int M = (int)outs.size();
    if (M == 0) return N;   // nothing extracted (degenerate) -> leave input untouched
    if (out_surface) {
        out_surface->voxel_size = (float)vox;
        out_surface->voxels.reserve(outs.size());
        for (const Out& o : outs) {
            out_surface->voxels.push_back({o.k.x, o.k.y, o.k.z, o.r, o.g, o.b, o.frame});
        }
    }
    std::vector<float> ox(3*M), orad(M); std::vector<uint8_t> orgb(3*M);
    if (out_frame) out_frame->resize(M);
    for (int i = 0; i < M; ++i) {
        ox[3*i]=outs[i].x; ox[3*i+1]=outs[i].y; ox[3*i+2]=outs[i].z;
        orgb[3*i]=outs[i].r; orgb[3*i+1]=outs[i].g; orgb[3*i+2]=outs[i].b;
        orad[i]=outs[i].rad;
        if (out_frame) (*out_frame)[i] = outs[i].frame;
    }
    xyz.swap(ox);
    if (have_rgb) rgb.swap(orgb); else rgb.clear();
    if (have_rad) radius.swap(orad); else radius.clear();
    return M;
}

} // namespace da
