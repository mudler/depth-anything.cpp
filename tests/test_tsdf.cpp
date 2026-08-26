// Ground-truth de-ghosting test for the normal-space TSDF (src/tsdf.cpp, task A).
//
// This is the HONEST ghost-separation metric the point-cloud harness lacked: build
// a surface we KNOW is doubled — two parallel sheets a fixed gap apart, straddling
// the true surface — and measure the perpendicular spread (thickness) to the known
// midplane before and after fusion. A real de-ghoster drives the thickness toward
// zero (the two sheets become one); a mere downsampler leaves it at ~gap/2 (both
// sheets survive, just sparser). We assert BOTH: fuse_tsdf collapses the double,
// and fuse_voxel (same voxel size) provably does NOT — so a passing run is proof
// the TSDF removes ghosting rather than just thinning the cloud.
#include "tsdf.hpp"
#include "fuse.hpp"
#include "geom_metrics.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

using namespace da;

static unsigned long long g_seed = 0x9e3779b97f4a7c15ULL;
static double urand(){ g_seed^=g_seed<<13; g_seed^=g_seed>>7; g_seed^=g_seed<<17;
                       return (double)(g_seed>>11)/9007199254740992.0; }
static double sym(){ return urand()*2.0-1.0; }

static int fails = 0;
static void check(bool ok, const char* msg){
    if(!ok){ std::fprintf(stderr,"FAIL: %s\n",msg); fails++; } else std::fprintf(stderr,"ok:   %s\n",msg);
}
static std::vector<double> to_d(const std::vector<float>& v, int n){
    std::vector<double> o(3*n); for (int i=0;i<3*n;++i) o[i]=v[i]; return o;
}
// Mean |z| — bias of the fused sheet off the true midplane (should stay ~0).
static double mean_abs_z(const std::vector<float>& v, int n){
    double s=0; for (int i=0;i<n;++i) s+=std::fabs(v[3*i+2]); return n? s/n : 0;
}

int main() {
    // A physical surface at z=0, observed as a DOUBLED sheet: copies at z=+/-gap/2
    // (residual mono/stitch misalignment) with a little in-sheet z-noise. The gap is
    // 3 voxels — larger than a cell, so a voxel downsample keeps both sheets. A
    // camera far up +z orients the normals (TSDF sign must be globally consistent).
    const int    NX   = 8000;
    const double gap   = 0.06;       // 3 * voxel
    const double vox   = 0.02;
    const double znoise = 0.004;     // << 0.5*vox, doesn't move the crossing
    const double p0[3] = {0,0,0}, nz[3] = {0,0,1};
    std::vector<float> cam = {0.f, 0.f, 5.f};   // single observing camera up +z

    auto build = [&](std::vector<float>& xyz, std::vector<uint8_t>& rgb, std::vector<float>& rad){
        xyz.clear(); rgb.clear(); rad.clear();
        for (int i=0;i<NX;++i){
            double x=sym(), y=sym();
            for (double zc : {+gap/2, -gap/2}) {
                double z = zc + sym()*znoise;
                xyz.push_back((float)x); xyz.push_back((float)y); xyz.push_back((float)z);
                rgb.push_back(180); rgb.push_back(180); rgb.push_back(180);
                rad.push_back(0.01f);
            }
        }
    };

    std::vector<float> xyz, rad; std::vector<uint8_t> rgb;
    build(xyz, rgb, rad);
    const int Nin = (int)(xyz.size()/3);
    double th_before = da::eval::thickness_to_plane(to_d(xyz,Nin).data(), Nin, p0, nz);
    std::fprintf(stderr, "input: %d pts, gap=%.3f, thickness=%.4f (expect ~gap/2=%.3f)\n",
                 Nin, gap, th_before, gap/2);
    check(th_before > 0.8*(gap/2), "doubled input starts thick (~gap/2)");

    // ---- fuse_tsdf: should MERGE the two sheets into one at the midplane ----
    {
        std::vector<float> tx=xyz, tr=rad; std::vector<uint8_t> tc=rgb;
        TsdfParams tp; tp.voxel=(float)vox; tp.trunc=5.f*(float)vox; tp.normal_radius=2.5f*(float)vox;
        TsdfSurface surface;
        int M = fuse_tsdf(tx, tc, tr, tp, nullptr, &cam, nullptr, nullptr, &surface);
        double th_after = da::eval::thickness_to_plane(to_d(tx,M).data(), M, p0, nz);
        double bias = mean_abs_z(tx, M);
        std::fprintf(stderr, "tsdf : %d pts, thickness=%.4f (%.1f%% of input), mean|z|=%.4f\n",
                     M, th_after, 100.0*th_after/th_before, bias);
        check(M > 1000 && M < Nin, "tsdf outputs a single sheet (fewer pts, non-empty)");
        check((int)surface.voxels.size() == M && std::abs(surface.voxel_size-(float)vox) < 1e-6f,
              "tsdf exposes exact occupancy alongside every output point");
        check(th_after < 0.35*th_before, "tsdf COLLAPSES the double (thickness -> near zero)");
        check(bias < 0.5*vox, "tsdf sheet sits on the true midplane (no bias)");
    }

    // ---- fuse_voxel at the same cell: provably does NOT merge (control) ----
    {
        std::vector<float> vx=xyz, vr=rad; std::vector<uint8_t> vc=rgb;
        FuseParams fp; fp.radius=(float)vox; fp.voxel=(float)vox;   // same scale as tsdf
        int M = fuse_voxel(vx, vc, vr, fp);
        double th_after = da::eval::thickness_to_plane(to_d(vx,M).data(), M, p0, nz);
        std::fprintf(stderr, "voxel: %d pts, thickness=%.4f (%.1f%% of input) [control]\n",
                     M, th_after, 100.0*th_after/th_before);
        check(th_after > 0.6*th_before, "voxel downsample leaves the double (still ghosted)");
    }

    std::fprintf(stderr, fails? "\n%d CHECK(S) FAILED\n" : "\nall tsdf checks passed\n", fails);
    return fails ? 1 : 0;
}
