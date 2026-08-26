#include "voxel_mesh.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace da;

static uint16_t u16(const std::vector<uint8_t>& b, size_t p) {
    return (uint16_t)(b[p] | ((uint16_t)b[p+1] << 8));
}
static uint32_t u32(const std::vector<uint8_t>& b, size_t p) {
    return (uint32_t)b[p] | ((uint32_t)b[p+1] << 8) |
           ((uint32_t)b[p+2] << 16) | ((uint32_t)b[p+3] << 24);
}
static int32_t i32(const std::vector<uint8_t>& b, size_t p) { return (int32_t)u32(b, p); }

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* msg) {
        std::fprintf(stderr, "%s %s\n", ok ? "ok  " : "FAIL", msg);
        if (!ok) ++fails;
    };

    TsdfSurface empty;
    check(encode_temporal_voxel_mesh(empty, 3).empty(), "empty surface emits no artifact");

    TsdfSurface one; one.voxel_size = 0.25f;
    one.voxels.push_back({2,-3,4,10,20,30,1});
    auto a = encode_temporal_voxel_mesh(one, 4);
    check(a.size() == 32 + 6*24, "single voxel emits six face records");
    check(a[0]=='D' && a[1]=='T' && a[2]=='V' && a[3]=='M' && u16(a,4)==1, "v1 magic and version");
    check(u32(a,16)==6 && u32(a,20)==6 && u32(a,28)==24, "single voxel header counts");
    check(i32(a,32)==2 && i32(a,36)==-3 && i32(a,40)==4, "exact signed integer key retained");
    check(u16(a,44)==1 && u16(a,46)==0xffff, "permanent face lifetime retained");
    check(a[49]==10 && a[50]==20 && a[51]==30, "RGB packed losslessly");

    TsdfSurface pair; pair.voxel_size = 1.f;
    pair.voxels = {{0,0,0,100,0,0,1}, {1,0,0,0,100,0,3}};
    auto p = encode_temporal_voxel_mesh(pair, 5);
    // Ten final exterior faces plus A's +X face, which is visible for [1,3).
    check(u32(p,16)==11 && u32(p,20)==10, "adjacent voxels cull shared final face but retain transient face");
    bool transient = false, forbidden = false;
    for (uint32_t n=0; n<u32(p,16); ++n) {
        size_t o = 32 + n*24;
        int x=i32(p,o), y=i32(p,o+4), z=i32(p,o+8); uint8_t d=p[o+16];
        if (x==0 && y==0 && z==0 && d==1 && u16(p,o+12)==1 && u16(p,o+14)==3) transient=true;
        if (x==1 && y==0 && z==0 && d==0) forbidden=true;
    }
    check(transient && !forbidden, "shared face has exactly the earlier voxel's lifetime");

    TsdfSurface same; same.voxel_size=1.f;
    same.voxels={{0,0,0,1,2,3,2},{1,0,0,4,5,6,2}};
    auto s=encode_temporal_voxel_mesh(same,3);
    check(u32(s,16)==10 && u32(s,20)==10, "same-frame neighbours never expose their shared face");

    TsdfSurface shuffled=pair;
    std::swap(shuffled.voxels[0],shuffled.voxels[1]);
    check(encode_temporal_voxel_mesh(shuffled,5)==p, "encoding is deterministic under input permutation");

    return fails ? 1 : 0;
}
