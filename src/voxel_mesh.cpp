#include "voxel_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace da {
namespace {

struct Key {
    int32_t x, y, z;
    bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B185EBCA87ULL;
        h ^= (uint64_t)(uint32_t)k.y * 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.z * 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct Face {
    Key key;
    uint16_t first, end;
    uint8_t dir, r, g, b;
};

static void put_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back((uint8_t)(v & 0xff));
    out.push_back((uint8_t)(v >> 8));
}
static void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back((uint8_t)(v >> (8 * i)));
}
static void put_i32(std::vector<uint8_t>& out, int32_t v) { put_u32(out, (uint32_t)v); }
static void put_f32(std::vector<uint8_t>& out, float v) {
    uint32_t u = 0;
    static_assert(sizeof(u) == sizeof(v), "float must be 32-bit");
    std::memcpy(&u, &v, sizeof(u));
    put_u32(out, u);
}

static uint16_t frame16(int frame, uint32_t frame_count) {
    if (frame < 0) return 0;
    uint32_t f = (uint32_t)frame;
    if (frame_count > 0 && f >= frame_count) f = frame_count - 1;
    return (uint16_t)std::min<uint32_t>(f, std::numeric_limits<uint16_t>::max() - 1);
}

} // namespace

std::vector<uint8_t> encode_temporal_voxel_mesh(const TsdfSurface& surface,
                                                uint32_t frame_count) {
    if (!(surface.voxel_size > 0.f) || !std::isfinite(surface.voxel_size) ||
        surface.voxels.empty()) return {};

    std::unordered_map<Key, const TsdfVoxel*, KeyHash> occupied;
    occupied.reserve(surface.voxels.size() * 2 + 1);
    for (const TsdfVoxel& v : surface.voxels) occupied[{v.x, v.y, v.z}] = &v;

    static constexpr int dxyz[6][3] = {
        {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
    };
    std::vector<Face> faces;
    faces.reserve(surface.voxels.size() * 3);
    uint32_t final_faces = 0;
    for (const TsdfVoxel& v : surface.voxels) {
        const uint16_t vf = frame16(v.first_frame, frame_count);
        for (uint8_t dir = 0; dir < 6; ++dir) {
            Key nk{v.x + dxyz[dir][0], v.y + dxyz[dir][1], v.z + dxyz[dir][2]};
            auto it = occupied.find(nk);
            uint16_t end = std::numeric_limits<uint16_t>::max();
            if (it != occupied.end()) {
                const uint16_t nf = frame16(it->second->first_frame, frame_count);
                // A shared face exists only while this voxel is present and its
                // neighbour is not. Equal-frame neighbours never expose it.
                if (nf <= vf) continue;
                end = nf;
            } else {
                ++final_faces;
            }
            faces.push_back({{v.x,v.y,v.z}, vf, end, dir, v.r, v.g, v.b});
        }
    }
    std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) {
        if (a.first != b.first) return a.first < b.first;
        if (a.end != b.end) return a.end < b.end;
        if (a.key.x != b.key.x) return a.key.x < b.key.x;
        if (a.key.y != b.key.y) return a.key.y < b.key.y;
        if (a.key.z != b.key.z) return a.key.z < b.key.z;
        return a.dir < b.dir;
    });

    constexpr uint16_t header_size = 32;
    constexpr uint32_t record_size = 24;
    std::vector<uint8_t> out;
    out.reserve(header_size + record_size * faces.size());
    out.insert(out.end(), {'D','T','V','M'});
    put_u16(out, 1);
    put_u16(out, header_size);
    put_u32(out, 0); // flags
    put_u32(out, frame_count);
    put_u32(out, (uint32_t)faces.size());
    put_u32(out, final_faces);
    put_f32(out, surface.voxel_size);
    put_u32(out, record_size);
    for (const Face& f : faces) {
        put_i32(out, f.key.x); put_i32(out, f.key.y); put_i32(out, f.key.z);
        put_u16(out, f.first); put_u16(out, f.end);
        out.push_back(f.dir); out.push_back(f.r); out.push_back(f.g); out.push_back(f.b);
        put_u32(out, 0);
    }
    return out;
}

} // namespace da
