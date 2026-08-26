#pragma once

#include "tsdf.hpp"

#include <cstdint>
#include <vector>

namespace da {

// DTVM v1 is a temporal exposed-face mesh.  Each record identifies one exact
// integer-grid face and the half-open frame interval [first_frame,end_frame)
// during which it is visible.  UINT16_MAX means the face remains visible.
// Geometry is derived from (voxel key,direction), so no rounded world-space
// vertices or complete per-frame meshes are stored.
//
// Header (32 bytes, little endian):
//   magic[4]="DTVM", u16 version=1, u16 header_size=32,
//   u32 flags, u32 frame_count, u32 face_count, u32 final_face_count,
//   f32 voxel_size, u32 record_size=24.
// Record (24 bytes, little endian):
//   i32 x,y,z, u16 first_frame,end_frame, u8 direction,r,g,b, u32 reserved.
// Directions are 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z.
std::vector<uint8_t> encode_temporal_voxel_mesh(const TsdfSurface& surface,
                                                uint32_t frame_count);

} // namespace da
