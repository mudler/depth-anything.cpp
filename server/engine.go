// engine.go — purego bindings to libdepthanything (the flat C API in
// include/da_capi.h, ABI 11). No cgo: the shared library is dlopen'd once and each
// model is loaded as its own da_ctx. A context is NOT thread-safe (one ggml
// backend + compute graph), so all inference is serialized by the caller.
package main

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// capi is the dlopen'd library with the DA3 C API bound by name.
type capi struct {
	handle uintptr

	abiVersion  func() int32
	load        func(string, int32) uintptr
	loadNested  func(string, string, int32) uintptr
	freeCtx     func(uintptr)
	lastErrP    func(uintptr) uintptr
	freeFloats  func(uintptr)
	freeBytes   func(uintptr)
	pointsMulti func(ctx uintptr, paths uintptr, n int32, confPct float64, ptSize float32,
		outN, outCounts, outXyz, outRgb, outRad unsafe.Pointer) int32
	pointsStream func(ctx uintptr, paths uintptr, n, chunk, overlap int32, confPct float64,
		ptSize float32, budget int32, icpRefine, loopClose, fuse, metric int32, fuseVoxel float64,
		outN, outCounts, outXyz, outRgb, outRad unsafe.Pointer) int32
	streamPoses     func(ctx uintptr, outPos, outFwd, outNFrames unsafe.Pointer) int32
	streamVoxelMesh func(ctx uintptr, outData, outSize unsafe.Pointer) int32
	// Scene-relative TSDF fusion knobs applied to the next pointsStream fuse.
	setFuseParams        func(ctx uintptr, voxelFrac, truncMult float64)
	setTemporalVoxelMesh func(ctx uintptr, enabled int32)
	gaussians            func(ctx uintptr, path string,
		outN, outXyz, outScale, outQuat, outRgb, outOpacity,
		outIntr, outW, outH unsafe.Pointer) int32
}

func openCAPI(libPath string) (*capi, error) {
	h, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", libPath, err)
	}
	a := &capi{handle: h}
	purego.RegisterLibFunc(&a.abiVersion, h, "da_capi_abi_version")
	if abi := a.abiVersion(); abi < 11 {
		return nil, fmt.Errorf("libdepthanything ABI %d is too old; demo requires >=11", abi)
	}
	purego.RegisterLibFunc(&a.load, h, "da_capi_load")
	purego.RegisterLibFunc(&a.loadNested, h, "da_capi_load_nested")
	purego.RegisterLibFunc(&a.freeCtx, h, "da_capi_free")
	purego.RegisterLibFunc(&a.lastErrP, h, "da_capi_last_error")
	purego.RegisterLibFunc(&a.freeFloats, h, "da_capi_free_floats")
	purego.RegisterLibFunc(&a.freeBytes, h, "da_capi_free_bytes")
	purego.RegisterLibFunc(&a.pointsMulti, h, "da_capi_points_multi")
	purego.RegisterLibFunc(&a.pointsStream, h, "da_capi_points_stream")
	purego.RegisterLibFunc(&a.streamPoses, h, "da_capi_stream_last_poses")
	purego.RegisterLibFunc(&a.streamVoxelMesh, h, "da_capi_stream_last_voxel_mesh")
	purego.RegisterLibFunc(&a.setFuseParams, h, "da_capi_set_fuse_params")
	purego.RegisterLibFunc(&a.setTemporalVoxelMesh, h, "da_capi_set_temporal_voxel_mesh")
	purego.RegisterLibFunc(&a.gaussians, h, "da_capi_gaussians")
	return a, nil
}

func (a *capi) lastErr(ctx uintptr) string { return cstr(a.lastErrP(ctx)) }

// cstr reads a NUL-terminated C string at p (0 -> "").
func cstr(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(unsafe.Pointer(p + i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

func cFloats(p uintptr, n int) []float32 {
	if p == 0 || n <= 0 {
		return nil
	}
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(p)), n))
	return out
}

func cBytes(p uintptr, n int) []byte {
	if p == 0 || n <= 0 {
		return nil
	}
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
	return out
}

// Cloud is a fused multi-view colored point cloud (world frame, OpenCV axes).
type Cloud struct {
	N       int
	XYZ     []float32 // 3N
	RGB     []byte    // 3N
	Rad     []float32 // N (per-point world radius)
	Normals []float32 // 3N surface normals (OpenCV space), empty unless surfels requested
	Counts  []int32   // per source view (frames outer; for build-up prefix sums)
	// Per-input-frame camera poses (OpenCV world axes), for viewer flythrough. Length 3F each.
	FramePos          []float32 // global camera centre per frame
	FrameFwd          []float32 // global unit view direction per frame
	TemporalVoxelMesh []byte    // DTVM v1 exposed-face lifetimes, when requested
}

// PointsMulti runs ONE cross-view pass over the given image files and returns the
// fused coherent cloud. confPct in [0,100], ptSize multiplies the per-point radius.
func (a *capi) PointsMulti(ctx uintptr, paths []string, confPct float64, ptSize float32) (*Cloud, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no images")
	}
	// Build a C char*[] in Go memory. Go's heap is non-moving, and the C side only
	// reads the strings during the (synchronous) call, so this is safe with KeepAlive.
	cs := make([][]byte, len(paths))
	pp := make([]*byte, len(paths))
	for i, s := range paths {
		b := append([]byte(s), 0)
		cs[i] = b
		pp[i] = &b[0]
	}
	var n int32
	counts := make([]int32, len(paths))
	var pXyz, pRgb, pRad uintptr
	rc := a.pointsMulti(ctx, uintptr(unsafe.Pointer(&pp[0])), int32(len(paths)), confPct, ptSize,
		unsafe.Pointer(&n), unsafe.Pointer(&counts[0]),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pRgb), unsafe.Pointer(&pRad))
	runtime.KeepAlive(cs)
	runtime.KeepAlive(pp)
	if rc != 0 {
		return nil, fmt.Errorf("points_multi: %s", a.lastErr(ctx))
	}
	np := int(n)
	cl := &Cloud{N: np, Counts: counts,
		XYZ: cFloats(pXyz, np*3), RGB: cBytes(pRgb, np*3), Rad: cFloats(pRad, np)}
	a.freeFloats(pXyz)
	a.freeBytes(pRgb)
	a.freeFloats(pRad)
	return cl, nil
}

func b2i32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// PointsStream runs the SLIDING-WINDOW streaming pipeline over an ordered frame
// list (time-lapse video): overlapping fused windows of `chunk` frames (sharing
// `overlap`) stitched into one global cloud. budget caps total points (0 = unlimited).
// Counts is per-input-frame (frame-major) for progressive build-up. The de-ghosting
// toggles map to StreamParams: icpRefine (per-seam point-to-plane ICP, task B),
// loopClose (loop closure + Sim3 pose-graph, task C), and fuse (final voxel/surface
// fusion, task A) with cell fuseVoxel metres (<=0 => default).
func (a *capi) PointsStream(ctx uintptr, paths []string, chunk, overlap int, confPct float64, ptSize float32, budget int,
	icpRefine, loopClose, fuse, metric, temporalVoxelMesh bool,
	fuseVoxel, fuseVoxelFrac, fuseTruncMult float64) (*Cloud, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no images")
	}
	// Scene-relative TSDF fusion knobs for this bake's fuse pass (0 => C-side default).
	// Set before the stream call; consumed inside da_capi_points_stream when fuse is on.
	a.setFuseParams(ctx, fuseVoxelFrac, fuseTruncMult)
	a.setTemporalVoxelMesh(ctx, b2i32(temporalVoxelMesh))
	cs := make([][]byte, len(paths))
	pp := make([]*byte, len(paths))
	for i, s := range paths {
		b := append([]byte(s), 0)
		cs[i] = b
		pp[i] = &b[0]
	}
	var n int32
	counts := make([]int32, len(paths))
	var pXyz, pRgb, pRad uintptr
	rc := a.pointsStream(ctx, uintptr(unsafe.Pointer(&pp[0])), int32(len(paths)),
		int32(chunk), int32(overlap), confPct, ptSize, int32(budget),
		b2i32(icpRefine), b2i32(loopClose), b2i32(fuse), b2i32(metric), fuseVoxel,
		unsafe.Pointer(&n), unsafe.Pointer(&counts[0]),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pRgb), unsafe.Pointer(&pRad))
	runtime.KeepAlive(cs)
	runtime.KeepAlive(pp)
	if rc != 0 {
		return nil, fmt.Errorf("points_stream: %s", a.lastErr(ctx))
	}
	np := int(n)
	cl := &Cloud{N: np, Counts: counts,
		XYZ: cFloats(pXyz, np*3), RGB: cBytes(pRgb, np*3), Rad: cFloats(pRad, np)}
	a.freeFloats(pXyz)
	a.freeBytes(pRgb)
	a.freeFloats(pRad)

	// Per-frame capture poses (for viewer flythrough), fetched separately because
	// pointsStream is at the FFI arg-count ceiling. Best-effort: a scene without poses
	// simply has no flythrough.
	var nFrames int32
	var pFramePos, pFrameFwd uintptr
	if a.streamPoses(ctx, unsafe.Pointer(&pFramePos), unsafe.Pointer(&pFrameFwd), unsafe.Pointer(&nFrames)) == 0 {
		if nf := int(nFrames); nf > 0 && pFramePos != 0 && pFrameFwd != 0 {
			cl.FramePos = cFloats(pFramePos, nf*3)
			cl.FrameFwd = cFloats(pFrameFwd, nf*3)
		}
		a.freeFloats(pFramePos)
		a.freeFloats(pFrameFwd)
	}
	if temporalVoxelMesh {
		var pMesh, meshSize uintptr
		if a.streamVoxelMesh(ctx, unsafe.Pointer(&pMesh), unsafe.Pointer(&meshSize)) == 0 && pMesh != 0 && meshSize > 0 {
			cl.TemporalVoxelMesh = cBytes(pMesh, int(meshSize))
			a.freeBytes(pMesh)
		}
	}
	return cl, nil
}

// GS is a single-image 3D-gaussian set (world frame, OpenCV axes).
type GS struct {
	N       int
	XYZ     []float32 // 3N means
	Scale   []float32 // 3N
	Quat    []float32 // 4N wxyz
	RGB     []float32 // 3N linear [0,1]
	Opacity []float32 // N
	// Input camera (pose is canonical identity): K at the processed resolution.
	Fx, Fy, Cx, Cy float32
	W, H           int
}

// Gaussians runs the GS reconstruction on one image (DA3-GIANT only).
func (a *capi) Gaussians(ctx uintptr, path string) (*GS, error) {
	var n, w, h int32
	var pXyz, pScale, pQuat, pRgb, pOp uintptr
	var intr [9]float32
	rc := a.gaussians(ctx, path, unsafe.Pointer(&n),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pScale), unsafe.Pointer(&pQuat),
		unsafe.Pointer(&pRgb), unsafe.Pointer(&pOp),
		unsafe.Pointer(&intr[0]), unsafe.Pointer(&w), unsafe.Pointer(&h))
	if rc != 0 {
		return nil, fmt.Errorf("gaussians: %s", a.lastErr(ctx))
	}
	np := int(n)
	g := &GS{N: np,
		XYZ: cFloats(pXyz, np*3), Scale: cFloats(pScale, np*3), Quat: cFloats(pQuat, np*4),
		RGB: cFloats(pRgb, np*3), Opacity: cFloats(pOp, np),
		Fx: intr[0], Fy: intr[4], Cx: intr[2], Cy: intr[5], W: int(w), H: int(h)}
	a.freeFloats(pXyz)
	a.freeFloats(pScale)
	a.freeFloats(pQuat)
	a.freeFloats(pRgb)
	a.freeFloats(pOp)
	return g, nil
}
