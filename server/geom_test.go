package main

import (
	"encoding/binary"
	"math"
	"testing"
)

// A flat grid of points on the z=0 plane must yield normals along ±z, and the
// quaternion built from that normal must rotate local +z back onto the plane normal.
func TestEstimateNormalsPlane(t *testing.T) {
	c := &Cloud{}
	const G = 20
	step := float32(0.05)
	for i := 0; i < G; i++ {
		for j := 0; j < G; j++ {
			c.XYZ = append(c.XYZ, float32(i)*step, float32(j)*step, 0)
			c.RGB = append(c.RGB, 200, 200, 200)
			c.Rad = append(c.Rad, step) // spacing == radius
		}
	}
	c.N = G * G
	c.Counts = []int32{int32(c.N)}
	estimateNormals(c)
	if len(c.Normals) != 3*c.N {
		t.Fatalf("normals not filled: %d", len(c.Normals))
	}
	// interior point (well away from the border) should have a clean ±z normal
	mid := (G/2)*G + G/2
	nz := math.Abs(float64(c.Normals[3*mid+2]))
	nxy := math.Hypot(float64(c.Normals[3*mid]), float64(c.Normals[3*mid+1]))
	if nz < 0.98 || nxy > 0.1 {
		t.Fatalf("interior normal not along z: n=(%.3f,%.3f,%.3f)", c.Normals[3*mid], c.Normals[3*mid+1], c.Normals[3*mid+2])
	}
	// quaternion's 3rd basis axis (local z) must equal the normal (up to sign)
	q := quatFromNormal(0, 0, 1)
	w, x, y, z := q[0], q[1], q[2], q[3]
	// 3rd column of R(q)
	cz := [3]float32{2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2*(x*x+y*y)}
	if math.Abs(float64(cz[2])-1) > 1e-5 || math.Hypot(float64(cz[0]), float64(cz[1])) > 1e-5 {
		t.Fatalf("quatFromNormal(z) local-z axis wrong: (%.4f,%.4f,%.4f)", cz[0], cz[1], cz[2])
	}
}

// quatFromNormal must return a unit quaternion whose local-z axis matches an arbitrary
// input normal (up to sign is not needed here — it should match exactly in direction).
func TestQuatFromNormalArbitrary(t *testing.T) {
	n := [3]float64{0.3, -0.6, 0.74}
	l := math.Sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
	nx, ny, nz := float32(n[0]/l), float32(n[1]/l), float32(n[2]/l)
	q := quatFromNormal(nx, ny, nz)
	qn := math.Sqrt(float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]))
	if math.Abs(qn-1) > 1e-4 {
		t.Fatalf("quat not unit: |q|=%.5f", qn)
	}
	w, x, y, z := q[0], q[1], q[2], q[3]
	cz := [3]float32{2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2*(x*x+y*y)}
	d := float64(cz[0]*nx + cz[1]*ny + cz[2]*nz)
	if d < 0.999 {
		t.Fatalf("local-z axis misaligned with normal: dot=%.5f", d)
	}
}

// orderWithinFramesByRadius must sort each frame block ascending by radius while
// preserving frame boundaries (Counts) and keeping XYZ/Rad in lockstep.
func TestOrderWithinFrames(t *testing.T) {
	c := &Cloud{Counts: []int32{3, 2}}
	// frame 0: radii 3,1,2 ; frame 1: radii 5,4  (deliberately unsorted)
	rads := []float32{3, 1, 2, 5, 4}
	for i, r := range rads {
		c.XYZ = append(c.XYZ, float32(i), 0, 0) // tag x = original index
		c.RGB = append(c.RGB, byte(i), 0, 0)
		c.Rad = append(c.Rad, r)
	}
	c.N = len(rads)
	orderWithinFramesByRadius(c)
	// frame 0 block [0,3) ascending: 1,2,3 ; frame 1 block [3,5): 4,5
	want := []float32{1, 2, 3, 4, 5}
	for i := range want {
		if c.Rad[i] != want[i] {
			t.Fatalf("radii not per-frame sorted: got %v want %v", c.Rad, want)
		}
	}
	// XYZ.x (original index tag) must have moved with the radius
	if c.XYZ[0] != 1 || c.XYZ[3] != 2 || c.XYZ[6] != 0 { // idx of r=1 is 1, r=2 is 2, r=3 is 0
		t.Fatalf("XYZ not permuted with radius: %v", c.XYZ)
	}
	// frame boundary intact: block 1 stays {4,5}, not mixed with frame 0
	if c.Rad[3] < c.Rad[2] {
		t.Fatalf("frame boundary violated: %v", c.Rad)
	}
}

// dedupFirstFrame must keep a cell for the FIRST frame that fills it and drop later
// frames' coincident re-observations, while keeping genuinely new cells and rebuilding
// Counts. This is what makes the reveal purely additive.
func TestDedupFirstFrame(t *testing.T) {
	c := &Cloud{Counts: []int32{2, 3}}
	// frame 0: two points in distinct cells (0,*,*) and (5,*,*)
	// frame 1: one coincident with frame0's first cell (should drop), two genuinely new
	pts := [][3]float32{
		{0.1, 0, 0}, {5.1, 0, 0}, // frame 0 -> cells x=0, x=5
		{0.2, 0, 0},               // frame 1 -> cell x=0 (dup of frame0) -> DROP
		{9.1, 0, 0}, {12.1, 0, 0}, // frame 1 -> new cells x=9, x=12 -> KEEP
	}
	for _, p := range pts {
		c.XYZ = append(c.XYZ, p[0], p[1], p[2])
		c.RGB = append(c.RGB, 1, 2, 3)
		c.Rad = append(c.Rad, 0.1)
	}
	c.N = len(pts)
	dropped := dedupFirstFrame(c, 1.0)
	if dropped != 1 {
		t.Fatalf("expected 1 coincident point dropped, got %d", dropped)
	}
	if c.N != 4 {
		t.Fatalf("expected 4 points kept, got %d", c.N)
	}
	// Counts rebuilt: frame0 keeps both, frame1 keeps 2 of its 3 (the dup dropped)
	if c.Counts[0] != 2 || c.Counts[1] != 2 {
		t.Fatalf("Counts not rebuilt correctly: %v (want [2 2])", c.Counts)
	}
	// kept points remain a frame-major subsequence (frame0's cells come before frame1's new ones)
	if c.XYZ[0] != 0.1 || c.XYZ[3] != 5.1 || c.XYZ[6] != 9.1 || c.XYZ[9] != 12.1 {
		t.Fatalf("frame order not preserved: %v", c.XYZ)
	}
}

// countLess is the reveal-boundary lookup: how many cubes have first-frame < k.
func TestCountLess(t *testing.T) {
	s := []int{0, 0, 1, 1, 1, 3, 4}
	for _, tc := range []struct{ k, want int }{{0, 0}, {1, 2}, {2, 5}, {3, 5}, {4, 6}, {99, 7}} {
		if g := countLess(s, tc.k); g != tc.want {
			t.Fatalf("countLess(%d)=%d want %d", tc.k, g, tc.want)
		}
	}
}

// Voxelising a cloud that fills a known box must produce one cube per occupied cell,
// each a 32-byte record with the expected half-extent in its scale fields.
func TestCloudToCubes(t *testing.T) {
	c := &Cloud{}
	// 8 points, 2 per cell across 4 distinct unit cells at cell=1.0
	pts := [][3]float32{
		{0.1, 0.1, 0.1}, {0.9, 0.9, 0.9}, // cell (0,0,0)
		{1.2, 0.1, 0.1}, {1.8, 0.5, 0.5}, // cell (1,0,0)
		{0.1, 1.3, 0.1}, // cell (0,1,0)
		{0.1, 0.1, 2.4}, // cell (0,0,2)
	}
	for _, p := range pts {
		c.XYZ = append(c.XYZ, p[0], p[1], p[2])
		c.RGB = append(c.RGB, 100, 150, 200)
		c.Rad = append(c.Rad, 0.1)
	}
	c.N = len(pts)
	blob := cloudToCubes(c, c.N, 1.0)
	nCubes := len(blob) / splatRow
	if nCubes != 4 {
		t.Fatalf("expected 4 occupied cells, got %d", nCubes)
	}
	// each record's scale fields = half-extent = 0.5*cell*voxelFill
	wantHalf := float32(0.5 * 1.0 * voxelFill)
	sx := math.Float32frombits(binary.LittleEndian.Uint32(blob[12:]))
	if math.Abs(float64(sx-wantHalf)) > 1e-6 {
		t.Fatalf("cube half-extent = %.4f, want %.4f", sx, wantHalf)
	}
	// empty cell size guard
	if cloudToCubes(c, c.N, 0) != nil {
		t.Fatalf("cell<=0 must yield nil")
	}
}

func TestParseTemporalMeshManifest(t *testing.T) {
	b := make([]byte, 32+2*24)
	copy(b, "DTVM")
	binary.LittleEndian.PutUint16(b[4:], 1)
	binary.LittleEndian.PutUint16(b[6:], 32)
	binary.LittleEndian.PutUint32(b[12:], 9)
	binary.LittleEndian.PutUint32(b[16:], 2)
	binary.LittleEndian.PutUint32(b[20:], 1)
	binary.LittleEndian.PutUint32(b[24:], math.Float32bits(0.125))
	binary.LittleEndian.PutUint32(b[28:], 24)
	m, ok := parseTemporalMeshManifest("surface.dtvm", b)
	if !ok {
		t.Fatal("valid DTVM header rejected")
	}
	if m.File != "surface.dtvm" || m.Format != "dtvm-1" || m.Frames != 9 ||
		m.Faces != 2 || m.FinalFaces != 1 || m.VoxelSize != 0.125 || m.Bytes != len(b) {
		t.Fatalf("unexpected manifest: %+v", m)
	}
	if _, ok := parseTemporalMeshManifest("bad.dtvm", b[:len(b)-1]); ok {
		t.Fatal("truncated DTVM accepted")
	}
}
