// scenes.go — sample-scene listing + the async video->scene pipeline. A video is
// turned into ONE coherent cloud via sliding-window streaming (overlapping fused
// windows stitched by a weighted-Umeyama Sim3), then sliced by frame into acc_*
// "build-up" steps for the viewer.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type manifestStep struct {
	Splat  string   `json:"splat"`
	Images []string `json:"images"`
	N      int      `json:"n"`
	NPts   int      `json:"npts,omitempty"` // primitives revealed at this step (viewer reveals a prefix of one file)
	Label  string   `json:"label,omitempty"`
}

// Cam is the input camera for gaussian scenes: the pose is canonical (identity),
// so the viewer places its eye at the origin looking down +z and uses this K
// (pixel units at W×H) to reproduce the exact input view the gaussians composite to.
type Cam struct {
	Fx float32 `json:"fx"`
	Fy float32 `json:"fy"`
	Cx float32 `json:"cx"`
	Cy float32 `json:"cy"`
	W  int     `json:"w"`
	H  int     `json:"h"`
}

type sceneManifest struct {
	Model string         `json:"model"`
	Mode  string         `json:"mode"`
	Steps []manifestStep `json:"steps"`
	Cam   *Cam           `json:"cam,omitempty"`
	// Per-input-frame capture poses (GL axes, y-up), for the viewer flythrough. 3F each.
	FramePos     []float32             `json:"frame_pos,omitempty"`
	FrameFwd     []float32             `json:"frame_fwd,omitempty"`
	TemporalMesh *temporalMeshManifest `json:"temporal_mesh,omitempty"`
}

type temporalMeshManifest struct {
	File       string  `json:"file"`
	Format     string  `json:"format"`
	Faces      uint32  `json:"faces"`
	FinalFaces uint32  `json:"final_faces"`
	Frames     uint32  `json:"frames"`
	VoxelSize  float32 `json:"voxel_size"`
	Bytes      int     `json:"bytes"`
}

type sceneInfo struct {
	Name   string `json:"name"`
	Label  string `json:"label"`
	Steps  int    `json:"steps"`
	Thumb  string `json:"thumb"`
	Source string `json:"source"`
	Model  string `json:"model"`
	Mode   string `json:"mode"`
}

type sceneJob struct {
	State string `json:"state"` // running | done | error
	Total int    `json:"total"`
	Done  int    `json:"done"`
	Kept  int    `json:"kept"`
	Scene string `json:"scene,omitempty"`
	Err   string `json:"error,omitempty"`
}

var slugRe = regexp.MustCompile(`[^a-z0-9_-]+`)

func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "-")
	s = slugRe.ReplaceAllString(s, "")
	if s == "" {
		s = "scene"
	}
	if len(s) > 48 {
		s = s[:48]
	}
	return s
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// GET /api/scenes
func (s *server) handleScenes(w http.ResponseWriter, r *http.Request) {
	ents, _ := os.ReadDir(s.scenesDir)
	out := []sceneInfo{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		mf := filepath.Join(s.scenesDir, e.Name(), "manifest.json")
		b, err := os.ReadFile(mf)
		if err != nil {
			continue
		}
		var m sceneManifest
		if json.Unmarshal(b, &m) != nil || len(m.Steps) == 0 {
			continue
		}
		thumb := ""
		if len(m.Steps[0].Images) > 0 {
			thumb = "/scenes-assets/" + e.Name() + "/" + m.Steps[0].Images[0]
		}
		src := "baked"
		if _, err := os.Stat(filepath.Join(s.scenesDir, e.Name(), ".uploaded")); err == nil {
			src = "uploaded"
		}
		out = append(out, sceneInfo{
			Name: e.Name(), Label: prettify(e.Name()), Steps: len(m.Steps),
			Thumb: thumb, Source: src, Model: m.Model, Mode: m.Mode,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	writeJSON(w, map[string]any{"scenes": out})
}

func prettify(s string) string {
	return strings.Title(strings.ReplaceAll(strings.ReplaceAll(s, "-", " "), "_", " "))
}

// GET /api/scene/status/{job}
func (s *server) handleSceneStatus(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/scene/status/")
	s.jobsMu.Lock()
	j, ok := s.jobs[id]
	s.jobsMu.Unlock()
	if !ok {
		http.Error(w, "no such job", 404)
		return
	}
	writeJSON(w, j)
}

// POST /api/scene/from-video  (multipart: video, name, model, mode, max_frames, conf_pct, point_size)
func (s *server) handleSceneFromVideo(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(512 << 20); err != nil {
		http.Error(w, "bad form: "+err.Error(), 400)
		return
	}
	file, hdr, err := r.FormFile("video")
	if err != nil {
		http.Error(w, "missing video", 400)
		return
	}
	defer file.Close()
	name := slug(r.FormValue("name"))
	if name == "scene" && hdr != nil {
		name = slug(strings.TrimSuffix(hdr.Filename, filepath.Ext(hdr.Filename)))
	}
	model := r.FormValue("model")
	if model == "" {
		model = "da3-base"
	}
	mode := r.FormValue("mode")
	if mode == "" {
		mode = "points"
	}
	maxFrames := atoiDefault(r.FormValue("max_frames"), 60)
	if maxFrames < 2 {
		maxFrames = 2
	}
	if maxFrames > 600 {
		maxFrames = 600
	}
	chunkSize := atoiDefault(r.FormValue("chunk_size"), 12)
	if chunkSize < 2 {
		chunkSize = 2
	}
	// Window is uncapped (was clamped to 24). The whole window goes through one cross-view
	// attention pass, so large windows on giant/nested can exhaust VRAM and hard-crash the
	// Vulkan backend — that's on the operator now. Note the C core also clamps the window to
	// the frame count, so the effective max is maxFrames.
	overlap := atoiDefault(r.FormValue("overlap"), 3)
	if overlap < 0 {
		overlap = 0
	}
	if overlap > chunkSize-1 {
		overlap = chunkSize - 1
	}
	// fps<=0 (the default) => auto: bakeVideo picks an fps so the whole clip yields
	// ~maxFrames frames. An explicit value overrides (clamped to a sane range).
	fps := atofDefault(r.FormValue("fps"), 0)
	if fps > 0 && fps < 0.5 {
		fps = 0.5
	}
	if fps > 30 {
		fps = 30
	}
	confPct := atofDefault(r.FormValue("conf_pct"), 55)
	ptSize := float32(atofDefault(r.FormValue("point_size"), 1.2))
	// De-ghosting toggles (checkboxes): B=per-seam ICP, C=loop closure, A=fusion.
	icpRefine := boolForm(r.FormValue("icp_refine"))
	loopClose := boolForm(r.FormValue("loop_close"))
	fuse := boolForm(r.FormValue("fuse"))
	metric := boolForm(r.FormValue("metric"))              // absolute-metres output (nested model only)
	fuseVoxel := atofDefault(r.FormValue("fuse_voxel"), 0) // 0 => C-side default (0.03 m)
	// Scene-relative TSDF knobs (mode-agnostic; only used when fuse is on). 0 => C default.
	fuseVoxelFrac := atofDefault(r.FormValue("fuse_voxel_frac"), 0) // voxel edge as fraction of bbox diag (detail)
	fuseTruncMult := atofDefault(r.FormValue("fuse_trunc_mult"), 0) // truncation as multiple of voxel (merge range)
	nearBias := atofDefault(r.FormValue("near_bias"), 0)            // 0 => uniform; >0 keeps more near, fewer far
	surfels := boolForm(r.FormValue("surfels"))                     // orient points as flat surface disks
	voxelRes := atofDefault(r.FormValue("voxel_res"), 100)          // cells across the bbox diagonal (mode=voxels)
	temporalMesh := boolForm(r.FormValue("temporal_mesh"))          // precomputed exposed-face playback mesh

	// save upload
	jobDir := filepath.Join(s.workDir, "uploads", name)
	_ = os.MkdirAll(jobDir, 0o755)
	vpath := filepath.Join(jobDir, "input"+filepath.Ext(hdr.Filename))
	out, err := os.Create(vpath)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		http.Error(w, err.Error(), 500)
		return
	}
	out.Close()

	s.jobsMu.Lock()
	s.jobs[name] = &sceneJob{State: "running"}
	s.jobsMu.Unlock()

	go func() {
		s.bakeSem <- struct{}{}
		defer func() { <-s.bakeSem }()
		err := s.bakeVideo(name, vpath, jobDir, model, mode, maxFrames, chunkSize, overlap, fps, confPct, ptSize,
			icpRefine, loopClose, fuse, metric, temporalMesh, fuseVoxel, fuseVoxelFrac, fuseTruncMult, nearBias, surfels, voxelRes)
		s.jobsMu.Lock()
		j := s.jobs[name]
		if err != nil {
			j.State, j.Err = "error", err.Error()
		} else {
			j.State, j.Scene = "done", name
		}
		s.jobsMu.Unlock()
	}()

	writeJSON(w, map[string]string{"job": name, "name": name})
}

// bakeVideo: ffmpeg -> frames -> DA3 -> acc_*.splat + thumbnails + manifest.
func (s *server) bakeVideo(name, vpath, jobDir, model, mode string, maxFrames, chunkSize, overlap int, fps, confPct float64, ptSize float32,
	icpRefine, loopClose, fuse, metric, temporalMesh bool, fuseVoxel, fuseVoxelFrac, fuseTruncMult, nearBias float64, surfels bool, voxelRes float64) error {
	framesDir := filepath.Join(jobDir, "frames")
	_ = os.RemoveAll(framesDir)
	_ = os.MkdirAll(framesDir, 0o755)
	// Auto fps: pick a rate so the WHOLE clip yields ~maxFrames frames. A fixed low
	// fps starves short clips (a 0.5s clip at 6fps gives 3 frames -> sparse cloud),
	// so when no explicit fps is set, derive it from the probed duration.
	if fps <= 0 {
		fps = 6 // fallback if the probe fails
		if dur := probeDurationSec(vpath); dur > 0.05 {
			fps = float64(maxFrames) / dur
			if fps < 1 {
				fps = 1
			}
			if fps > 30 {
				fps = 30
			}
		}
	}
	// Bounded extraction (fps cap + downscale), then even stride to maxFrames.
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", vpath,
		"-vf", fmt.Sprintf("fps=%g,scale=640:-2", fps), filepath.Join(framesDir, "all%05d.jpg"))
	if b, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg: %v: %s", err, string(b))
	}
	all, _ := filepath.Glob(filepath.Join(framesDir, "all*.jpg"))
	sort.Strings(all)
	if len(all) < 2 {
		return fmt.Errorf("video produced %d frames (need >=2)", len(all))
	}
	stride := (len(all) + maxFrames - 1) / maxFrames
	if stride < 1 {
		stride = 1
	}
	var sel []string
	for i := 0; i < len(all) && len(sel) < maxFrames; i += stride {
		sel = append(sel, all[i])
	}

	sceneDir := filepath.Join(s.scenesDir, name)
	_ = os.RemoveAll(sceneDir)
	_ = os.MkdirAll(sceneDir, 0o755)
	_ = os.WriteFile(filepath.Join(sceneDir, ".uploaded"), []byte("1"), 0o644)

	s.setJob(name, func(j *sceneJob) { j.Total = len(sel) })

	mi := findModel(model)
	if mi == nil {
		return fmt.Errorf("unknown model %q", model)
	}

	// Gaussian mode: single representative (middle) frame -> GS splat.
	if mode == "gaussians" {
		if !mi.Gaussians {
			return fmt.Errorf("model %q has no gaussian head (use da3-giant)", model)
		}
		mid := sel[len(sel)/2]
		thumb := "view_0.jpg"
		_ = copyScaled(mid, filepath.Join(sceneDir, thumb), 360)
		// GS head is fixed at a 224x224 (16x16 patch) input.
		g224 := filepath.Join(jobDir, "g224.jpg")
		if e := squareResize(mid, g224, 224); e != nil {
			return e
		}
		var splat []byte
		var cam *Cam
		err := s.infer(func() error {
			return s.reg.WithModel(model, func(ctx uintptr, _ *ModelInfo) error {
				g, e := s.api.Gaussians(ctx, g224)
				if e != nil {
					return e
				}
				splat = gaussiansToSplat(g, s.maxSplats)
				cam = &Cam{Fx: g.Fx, Fy: g.Fy, Cx: g.Cx, Cy: g.Cy, W: g.W, H: g.H}
				return nil
			})
		})
		if err != nil {
			return err
		}
		if e := os.WriteFile(filepath.Join(sceneDir, "acc_full.splat"), splat, 0o644); e != nil {
			return e
		}
		s.setJob(name, func(j *sceneJob) { j.Done, j.Kept = len(sel), 1 })
		return writeManifest(sceneDir, sceneManifest{Model: model, Mode: mode, Cam: cam, Steps: []manifestStep{
			{Splat: "acc_full.splat", Images: []string{thumb}, N: 1, Label: "gaussians · single frame"}}})
	}

	// Point-cloud mode: sliding-window streaming over all selected frames.
	if !mi.Pose {
		return fmt.Errorf("model %q has no camera pose; cannot fuse frames", model)
	}
	// thumbnails first (so progress feels live)
	thumbs := make([]string, len(sel))
	for i, f := range sel {
		thumbs[i] = fmt.Sprintf("view_%d.jpg", i)
		_ = copyScaled(f, filepath.Join(sceneDir, thumbs[i]), 360)
	}

	var cloud *Cloud
	err := s.infer(func() error {
		return s.reg.WithModel(model, func(ctx uintptr, _ *ModelInfo) error {
			// Metric output only makes sense for a nested (metric-capable) model; the
			// C side would reject metric on a plain pose model, so gate it here.
			c, e := s.api.PointsStream(ctx, sel, chunkSize, overlap, confPct, ptSize, s.maxSplats,
				icpRefine, loopClose, fuse, metric && mi.Metric,
				temporalMesh && mode == "voxels" && fuse, fuseVoxel, fuseVoxelFrac, fuseTruncMult)
			if e != nil {
				return e
			}
			cloud = c
			return nil
		})
	})
	if err != nil {
		return err
	}
	s.setJob(name, func(j *sceneJob) { j.Done = len(sel); j.Kept = len(sel) })

	// Depth-weighted density: keep more near-camera points, fewer far ones. Applied
	// once here so the build-up below (which slices Counts prefixes) sees the thinned
	// cloud with rebuilt per-view counts. nearBias<=0 leaves the cloud untouched.
	if nearBias > 0 && mode != "voxels" {
		before := cloud.N
		cloud = thinByDepth(cloud, nearBias)
		log.Printf("scene %s: near-bias %.2f thinned %d -> %d points", name, nearBias, before, cloud.N)
	}

	// Voxel (Minecraft) mode snaps points onto a grid whose cell = bbox-diagonal /
	// voxelRes; each build-up step voxelises its prefix at that fixed cell so blocks
	// stay put as the scene fills in. Surfels (oriented disks) only apply to the smooth
	// point/metric output — pointless under voxelisation, so we skip the normal pass.
	asVoxels := mode == "voxels"
	var cell float64
	if asVoxels {
		if fuse {
			// Follow the fusion grid: the cloud is already one point per fusion voxel,
			// so render one cube per fusion voxel instead of re-blocking on a second
			// grid. Use the same scene-relative fraction the C-side TSDF used (see
			// da_capi/tsdf fuse_voxel_frac; 0 => its 0.4%-of-bbox-diagonal default).
			frac := fuseVoxelFrac
			if frac <= 0 {
				frac = 0.004
			}
			cell = frac * cloudBBoxDiag(cloud, cloud.N)
			log.Printf("scene %s: voxel mode follows fusion cell=%.5g (frac=%.4g)", name, cell, frac)
		} else {
			if voxelRes < 1 {
				voxelRes = 100
			}
			cell = cloudBBoxDiag(cloud, cloud.N) / voxelRes
			log.Printf("scene %s: voxel mode res=%.0f cell=%.5g", name, voxelRes, cell)
		}
	} else {
		// Additive reveal: overlapping frames re-observe the same surface, so revealing a
		// new frame otherwise re-blends already-shown regions. Collapse coincident points
		// (first frame to fill a ~point-spacing cell wins) so each surface patch is owned by
		// one frame. Cell = median point radius (≈ local sample spacing), scene-relative.
		if medRad := float64(pctile(cloud.Rad, 50)); medRad > 0 {
			dropped := dedupFirstFrame(cloud, medRad)
			log.Printf("scene %s: overlap dedup cell=%.5g dropped %d -> %d pts", name, medRad, dropped, cloud.N)
		}
		if surfels {
			estimateNormals(cloud)
		}
	}

	// Build-up: emit ONE full primitive file, ordered so the viewer can reveal a growing
	// PREFIX of it (no reloading). Points are ordered frame-major then high-confidence
	// (small-radius) first within each frame; cubes are ordered by first-touching frame.
	// Each step then just carries how many primitives are revealed at that point (NPts).
	if !asVoxels {
		orderWithinFramesByRadius(cloud)
	}
	var fullBlob []byte
	var cubeFirst []int // per-cube first frame (voxels), for the reveal boundary
	if asVoxels {
		fullBlob, cubeFirst = cloudToCubesOrdered(cloud, cell)
	} else {
		fullBlob = pointsToSplat(cloud, cloud.N, 1.0)
	}
	const fullName = "acc_full.splat"
	if e := os.WriteFile(filepath.Join(sceneDir, fullName), fullBlob, 0o644); e != nil {
		return e
	}
	var temporalManifest *temporalMeshManifest
	if asVoxels && len(cloud.TemporalVoxelMesh) > 0 {
		const meshName = "temporal_faces.dtvm"
		if e := os.WriteFile(filepath.Join(sceneDir, meshName), cloud.TemporalVoxelMesh, 0o644); e != nil {
			return e
		}
		if info, ok := parseTemporalMeshManifest(meshName, cloud.TemporalVoxelMesh); ok {
			temporalManifest = info
			log.Printf("scene %s: temporal mesh %d faces (%d final), %.2f MiB", name,
				info.Faces, info.FinalFaces, float64(info.Bytes)/(1024*1024))
		}
	}
	totalPrims := len(fullBlob) / splatRow

	steps := []manifestStep{}
	prefix := 0
	nv := cloud.N0(len(sel))
	bstep := (nv + 29) / 30
	if bstep < 1 {
		bstep = 1
	}
	for k := 1; k <= nv; k++ {
		prefix += int(cloud.Counts[k-1])
		if k < 2 {
			continue
		}
		if k%bstep != 0 && k != nv {
			continue
		}
		npts := prefix // points revealed = cumulative through frame k
		if asVoxels {
			npts = countLess(cubeFirst, k) // cubes whose first frame < k
		}
		if npts > totalPrims {
			npts = totalPrims
		}
		label := ""
		if k == nv {
			npts = totalPrims
			if asVoxels {
				label = fmt.Sprintf("full %d-view voxels · %d cubes", k, totalPrims)
			} else {
				label = fmt.Sprintf("full %d-view cloud · %d pts", k, totalPrims)
			}
		}
		steps = append(steps, manifestStep{Splat: fullName, Images: thumbs[:k], N: k, NPts: npts, Label: label})
	}

	// Per-frame capture poses for the viewer flythrough: flip OpenCV(y-down,z-fwd) ->
	// GL(y-up) to match the rendered cloud (the splat encoder applies the same flip to
	// point positions). Failed-window frames come back as zero; forward/back-fill them
	// from the nearest valid pose so the camera path stays continuous.
	fpos, ffwd := frameGLPoses(cloud, len(sel))
	return writeManifest(sceneDir, sceneManifest{Model: model, Mode: mode, Steps: steps,
		FramePos: fpos, FrameFwd: ffwd, TemporalMesh: temporalManifest})
}

func parseTemporalMeshManifest(name string, b []byte) (*temporalMeshManifest, bool) {
	if len(b) < 32 || string(b[:4]) != "DTVM" || binary.LittleEndian.Uint16(b[4:]) != 1 ||
		binary.LittleEndian.Uint16(b[6:]) != 32 || binary.LittleEndian.Uint32(b[28:]) != 24 {
		return nil, false
	}
	faces := binary.LittleEndian.Uint32(b[16:])
	if uint64(32)+uint64(faces)*24 != uint64(len(b)) {
		return nil, false
	}
	return &temporalMeshManifest{
		File: name, Format: "dtvm-1", Frames: binary.LittleEndian.Uint32(b[12:]),
		Faces: faces, FinalFaces: binary.LittleEndian.Uint32(b[20:]),
		VoxelSize: math.Float32frombits(binary.LittleEndian.Uint32(b[24:])), Bytes: len(b),
	}, true
}

// frameGLPoses returns per-frame camera centre + forward in GL axes (length 3F each),
// or (nil,nil) if the cloud carries no poses. Zero (failed) frames are gap-filled.
func frameGLPoses(c *Cloud, nSel int) (pos, fwd []float32) {
	if len(c.FramePos) != 3*nSel || len(c.FrameFwd) != 3*nSel || nSel == 0 {
		return nil, nil
	}
	pos = make([]float32, 3*nSel)
	fwd = make([]float32, 3*nSel)
	valid := make([]bool, nSel)
	for i := 0; i < nSel; i++ {
		pos[3*i], pos[3*i+1], pos[3*i+2] = c.FramePos[3*i], -c.FramePos[3*i+1], -c.FramePos[3*i+2]
		fwd[3*i], fwd[3*i+1], fwd[3*i+2] = c.FrameFwd[3*i], -c.FrameFwd[3*i+1], -c.FrameFwd[3*i+2]
		fx, fy, fz := fwd[3*i], fwd[3*i+1], fwd[3*i+2]
		valid[i] = fx*fx+fy*fy+fz*fz > 0.25 // a real unit forward (failed frames are ~0)
	}
	copyPose := func(dst, src int) {
		pos[3*dst], pos[3*dst+1], pos[3*dst+2] = pos[3*src], pos[3*src+1], pos[3*src+2]
		fwd[3*dst], fwd[3*dst+1], fwd[3*dst+2] = fwd[3*src], fwd[3*src+1], fwd[3*src+2]
	}
	first := -1
	for i := 0; i < nSel; i++ {
		if valid[i] {
			first = i
			break
		}
	}
	if first < 0 {
		return nil, nil // no usable poses (all windows failed)
	}
	last := first
	for i := first + 1; i < nSel; i++ { // forward-fill interior/trailing gaps
		if valid[i] {
			last = i
		} else {
			copyPose(i, last)
		}
	}
	for i := 0; i < first; i++ { // back-fill the leading gap
		copyPose(i, first)
	}
	return pos, fwd
}

// N0 guards Counts length vs the number of selected views.
func (c *Cloud) N0(nViews int) int {
	if len(c.Counts) < nViews {
		return len(c.Counts)
	}
	return nViews
}

func writeManifest(dir string, m sceneManifest) error {
	b, _ := json.MarshalIndent(m, "", "  ")
	return os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0o644)
}

func copyScaled(src, dst string, px int) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", src,
		"-vf", fmt.Sprintf("scale=%d:-2", px), dst)
	return cmd.Run()
}

// squareResize center-crops to a px*px square (the GS head's fixed input size).
func squareResize(src, dst string, px int) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", src,
		"-vf", fmt.Sprintf("scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d", px, px, px, px), dst)
	return cmd.Run()
}

func (s *server) setJob(name string, fn func(*sceneJob)) {
	s.jobsMu.Lock()
	if j := s.jobs[name]; j != nil {
		fn(j)
	}
	s.jobsMu.Unlock()
}

func atoiDefault(s string, d int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return v
	}
	return d
}

func atofDefault(s string, d float64) float64 {
	if v, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
		return v
	}
	return d
}

// boolForm parses a checkbox/toggle form value ("1"/"true"/"on"/"yes" => true).
func boolForm(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// probeDurationSec returns the clip length in seconds via ffprobe, or 0 on failure.
func probeDurationSec(path string) float64 {
	out, err := exec.Command("ffprobe", "-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1", path).Output()
	if err != nil {
		return 0
	}
	d, _ := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	return d
}
