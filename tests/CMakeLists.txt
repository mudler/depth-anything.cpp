function(da_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE depthanything)
  target_include_directories(${name} PRIVATE
      ${DA_ROOT}/src ${DA_ROOT}/include
      ${DA_ROOT}/tests ${DA_ROOT}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      WORKING_DIRECTORY ${DA_ROOT}
      ENVIRONMENT "DA_TEST_GGUF=${DA_ROOT}/models/depth-anything-base-f32.gguf;DA_TEST_BASELINE=${DA_ROOT}/dumps/reference.gguf;DA_TEST_BASELINE_MV=${DA_ROOT}/dumps/reference_mv.gguf;DA_TEST_BASELINE_MV4=${DA_ROOT}/dumps/reference_mv4.gguf;DA_TEST_GGUF_GIANT=${DA_ROOT}/models/depth-anything-giant-f32.gguf;DA_TEST_BASELINE_GIANT=${DA_ROOT}/dumps/reference_giant.gguf;DA_TEST_GGUF_METRIC=${DA_ROOT}/models/depth-anything-nested-metric.gguf;DA_TEST_GGUF_ANYVIEW=${DA_ROOT}/models/depth-anything-nested-anyview.gguf;DA_TEST_BASELINE_NESTED=${DA_ROOT}/dumps/reference_nested.gguf;DA_TEST_BASELINE_PREPROC=${DA_ROOT}/dumps/reference_preproc_real.gguf;DA_TEST_PREPROC_PNG=${DA_ROOT}/dumps/preproc_real_input.png;DA_TEST_BASELINE_NATIVE=${DA_ROOT}/dumps/reference_native.gguf;DA_TEST_NATIVE_PNG=${DA_ROOT}/dumps/native_input.png;DA_TEST_GGUF_MONO=${DA_ROOT}/models/depth-anything-mono-large-f32.gguf;DA_TEST_BASELINE_MONO=${DA_ROOT}/dumps/reference_mono.gguf;DA_TEST_MONO_PNG=${DA_ROOT}/dumps/mono_input.png;DA_TEST_GGUF_AUX=${DA_ROOT}/models/depth-anything-base-aux-f32.gguf;DA_TEST_BASELINE_RAYS=${DA_ROOT}/dumps/reference_rays.gguf;DA_TEST_BASELINE_RAY_POSE=${DA_ROOT}/dumps/reference_ray_pose.gguf;DA_TEST_GGUF_DA2=${DA_ROOT}/models/depth-anything2-large-f32.gguf;DA_TEST_BASELINE_DA2=${DA_ROOT}/dumps/reference_da2.gguf")
endfunction()

da_add_test(test_model_loader)
da_add_test(test_backend)
da_add_test(test_ggml_extend)
da_add_test(test_capi)
da_add_test(test_capi_dense)
da_add_test(test_capi_da2)
da_add_test(test_preprocess)
da_add_test(test_preprocess_real)
da_add_test(test_rope2d)
da_add_test(test_uv_posembed)
da_add_test(test_backbone_prepare)
da_add_test(test_backbone)
da_add_test(test_backbone_mv)
da_add_test(test_backbone_mv4)
da_add_test(test_backbone_giant)
da_add_test(test_backbone_metric)
da_add_test(test_backbone_da2)
da_add_test(test_engine_da2)
da_add_test(test_metric_head)
da_add_test(test_nested_align)
da_add_test(test_engine_metric)
da_add_test(test_giant_depth_pose)
da_add_test(test_gs_head)
da_add_test(test_gs_adapter)
da_add_test(test_winograd)
da_add_test(test_linalg)
da_add_test(test_sim3)
da_add_test(test_stream_seam)
da_add_test(test_geom_metrics)
da_add_test(test_synth_scene)
da_add_test(test_stream_synth)
da_add_test(test_fuse)
da_add_test(test_tsdf)
da_add_test(test_voxel_mesh)
da_add_test(test_icp)
da_add_test(test_posegraph)
da_add_test(test_stream_loop)
da_add_test(test_ray_pose)
da_add_test(test_dpt_blocks)
da_add_test(test_dpt_head)
da_add_test(test_head_rays)
da_add_test(test_engine_depth)
da_add_test(test_engine_depth_native)
da_add_test(test_engine_mono)
da_add_test(test_fused_depth)
da_add_test(test_reconstruct)
da_add_test(test_cam_pose)
da_add_test(test_engine_pose)
da_add_test(test_engine_mv)
da_add_test(test_quantize)
da_add_test(test_quantize_accuracy)

# Standalone harness for scripts/oracle_fuse.py (takes CLI args; not a ctest).
add_executable(fuse_dump fuse_dump.cpp)
target_link_libraries(fuse_dump PRIVATE depthanything)
target_include_directories(fuse_dump PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Standalone harness for scripts/oracle_icp.py (takes CLI args; not a ctest).
add_executable(icp_dump icp_dump.cpp)
target_link_libraries(icp_dump PRIVATE depthanything)
target_include_directories(icp_dump PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Standalone harness for scripts/oracle_posegraph.py (takes CLI args; not a ctest).
add_executable(posegraph_dump posegraph_dump.cpp)
target_link_libraries(posegraph_dump PRIVATE depthanything)
target_include_directories(posegraph_dump PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Real-clip streaming de-ghosting regression (takes a model + frame dir; not a ctest).
add_executable(stream_regress stream_regress.cpp)
target_link_libraries(stream_regress PRIVATE depthanything)
target_include_directories(stream_regress PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Metric-streaming prototype: scale-consistency of nested depth_metric across
# consecutive frames (takes anyview+metric gguf + frame dir; not a ctest).
add_executable(metric_probe metric_probe.cpp)
target_link_libraries(metric_probe PRIVATE depthanything)
target_include_directories(metric_probe PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Metric-streaming end-to-end check (relative vs metric cloud scale; not a ctest).
add_executable(metric_stream_check metric_stream_check.cpp)
target_link_libraries(metric_stream_check PRIVATE depthanything)
target_include_directories(metric_stream_check PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Max metric depth predicted across clips (takes anyview+metric gguf + dirs; not a ctest).
add_executable(metric_maxdepth metric_maxdepth.cpp)
target_link_libraries(metric_maxdepth PRIVATE depthanything)
target_include_directories(metric_maxdepth PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Standalone harness for scripts/parity_glb.py (takes CLI args; not a ctest).
add_executable(glb_parity_dump glb_parity_dump.cpp)
target_link_libraries(glb_parity_dump PRIVATE depthanything)
target_include_directories(glb_parity_dump PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)

# Standalone harness for scripts/parity_colmap.py (takes CLI args; not a ctest).
add_executable(colmap_parity_dump colmap_parity_dump.cpp)
target_link_libraries(colmap_parity_dump PRIVATE depthanything)
target_include_directories(colmap_parity_dump PRIVATE
    ${DA_ROOT}/src ${DA_ROOT}/include ${DA_ROOT}/third_party)
