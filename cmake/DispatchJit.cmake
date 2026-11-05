# HT dispatch uses DeepJIT; the remaining device kernels retain their AOT build.
find_package(Torch REQUIRED)
if(NOT IS_DIRECTORY "${TORCH_NPU_PATH}/include")
    message(FATAL_ERROR "Dispatch JIT requires TORCH_NPU_PATH pointing to torch_npu")
endif()
find_library(DEEPEP_TORCH_NPU NAMES torch_npu PATHS "${TORCH_NPU_PATH}/lib" NO_DEFAULT_PATH REQUIRED)
set_target_properties(_C PROPERTIES CXX_STANDARD 20)
target_compile_definitions(_C PRIVATE DJ_DISABLE_GIL=1)
separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(_C PRIVATE ${_torch_flags} -Wno-missing-field-initializers
    -Wno-attributes -Wno-deprecated-declarations -Wno-unused-function)
target_include_directories(_C SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include" "${TORCH_NPU_PATH}/include/third_party/acl/inc")
target_include_directories(_C PRIVATE "${CMAKE_SOURCE_DIR}/third-party/deep_jit/include")
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} "${DEEPEP_TORCH_NPU}")
# Preserve the relative includes used by the templated dispatch implementation.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/csrc/kernels" DESTINATION jit
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(FILES "${CMAKE_SOURCE_DIR}/third-party/deep_jit/LICENSE" DESTINATION jit RENAME DeepJIT.LICENSE)
