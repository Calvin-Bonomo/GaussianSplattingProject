cmake_minimum_required(VERSION 3.15...4.2)
project(gs_rasterizer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PY_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PY_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PY_PATH}/lib" REQUIRED)

add_library(gs_rasterizer MODULE gs_rasterizer.cpp util.cu wrapper.cu rasterize.cu)

set_target_properties(gs_rasterizer PROPERTIES
    TORCH_CUDA_ARCH_LIST "75;86;89"
    CXX_STANDARD 17
    CUDA_STANDARD 17
    PREFIX "")

target_link_libraries(gs_rasterizer PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" Python3::Module)

target_include_directories(gs_rasterizer PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

target_compile_options(gs_rasterizer PRIVATE ${TORCH_CXX_FLAGS})
