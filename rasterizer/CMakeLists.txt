cmake_minimum_required(VERSION 3.15...4.2)
project(GAUSSIAN_RENDERER LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(Torch REQUIRED)

pybind11_add_module(GAUSSIAN_RENDERER test.cpp)
target_link_libraries(GAUSSIAN_RENDERER PRIVATE "${TORCH_LIBRARIES}")
install(TARGETS GAUSSIAN_RENDERER DESTINATION ..)
