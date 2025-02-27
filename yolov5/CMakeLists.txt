# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)


# project information
project(detect)


# set ascend path
set(ToolChainRoot "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/") #   /usr/local/Ascend/ascend-toolkit/latest
set(ToolChainLib ${ToolChainRoot}/lib64)
set(TooChainInclude ${ToolChainRoot}/include)

# set compiler
add_compile_options(-std=c++11)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

find_package(OpenCV)

# set ascend path
set(OpencvRoot "/home/HwHiAiUser/ascend_ddk/arm") #   
set(OpencvLib ${OpencvRoot}/lib)
set(OpencvInclude /usr/include/opencv4/)


# add global include
include_directories(${TooChainInclude} /home/HwHiAiUser/Ascend/ascend-toolkit/20.2.rc1/arm64-linux/acllib/include ${OpencvInclude})
link_directories(/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/ ${OpenCV_LIBS})

# set output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

add_subdirectory("./src")

add_subdirectory("./demo")
