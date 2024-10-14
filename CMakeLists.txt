cmake_minimum_required(VERSION 3.10)
project(Task04)

set(Torch_DIR /home/hence/下载/libtorch/share/cmake/Torch)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    )

aux_source_directory(./src ALL_SRCS)

add_executable(Task04 main.cpp ${ALL_SRCS})

target_link_libraries(Task04 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}

)

