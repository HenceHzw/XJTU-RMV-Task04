cmake_minimum_required(VERSION 3.10)
project(Task04)

set(Torch_DIR /home/hence/下载/libtorch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Qt5")  # Qt 安装路径


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Multimedia)

link_directories(/opt/MVS/lib/64)

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
    Qt5::Core 
    Qt5::Widgets 
    Qt5::Multimedia
    /opt/MVS/lib/64/libMvCameraControl.so
    
)

