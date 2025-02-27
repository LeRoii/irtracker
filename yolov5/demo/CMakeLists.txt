message("~~~~~~~~~~~~~~~~~~~building yolo_demo~~~~~~~~~~~~~~~~~~~~~~~~~~")
set(yolo_demo_source ${CMAKE_CURRENT_SOURCE_DIR}/yolo_demo.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/tracker.cpp)

find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)

# 指定生成目标
add_executable(yolo_demo ${yolo_demo_source})

target_include_directories(yolo_demo PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(yolo_demo 
yolo_sdk  
${OpenCV_LIBS}
yaml-cpp
spdlog::spdlog
 -ltracker)


