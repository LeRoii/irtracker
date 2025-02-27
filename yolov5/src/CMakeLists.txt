


aux_source_directory(. DIR_SRCS)

# 指定生成目标
add_library(yolo_sdk SHARED ${DIR_SRCS})




target_include_directories(yolo_sdk PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(yolo_sdk ascendcl stdc++  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs dl rt)
