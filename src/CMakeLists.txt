cmake_minimum_required(VERSION 3.5)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR}
					${OpenCV_INCLUDE_DIRS}
					${PROJECT_SOURCE_DIR}/inc )

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/lib)

message(STATUS "binary dir: " ${PROJECT_BINARY_DIR})
message(STATUS "source dir: " ${PROJECT_SOURCE_DIR})


set(SRC_LIST   gn.cpp)
add_executable(gn   ${SRC_LIST} )
add_dependencies(gn   ${OpenCV_LIBS} ${EIGEN3_LIBRARIES}  )
target_link_libraries(gn  ${OpenCV_LIBS} ${EIGEN3_LIBRARIES})


add_executable(lm   lm.cpp )
add_dependencies(lm   ${OpenCV_LIBS} ${EIGEN3_LIBRARIES}  )
target_link_libraries(lm  ${OpenCV_LIBS} ${EIGEN3_LIBRARIES})

add_executable(dl   dl.cpp )
add_dependencies(dl   ${OpenCV_LIBS} ${EIGEN3_LIBRARIES}  )
target_link_libraries(dl  ${OpenCV_LIBS} ${EIGEN3_LIBRARIES})



