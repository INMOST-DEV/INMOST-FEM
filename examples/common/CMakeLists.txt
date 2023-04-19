
add_subdirectory(primitives)
add_library(example_com example_common.cpp)
set_target_properties(example_com PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/example_common.h)
target_link_libraries(example_com PUBLIC inmost Fem_inmost)
target_link_libraries(example_com INTERFACE MeshGen)
target_include_directories(example_com PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})