
add_subdirectory(primitives)
add_library(excom example_common.cpp)
set_target_properties(excom PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/example_common.h)
target_link_libraries(excom PUBLIC inmost Fem_inmost)
target_link_libraries(excom INTERFACE MeshGen)
target_include_directories(excom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})