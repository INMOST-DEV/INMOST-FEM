add_library(MeshGen mesh_gen.cpp)
target_link_libraries(MeshGen PUBLIC inmost)
set_target_properties(MeshGen PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/mesh_gen.h)
target_include_directories(MeshGen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# target_include_directories(MeshGen
#         PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
#         )
# install(FILES mesh_gen.h
#         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/primitives
# )

#set(CARNUM_TARGETS MeshGen ${CARNUM_TARGETS} PARENT_SCOPE)