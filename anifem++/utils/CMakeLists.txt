set(FEM_UTILS_SOURCES mesh_utils.cpp utils.cpp)
set(FEM_UTILS_HEADERS mesh_utils.h utils.h utils.inl)

add_library(Fem_utils ${FEM_UTILS_SOURCES} ${FEM_UTILS_HEADERS})
target_link_libraries(Fem_utils PUBLIC Fem inmost)
target_link_libraries(Fem_utils PRIVATE Fem_inmost)
target_include_directories(Fem_utils PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
install(FILES ${FEM_UTILS_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/utils)
set(ANIFEM_TARGETS Fem_utils ${ANIFEM_TARGETS} PARENT_SCOPE)