add_subdirectory(operations)
add_subdirectory(spaces)

add_library(QuadFormulas quadrature_formulas.h quadrature_formulas.cpp)
if (WITH_EIGEN)
    target_link_libraries(QuadFormulas PUBLIC Eigen3::Eigen)
endif()
target_include_directories(QuadFormulas PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )       
install(FILES quadrature_formulas.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/fem)

list(TRANSFORM OPERATION_HEADERS PREPEND "operations/")
list(TRANSFORM SPACES_HEADERS PREPEND "spaces/")
add_library(Fem diff_tensor.h fem_space.h fem_space.cpp fem_space.inl expr_interface.h expr_interface.inl numeric_expr.h numeric_expr.inl
                operators.h operators.inl geometry.h fem_memory_parallel.h mutex_type.h fem_memory.h 
                tetdofmap.h tetdofmap.inl tetdofmap.cpp ${SPACES_HEADERS} ${OPERATION_HEADERS})
target_link_libraries(Fem PUBLIC QuadFormulas)
if (WITH_EIGEN)
    target_link_libraries(Fem PUBLIC Eigen3::Eigen)
endif()

target_include_directories(Fem INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
   
install(FILES 
            diff_tensor.h fem_space.h fem_space.inl expr_interface.h expr_interface.inl numeric_expr.h numeric_expr.inl
            operators.h operators.inl geometry.h fem_memory.h mutex_type.h fem_memory_parallel.h
            tetdofmap.h tetdofmap.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/fem)

set(ANIFEM_TARGETS Fem QuadFormulas ${ANIFEM_TARGETS} PARENT_SCOPE)        