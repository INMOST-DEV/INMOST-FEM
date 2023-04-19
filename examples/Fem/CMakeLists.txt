project(FemInterfaces)

add_library(example_com Ani/example_common.cpp)
set_target_properties(example_com PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/Ani/example_common.hpp)
target_link_libraries(example_com PUBLIC inmost)

add_executable(diffusion Ani/diffusion.cpp)
target_link_libraries(diffusion CarNum example_com)

add_executable(dyn_diffusion Ani/dyn_diffusion.cpp)
target_link_libraries(dyn_diffusion CarNum example_com stdc++fs)

add_executable(stokes-navier Ani/StokesNavier.cpp)
target_link_libraries(stokes-navier CarNum example_com stdc++fs)

set(FEM_EXAMPLES diffusion dyn_diffusion stokes-navier)

if (WITH_KINSOL)
    add_executable(bendbeam Ani/BendBeam.cpp)
    target_link_libraries(bendbeam CarNum example_com stdc++fs)
    if (WITH_CASADI)
        target_link_libraries(bendbeam casadi)
    endif()
    set(FEM_EXAMPLES ${FEM_EXAMPLES} bendbeam)
endif()

install(TARGETS ${FEM_EXAMPLES} EXPORT ${package_NAME}-targets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples/Fem/Ani)