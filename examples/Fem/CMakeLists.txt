project(FemInterfaces)

add_executable(diffusion Ani/diffusion.cpp)
target_link_libraries(diffusion CarNum Fem_utils)

add_executable(dyn_diffusion Ani/dyn_diffusion.cpp)
target_link_libraries(dyn_diffusion CarNum Fem_utils stdc++fs)

add_executable(stokes-navier Ani/StokesNavier.cpp)
target_link_libraries(stokes-navier CarNum Fem_utils stdc++fs)

set(FEM_EXAMPLES diffusion dyn_diffusion stokes-navier)

if (WITH_KINSOL)
    add_executable(bendbeam Ani/BendBeam.cpp)
    target_link_libraries(bendbeam CarNum Fem_utils stdc++fs)
    if (WITH_CASADI)
        target_link_libraries(bendbeam casadi)
    endif()
    set(FEM_EXAMPLES ${FEM_EXAMPLES} bendbeam)
endif()

install(TARGETS ${FEM_EXAMPLES} EXPORT ${package_NAME}-targets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples/Fem/Ani)