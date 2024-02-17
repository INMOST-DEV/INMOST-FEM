add_library(AutoDiff autodiff.cpp 
                    cauchy_strain_autodiff.h 
                    autodiff.h autodiff.inl cauchy_strain_invariants.h cauchy_strain_invariants.inl 
                    physical_tensors.h physical_tensors.inl physical_tensors_3d.inl 
                    ../fem/geometry.h)

if (WITH_EIGEN)
    target_link_libraries(AutoDiff INTERFACE Eigen3::Eigen)
endif()

install(FILES 
        cauchy_strain_autodiff.h 
        autodiff.h autodiff.inl 
        cauchy_strain_invariants.h cauchy_strain_invariants.inl 
        physical_tensors.h physical_tensors.inl physical_tensors_3d.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/autodiff)

set(ANIFEM_TARGETS AutoDiff ${ANIFEM_TARGETS} PARENT_SCOPE)