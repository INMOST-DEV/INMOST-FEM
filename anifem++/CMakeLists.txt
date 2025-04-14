add_subdirectory(interval)
add_subdirectory(fem)
if (WITH_INMOST)
    add_subdirectory(inmost_interface)
    if (WITH_KINSOL)
        add_subdirectory(kinsol_interface)
    endif()
endif()
add_subdirectory(autodiff)
if (WITH_INMOST)
    add_subdirectory(utils)
endif()

add_library(AniFem++ INTERFACE)
target_link_libraries(AniFem++ INTERFACE Fem)
if (WITH_INMOST)
    target_link_libraries(AniFem++ INTERFACE Fem_inmost)
endif()

set(ANIFEM_TARGETS AniFem++ ${ANIFEM_TARGETS} PARENT_SCOPE)





