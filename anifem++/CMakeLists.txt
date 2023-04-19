add_subdirectory(fem)
if (WITH_INMOST)
    add_subdirectory(inmost_interface)
    if (WITH_KINSOL)
        add_subdirectory(kinsol_interface)
    endif()
endif()

add_library(AniFem++ INTERFACE)
target_link_libraries(AniFem++ INTERFACE Fem)
if (WITH_INMOST)
    target_link_libraries(AniFem++ INTERFACE Fem_inmost)
    if (WITH_KINSOL)
        target_link_libraries(AniFem++ INTERFACE SUNSolver)
    endif()
endif()

set(ANIFEM_TARGETS ${ANIFEM_TARGETS} PARENT_SCOPE)





