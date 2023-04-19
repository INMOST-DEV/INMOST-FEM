add_library(SUNSolver SUNNonlinearSolver.cpp SUNNonlinearSolver.h) 
target_link_libraries(SUNSolver PUBLIC inmost SUNDIALS::kinsol)
target_include_directories(SUNSolver
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
install(FILES SUNNonlinearSolver.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/kinsol_interface
        )                    

set(ANIFEM_TARGETS SUNSolver ${ANIFEM_TARGETS} PARENT_SCOPE)
