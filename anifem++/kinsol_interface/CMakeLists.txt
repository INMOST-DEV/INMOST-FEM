add_library(AFSUNSolver SUNNonlinearSolver.cpp SUNNonlinearSolver.h) 
target_link_libraries(AFSUNSolver PUBLIC inmost SUNDIALS::kinsol)
target_include_directories(AFSUNSolver
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
install(FILES SUNNonlinearSolver.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/kinsol_interface
        )                    

set(ANIFEM_TARGETS AFSUNSolver ${ANIFEM_TARGETS} PARENT_SCOPE)
