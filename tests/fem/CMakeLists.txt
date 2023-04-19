add_subdirectory(operations)
add_subdirectory(spaces)
set(ANIFEM_TEST ${CMAKE_CURRENT_SOURCE_DIR}/tetdofmap_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/femspace_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/quadrature_formulas_test.cpp
                ${ANIFEM_TEST} PARENT_SCOPE)               