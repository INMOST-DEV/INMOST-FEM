set(HELP_TESTS  ${CMAKE_CURRENT_SOURCE_DIR}/ext_array_enum_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/interval_map_test.cpp)
# if (WITH_INMOST)
#     set(IINMOST_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/<name>_test.cpp ${SOLVER_TESTS})
# endif()
set(ANIFEM_TEST ${IINMOST_TESTS} ${HELP_TESTS} ${ANIFEM_TEST} PARENT_SCOPE)