set(ANIFEM_TEST ${CMAKE_CURRENT_SOURCE_DIR}/int_tet_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/int_face_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/int_edge_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/int_pnt_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/eval_test.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/dc_on_dof_test.cpp
                ${ANIFEM_TEST} PARENT_SCOPE)