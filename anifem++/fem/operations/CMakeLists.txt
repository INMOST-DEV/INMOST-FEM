set(OPERATION_HEADERS core.h core.inl dc_on_dof.h dc_on_dof.inl eval.h eval.inl 
    int_edge.h int_edge.inl int_face.h int_face.inl int_pnt.h 
    int_pnt.inl int_tet.h int_tet.inl interpolate.h interpolate.inl operations.h)
set(OPERATION_HEADERS ${OPERATION_HEADERS} PARENT_SCOPE)  

install(FILES ${OPERATION_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/fem/operations/)    