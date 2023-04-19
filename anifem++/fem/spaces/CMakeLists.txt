set(SPACES_HEADERS  common.h 
                    poly_0.h poly_1.h poly_2.h poly_3.h 
                    raviant_thomas_0.h nedelec_0.h crouzeix_raviart_1.h 
                    bubble_4.h
                    spaces.h)
set(SPACES_HEADERS ${SPACES_HEADERS} PARENT_SCOPE)  

install(FILES ${SPACES_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/fem/spaces/)