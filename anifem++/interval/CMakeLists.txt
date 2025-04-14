add_library(interval interval.cpp ext_array_enum.h ext_array_enum.inl interval_enum.h interval_enum.inl interval_map.h interval_map.inl)

target_include_directories(interval
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

install(FILES
            ext_array_enum.h ext_array_enum.inl interval_enum.h interval_enum.inl
            interval_map.h interval_map.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/interval
        )

set(ANIFEM_TARGETS interval ${ANIFEM_TARGETS} PARENT_SCOPE) 