add_library(Fem_inmost
        fem.h fem.cpp
        assembler.h assembler.cpp
        elemental_assembler.h elemental_assembler.cpp
        for_assembler.h for_assembler.inl for_assembler.cpp
        )

target_link_libraries(Fem_inmost PUBLIC Fem inmost)
#target_compile_definitions(Fem_inmost PRIVATE -DNO_ASSEMBLER_TIMERS)

target_include_directories(Fem_inmost
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
if (NOT WITH_ASSEMBLER_TIMERS)
        target_compile_definitions(Fem_inmost PUBLIC -DNO_ASSEMBLER_TIMERS)
endif()

install(FILES
            fem.h assembler.h elemental_assembler.h elemental_assembler.inl for_assembler.h for_assembler.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/inmost_interface
        )

set(ANIFEM_TARGETS Fem_inmost ${ANIFEM_TARGETS} PARENT_SCOPE)        






