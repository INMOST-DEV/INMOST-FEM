add_library(Fem_inmost
#         fem.h fem.cpp
#         assembler.h assembler.cpp
        func_wrap.h func_wrap.inl timer_wrap.h ext_array_enum.h ext_array_enum.inl interval_enum.h interval_enum.inl
        elemental_assembler.h elemental_assembler.cpp elemental_assembler.inl
        ordering.h ordering.inl global_enumerator.h global_enumerator.cpp assembler.h assembler.inl fem.h
#         for_assembler.h for_assembler.inl for_assembler.cpp
        )

target_link_libraries(Fem_inmost PUBLIC Fem inmost)
# #target_compile_definitions(Fem_inmost PRIVATE -DNO_ASSEMBLER_TIMERS)

target_include_directories(Fem_inmost
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
if (NOT WITH_ASSEMBLER_TIMERS)
        target_compile_definitions(Fem_inmost PUBLIC -DNO_ASSEMBLER_TIMERS)
endif()

install(FILES
            func_wrap.h func_wrap.inl timer_wrap.h ext_array_enum.h ext_array_enum.inl interval_enum.h interval_enum.inl
            elemental_assembler.h 
            ordering.h ordering.inl global_enumerator.h assembler.h assembler.inl fem.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anifem++/inmost_interface
        )

set(ANIFEM_TARGETS Fem_inmost ${ANIFEM_TARGETS} PARENT_SCOPE)        
