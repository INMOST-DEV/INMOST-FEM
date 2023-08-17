
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

set(ANIFEM_TEST "") # - list of tests
add_subdirectory(fem)
add_subdirectory(kinsol_interface)
add_subdirectory(inmost_interface)

add_executable( anifem_test ${ANIFEM_TEST} )
target_link_libraries(anifem_test gtest_main AniFem++)
if (WITH_KINSOL)
  target_link_libraries(anifem_test AFSUNSolver)
endif()
gtest_discover_tests(anifem_test)
