include(FetchContent)
include(ExternalProject)

message("-- Download CasAdi")
FetchContent_Declare(casadi_get
        URL https://github.com/casadi/casadi/archive/3.5.5.tar.gz
        UPDATE_DISCONNECTED TRUE
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${LIB_DOWNLOAD_PATH}/casadi"
        )
FetchContent_GetProperties(casadi_get)
if(NOT casadi_get_POPULATED)
    FetchContent_Populate(casadi_get)
endif()

file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/casadi/install")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/casadi/install/${CMAKE_INSTALL_INCLUDEDIR}")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/casadi/install/${CMAKE_INSTALL_LIBDIR}")

if( WIN32 )
    set(LIB_TYPE dylib)
    set(LIB_PREF "")
else( WIN32 )
    set(LIB_TYPE so)
    set(LIB_PREF lib)
endif( WIN32 )
set(casadi_LIBRARY "${casadi_get_SOURCE_DIR}/install/${CMAKE_INSTALL_LIBDIR}/${LIB_PREF}casadi.${LIB_TYPE}")

enable_language(C)
ExternalProject_Add(casadi_get
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${casadi_get_SOURCE_DIR}"
        BINARY_DIR "${casadi_get_SOURCE_DIR}/build"
        INSTALL_DIR "${casadi_get_SOURCE_DIR}/install"
        CMAKE_ARGS  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}"
                    "-DBUILD_SHARED_LIBS=ON"
                    "-DCMAKE_INSTALL_PREFIX=${casadi_get_SOURCE_DIR}/install"
                    "-DLIB_PREFIX=${casadi_get_SOURCE_DIR}/install/${CMAKE_INSTALL_LIBDIR}"
                    "-DINCLUDE_PREFIX=${casadi_get_SOURCE_DIR}/install/${CMAKE_INSTALL_INCLUDEDIR}"
        BUILD_BYPRODUCTS "${casadi_LIBRARY}"           
        )

add_library(casadi UNKNOWN IMPORTED GLOBAL)
add_dependencies(casadi casadi_get)
set_target_properties(casadi PROPERTIES IMPORTED_LOCATION "${casadi_LIBRARY}")
target_include_directories(casadi INTERFACE "${LIB_DOWNLOAD_PATH}/casadi/install/${CMAKE_INSTALL_INCLUDEDIR}")
set(casadi_DOWNLOADED TRUE)
install(DIRECTORY "${LIB_DOWNLOAD_PATH}/casadi/install/" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}") 