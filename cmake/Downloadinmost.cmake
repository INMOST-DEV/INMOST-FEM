include(FetchContent)
include(ExternalProject)

set(inmost_url https://github.com/INMOST-DEV/INMOST/archive/5fdd88f33aec5f2f6b74e02e227b021bb2daa0ac.tar.gz)
if (DEFINED ENV{INMOST_SRC_ARCHIVE})
        if(EXISTS $ENV{INMOST_SRC_ARCHIVE})
            set(inmost_url $ENV{INMOST_SRC_ARCHIVE})
        endif()
endif()

message("-- Download INMOST")
FetchContent_Declare(inmost_get
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL "${inmost_url}"
        UPDATE_DISCONNECTED TRUE
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${LIB_DOWNLOAD_PATH}/inmost"
        )
FetchContent_GetProperties(inmost_get)
if(NOT inmost_get_POPULATED)
    FetchContent_Populate(inmost_get)
endif()

file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/inmost/install")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/inmost/install/include")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/inmost/install/lib")

if( WIN32 )
    set(LIB_TYPE lib)
    set(LIB_PREF "")
else( WIN32 )
    set(LIB_TYPE a)
    set(LIB_PREF lib)
endif( WIN32 )
set(inmost_LIBRARY_DIR "${inmost_get_SOURCE_DIR}/install/lib")
set(inmost_LIBRARY "${inmost_LIBRARY_DIR}/${LIB_PREF}inmost.${LIB_TYPE}")

enable_language(C)
ExternalProject_Add(inmost_get
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${inmost_get_SOURCE_DIR}"
        BINARY_DIR "${inmost_get_SOURCE_DIR}/build"
        INSTALL_DIR "${inmost_get_SOURCE_DIR}/install"
        CMAKE_ARGS  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                    "-DCMAKE_INSTALL_PREFIX=${inmost_get_SOURCE_DIR}/install"
                    "-DCOMPILE_EXAMPLES=OFF"
                    "-DCOMPILE_TESTS=OFF"
                    "-DUSE_AUTODIFF=OFF"
                    "-DUSE_MESH=ON"
                    "-DUSE_MPI=${WITH_MPI}"
                    "-DUSE_NONLINEAR=ON"
                    "-DUSE_OMP=${WITH_OPENMP}"
                    "-DUSE_OPENCL=${WITH_OPENCL}"
                    "-DUSE_OPTIMIZER=ON"
                    "-DUSE_PARTITIONER=ON"
                    "-DUSE_SOLVER=ON"
        BUILD_BYPRODUCTS "${inmost_LIBRARY}"            
        )

add_library(inmost STATIC IMPORTED GLOBAL)
add_dependencies(inmost inmost_get)
set_target_properties(inmost PROPERTIES IMPORTED_LOCATION "${inmost_LIBRARY}")
target_include_directories(inmost INTERFACE "${LIB_DOWNLOAD_PATH}/inmost/install/include")

set(INMOST_INCLUDE_DIRS "${inmost_get_SOURCE_DIR}/install/include")
set(INMOST_LIBRARY_DIRS "${inmost_get_SOURCE_DIR}/install/lib")
set(INMOST_LIBRARIES inmost)
set(INMOST_DEFINITIONS "")

if (WITH_MPI)
    list(APPEND INMOST_LIBRARIES MPI::MPI_CXX)
endif()
if (WITH_OPENCL)
    list(APPEND INMOST_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
endif()

set(inmost_DOWNLOADED ON CACHE BOOL "")
mark_as_advanced(inmost_DOWNLOADED)
install(DIRECTORY "${LIB_DOWNLOAD_PATH}/inmost/install/" 
        DESTINATION ".")