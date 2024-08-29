include(FetchContent)
include(ExternalProject)

set(eigen3_url https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
if (DEFINED ENV{EIGEN3_SRC_ARCHIVE})
        if(EXISTS $ENV{EIGEN3_SRC_ARCHIVE})
            set(eigen3_url $ENV{EIGEN3_SRC_ARCHIVE})
        endif()
endif()

message("-- Download Eigen3")
FetchContent_Declare(
        eigen3_get
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL "${eigen3_url}"
        UPDATE_DISCONNECTED TRUE
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${LIB_DOWNLOAD_PATH}/eigen3"
)

FetchContent_GetProperties(eigen3_get)
if(NOT eigen3_get_POPULATED)
    FetchContent_Populate(eigen3_get)
endif()

file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/eigen3/install")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/eigen3/install/${CMAKE_INSTALL_INCLUDEDIR}")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/eigen3/install/${CMAKE_INSTALL_INCLUDEDIR}/eigen3")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/eigen3/install/${CMAKE_INSTALL_DATAROOTDIR}")

enable_language(C)
ExternalProject_Add(eigen3_get
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${eigen3_get_SOURCE_DIR}"
        BINARY_DIR "${eigen3_get_SOURCE_DIR}/build"
        INSTALL_DIR "${eigen3_get_SOURCE_DIR}/install"
        CMAKE_ARGS  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DBUILD_TESTING=OFF"
                    "-DEIGEN_BUILD_DOC=OFF"
                    "-DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCMAKE_INSTALL_PREFIX=${eigen3_get_SOURCE_DIR}/install"
                    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
                    "-DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}"
                    "-DCMAKE_INSTALL_DATAROOTDIR=${CMAKE_INSTALL_DATAROOTDIR}"
        )

set (EIGEN3_DEFINITIONS  "")
set (EIGEN3_INCLUDE_DIR  "${LIB_DOWNLOAD_PATH}/eigen3/install/${CMAKE_INSTALL_INCLUDEDIR}/eigen3")
set (EIGEN3_INCLUDE_DIRS "${LIB_DOWNLOAD_PATH}/eigen3/install/${CMAKE_INSTALL_INCLUDEDIR}/eigen3")
set (EIGEN3_ROOT_DIR     "${LIB_DOWNLOAD_PATH}/eigen3/install/")

add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)        
add_dependencies(Eigen3::Eigen eigen3_get)
target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIRS})
if (EIGEN3_DEFINITIONS})
    target_compile_definitions(Eigen3::Eigen ${EIGEN3_DEFINITIONS})
endif()    
set(eigen3_DOWNLOADED ON CACHE BOOL "")
mark_as_advanced(eigen3_DOWNLOADED)
install(DIRECTORY "${LIB_DOWNLOAD_PATH}/eigen3/install/" 
        DESTINATION ".")