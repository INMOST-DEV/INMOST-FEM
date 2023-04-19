include(FetchContent)
include(ExternalProject)

message("-- Download SUNDIALS")
FetchContent_Declare(sundials_get
        URL https://github.com/LLNL/sundials/archive/v5.6.1.tar.gz
        UPDATE_DISCONNECTED TRUE
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${LIB_DOWNLOAD_PATH}/sundials"
        )
FetchContent_GetProperties(sundials_get)
if(NOT sundials_get_POPULATED)
    FetchContent_Populate(sundials_get)
endif()

file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/sundials/install")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/sundials/install/${CMAKE_INSTALL_INCLUDEDIR}")
file(MAKE_DIRECTORY "${LIB_DOWNLOAD_PATH}/sundials/install/${CMAKE_INSTALL_LIBDIR}")
if( WIN32 )
    set(LIB_TYPE dylib)
    set(LIB_PREF "")
else( WIN32 )
    set(LIB_TYPE so)
    set(LIB_PREF lib)
endif( WIN32 )
macro(create_sundials_LIBRARY ltarget)
    set(sundials_${ltarget}_LIBRARY "${sundials_get_SOURCE_DIR}/install/${CMAKE_INSTALL_LIBDIR}/${LIB_PREF}sundials_${ltarget}.${LIB_TYPE}")
endmacro() 
create_sundials_LIBRARY("kinsol") 
create_sundials_LIBRARY("cvode") 

enable_language(C)
ExternalProject_Add(sundials_get
        PREFIX "${LIB_DOWNLOAD_PATH}"
        SOURCE_DIR "${sundials_get_SOURCE_DIR}"
        BINARY_DIR "${sundials_get_SOURCE_DIR}/build"
        INSTALL_DIR "${sundials_get_SOURCE_DIR}/install"
        CMAKE_ARGS  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DBUILD_TESTING=OFF"
                    "-DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}"
                    "-DBUILD_SHARED_LIBS=ON"
                    "-DCMAKE_INSTALL_PREFIX=${sundials_get_SOURCE_DIR}/install"
                    "-DBUILD_ARKODE=OFF"
                    "-DBUILD_CVODE=ON"
                    "-DBUILD_CVODES=OFF"
                    "-DBUILD_FORTRAN77_INTERFACE=OFF"
                    "-DBUILD_FORTRAN_MODULE_INTERFACE=OFF"
                    "-DBUILD_IDA=OFF"
                    "-DBUILD_IDAS=OFF"
                    "-DBUILD_KINSOL=ON"
                    "-DEXAMPLES_INSTALL=OFF"
                    "-DENABLE_MPI=${WITH_MPI}"
                    "-DENABLE_OPENMP=${WITH_OPENMP}"
                    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
                    "-DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}"
        BUILD_BYPRODUCTS "${sundials_kinsol_LIBRARY}" "${sundials_cvode_LIBRARY}"            
        )

function(get_sundials_target ltarget)
    set(sundials_${ltarget}_LIBRARY "${sundials_get_SOURCE_DIR}/install/${CMAKE_INSTALL_LIBDIR}/${LIB_PREF}sundials_${ltarget}.${LIB_TYPE}")
    add_library(SUNDIALS::${ltarget} UNKNOWN IMPORTED GLOBAL)
    add_dependencies(SUNDIALS::${ltarget} sundials_get)
    set_target_properties(SUNDIALS::${ltarget} PROPERTIES IMPORTED_LOCATION "${sundials_${ltarget}_LIBRARY}")
    target_include_directories(SUNDIALS::${ltarget} INTERFACE "${LIB_DOWNLOAD_PATH}/sundials/install/${CMAKE_INSTALL_INCLUDEDIR}")
    target_link_libraries(SUNDIALS::${ltarget} INTERFACE "\$<LINK_ONLY:m>")
    target_compile_definitions(SUNDIALS::${ltarget} INTERFACE SUNDIALS_VERSION_MAJOR=5 SUNDIALS_VERSION_MINOR=6 SUNDIALS_VERSION_PATCH=1)
endfunction(get_sundials_target ltarget)

get_sundials_target("kinsol")
get_sundials_target("cvode")

set(sundials_DOWNLOADED TRUE)  
install(DIRECTORY "${LIB_DOWNLOAD_PATH}/sundials/install/" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}")       