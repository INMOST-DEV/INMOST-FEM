cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

set(PACKAGE_MAJOR_VERSION 1)
set(PACKAGE_MINOR_VERSION 0)
set(PACKAGE_PATCH_VERSION 0)
set(PACKAGE_VERSION "${PACKAGE_MAJOR_VERSION}.${PACKAGE_MINOR_VERSION}.${PACKAGE_PATCH_VERSION}")
set(PACKAGE_NAME "AniFem++")

string(TOLOWER ${PACKAGE_NAME} package_NAME)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} DESCRIPTION "Flexible toolkit for constructing finite element method discretizations" LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" "${CMAKE_MODULE_PATH}")

option(COMPILE_EXAMPLES "Compile usage examples" OFF)
option(COMPILE_TESTS "Compile unit tests" OFF)

option(WITH_MPI "Add MPI support" OFF)
option(WITH_OPENMP "Add OpenMP support" OFF)
option(WITH_INMOST "Add INMOST library" ON)
option(WITH_EIGEN "Add Eigen3 library" OFF)
option(WITH_KINSOL "Add KINSOL library" OFF)
option(WITH_CASADI "Add CasAdi library" OFF)

option(DOWNLOAD_casadi "Attempts to download CasAdi if library not found" OFF)
option(DOWNLOAD_sundials "Attempts to download SUNDIALS (KINSOL) if library not found" OFF)
option(DOWNLOAD_eigen3 "Attempts to download Eigen3 if library not found" OFF)
option(DOWNLOAD_inmost "Attempts to download INMOST if library not found" OFF)
set(LIB_DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/external_packages")

option(WITH_ASSEMBLER_TIMERS "Add timers in global FEM assembler structure" OFF)

#set(CMAKE_CXX_FLAGS_RELEASE "-march=native -DNDEBUG -flto -ffast-math -O3 -msse2")

if(WITH_MPI)
	find_package(MPI REQUIRED)
	link_libraries(MPI::MPI_CXX)
	add_compile_definitions("WITH_MPI")
endif()

if(WITH_OPENMP)
	find_package(OpenMP REQUIRED)
	link_libraries(OpenMP::OpenMP_CXX)
	add_compile_definitions("WITH_OPENMP")
endif()

# Get correct build paths automatically, but expose CMAKE_INSTALL_LIBDIR
# as a regular cache variable so that a user can more easily see what
# the library dir was set to be by GNUInstallDirs.
include(GNUInstallDirs)
mark_as_advanced(CLEAR CMAKE_INSTALL_LIBDIR)

if (WITH_INMOST)
	if (NOT TARGET inmost)
		if(NOT DOWNLOAD_inmost)
			find_package(INMOST REQUIRED)
		else()
			find_package(INMOST QUIET)
			if (NOT INMOST_FOUND)
				include(cmake/Downloadinmost.cmake)
			endif()
		endif()	
		if(DEFINED USE_MPI)
			if ( NOT (${USE_MPI} STREQUAL ${WITH_MPI}) )
				if (NOT USE_MPI)
					message(ERROR "Attempt link NOMPI version to MPI version of the package")
				else()
					message(WARNING "MPI version of INMOST links to NOMPI version of the package")  
				endif()
			endif()
		endif()
		target_compile_definitions(inmost INTERFACE ${INMOST_DEFINITIONS})
		target_link_directories(inmost INTERFACE ${INMOST_LIBRARY_DIRS})
		target_link_libraries(inmost INTERFACE ${INMOST_LIBRARIES})
		target_include_directories(inmost INTERFACE ${INMOST_INCLUDE_DIRS})
	endif()
endif()

if (WITH_EIGEN)
	if (NOT TARGET Eigen3::Eigen)
		if (NOT DOWNLOAD_eigen3)
			find_package (Eigen3 3.4 REQUIRED )
		else()
			find_package (Eigen3 3.4 QUIET )
			if (NOT Eigen3_FOUND)
				include(cmake/Downloadeigen3.cmake)
			endif()
		endif()	
	endif()
	add_compile_definitions("WITH_EIGEN")
endif()

if (WITH_KINSOL)
	if (NOT TARGET SUNDIALS::kinsol)
		set(_sundials_components kinsol)

		if (NOT DOWNLOAD_sundials)
			find_package(SUNDIALS COMPONENTS ${_sundials_components} REQUIRED)
		else()
			find_package(SUNDIALS COMPONENTS ${_sundials_components} QUIET)
			if (NOT SUNDIALS_FOUND)
				include(cmake/Downloadsundials.cmake)
			endif()
		endif()	
	endif()
	add_compile_definitions($<$<BOOL:${WITH_KINSOL}>:"WITH_KINSOL">)
endif()

if (WITH_CASADI)
	if (NOT TARGET casadi)
		if (NOT DOWNLOAD_casadi)
			find_package(casadi REQUIRED)
		else()
			find_package(casadi QUIET)
			if (NOT casadi_FOUND)
				include(cmake/Downloadcasadi.cmake)
			endif()
		endif()
	endif()
	add_compile_definitions("WITH_CASADI")
endif()

set(ANIFEM_TARGETS "")

add_subdirectory(anifem++) #will install project targets 

if(COMPILE_EXAMPLES)
    add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
if(COMPILE_TESTS)
    add_subdirectory(tests)
endif(COMPILE_TESTS)

install(TARGETS ${ANIFEM_TARGETS}
        EXPORT ${package_NAME}-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT "${package_NAME}-targets" 
		NAMESPACE ${PACKAGE_NAME}::
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${package_NAME}")
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
set(DOWNLOADED_LIBS_ROOT ${CMAKE_INSTALL_PREFIX})
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${package_NAME}-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${package_NAME}"
  PATH_VARS DOWNLOADED_LIBS_ROOT
  )
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${package_NAME}ConfigVersion.cmake"
  VERSION "${PACKAGE_VERSION}"
  COMPATIBILITY SameMinorVersion
)
# install the generated configuration files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${package_NAME}-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${package_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${package_NAME}"
  )







