include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT DEFINED tinytensor_INSTALL_CMAKEDIR)
   set(tinytensor_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tinytensor"
       CACHE STRING "Path to tinytensor CMake files")
endif ()

install(TARGETS tinytensor EXPORT tinytensor_Targets
        RUNTIME COMPONENT tinytensor_Runtime
        LIBRARY COMPONENT tinytensor_Runtime
        NAMELINK_COMPONENT tinytensor_Development
        ARCHIVE COMPONENT tinytensor_Development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${tinytensor_SOURCE_DIR}/include/" "${tinytensor_BINARY_DIR}/include/"
        TYPE INCLUDE
        COMPONENT tinytensor_Development)

if (BUILD_SHARED_LIBS)
    set(type shared)
    message(STATUS "Building tinytensor shared lib")
else ()
    set(type static)
    message(STATUS "Building tinytensor static lib")
endif ()

install(EXPORT tinytensor_Targets
        DESTINATION "${tinytensor_INSTALL_CMAKEDIR}"
        NAMESPACE tinytensor::
        FILE tinytensor-${type}-targets.cmake
        COMPONENT tinytensor_Development)

write_basic_package_version_file(
    tinytensorConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/tinytensorConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/tinytensorConfigVersion.cmake"
        DESTINATION "${tinytensor_INSTALL_CMAKEDIR}"
        COMPONENT tinytensor_Development)
