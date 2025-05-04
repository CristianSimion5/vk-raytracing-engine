#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)

#--------------------------------------------------------------------------------------------------
# Project setting
project(vk-rt-engine LANGUAGES C CXX)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

#--------------------------------------------------------------------------------------------------
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()

## Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
endif()

#--------------------------------------------------------------------------------------------------
# Package shared by all projects
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_nvpro_core_lib()

message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})

#--------------------------------------------------------------------------------------------------
# C++ target and defines
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJECT_NAME})
_add_project_definitions(${PROJECT_NAME})


#--------------------------------------------------------------------------------------------------
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB EXTRA_COMMON common/*.*)
list(APPEND COMMON_SOURCE_FILES ${EXTRA_COMMON})
include_directories(common)

file (GLOB NRD_INCLUDE_HEADERS "_NRD_SDK/Include/*.h" "_NRD_SDK/Include/*.hpp")
file (GLOB NRD_INTEGRATION_HEADERS "_NRD_SDK/Integration/*.h" "_NRD_SDK/Integration/*.hpp")
include_directories(_NRD_SDK/Include)
include_directories(_NRD_SDK/Integration)
include_directories(_NRI_SDK/Include)

#--------------------------------------------------------------------------------------------------
# GLSL to SPIR-V custom build
compile_glsl_directory(
	SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders" 
	DST "${CMAKE_CURRENT_SOURCE_DIR}/spv" 
	VULKAN_TARGET "vulkan1.2"
	DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
	)


#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJECT_NAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES})
target_sources(${PROJECT_NAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJECT_NAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJECT_NAME} PUBLIC ${GLSL_SOURCES} ${GLSL_HEADERS})

#target_sources(${PROJECT_NAME} PUBLIC ${NRD_INTEGRATION_HEADERS} ${NRD_INCLUDE_HEADERS})

#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"       FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Sources"      FILES ${SOURCE_FILES})
source_group("Headers"      FILES ${HEADER_FILES})
source_group("Shader Sources"  FILES ${GLSL_SOURCES})
source_group("Shader Headers"  FILES ${GLSL_HEADERS})

#source_group("NRD" FILES ${NRD_INTEGRATION_HEADERS} ${NRD_INCLUDE_HEADERS})

#--------------------
# Linking NRD
#

#message(STATUS "NRD_INTEGRATION_HEADERS = ${NRD_INTEGRATION_HEADERS}")

#target_include_directories(${PROJECT_NAME} INTERFACE ${NRD_DIR}/Include)
#target_include_directories(${PROJECT_NAME} INTERFACE ${NRD_DIR}/Integration)

#target_link_libraries(${PROJECT_NAME} PRIVATE path/to/sdk/libraries/library1.dll path/to/sdk/libraries/library2.dll)
# Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
#add_library(NRD SHARED IMPORTED)
# You can define two import-locations: one for debug and one for release.
#set_target_properties(NRD PROPERTIES IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/_NRD_SDK/Lib/Debug/NRD.lib)
#set_target_properties(NRD PROPERTIES IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/_NRD_SDK/Lib/Release/NRD.lib)

#target_link_libraries(${PROJECT_NAME} NRD)

#--------------------------------------------------------------------------------------------------
# Linkage
#
target_link_libraries(${PROJECT_NAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJECT_NAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJECT_NAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)


#--------------------------------------------------------------------------------------------------
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJECT_NAME} )


install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJECT_NAME}/spv")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJECT_NAME}/spv")


#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
install(DIRECTORY "media" 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" 
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug")