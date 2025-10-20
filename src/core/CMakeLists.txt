target_sources(geometry
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ava_instanciation.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ava_instanciation.cpp TARGET_DIRECTORY geometry PROPERTIES LANGUAGE CUDA)
endif()
