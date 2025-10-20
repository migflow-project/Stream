target_sources(geometry
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vec.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/lbvh.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vec.cpp TARGET_DIRECTORY geometry PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp TARGET_DIRECTORY geometry PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp TARGET_DIRECTORY geometry PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/lbvh.cpp TARGET_DIRECTORY geometry PROPERTIES LANGUAGE CUDA)
endif()
