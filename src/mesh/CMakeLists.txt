target_sources(mesh
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mesh.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DirectAlphaShape2D.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/mesh.cpp TARGET_DIRECTORY mesh PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/DirectAlphaShape2D.cpp TARGET_DIRECTORY mesh PROPERTIES LANGUAGE CUDA)
endif()
