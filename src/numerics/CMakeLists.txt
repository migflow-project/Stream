target_sources(numerics
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
endif()
