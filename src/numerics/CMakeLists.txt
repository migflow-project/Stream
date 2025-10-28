target_sources(numerics
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/conjugate_gradient.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/preconditioner_jacobi.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/conjugate_gradient.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/preconditioner_jacobi.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
endif()
