set(NUMERICS_TESTS
    cg_jacobi
)

foreach(T ${NUMERICS_TESTS})
    add_executable(${T} "${T}.cpp")
    target_link_libraries(${T} PRIVATE numerics)
    if (ENABLE_CUDA) 
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${T}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()
