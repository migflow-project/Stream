set(IMR26_TESTS
    uniform2D
)

foreach(T ${IMR26_TESTS})
    add_executable(${T} "${T}.cpp")
    target_link_libraries(${T} PRIVATE mesh)
    if (ENABLE_CUDA) 
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${T}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()
