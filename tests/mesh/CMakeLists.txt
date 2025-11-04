set(MESH_TESTS
    init_mesh
    insert_morton_neighbors
    insert_by_circumsphere_check
)

foreach(T ${MESH_TESTS})
    add_executable(${T} "${T}.cpp")
    target_link_libraries(${T} PRIVATE mesh)
    if (ENABLE_CUDA) 
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${T}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()
