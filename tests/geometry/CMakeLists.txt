set(GEOMETRY_TESTS
    vec                             
    lbvh_simple_collision            
    lbvh_random_collision
    lbvh_random_collision3D
    lbvh_agglo_simple_collision
    lbvh_agglo_random_collision
    lbvh_agglo_random_collision3D
    traversal_stack
)

foreach(T ${GEOMETRY_TESTS})
    add_executable(${T} "${T}.cpp")
    target_link_libraries(${T} PRIVATE geometry)
    if (ENABLE_CUDA) 
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${T}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()
