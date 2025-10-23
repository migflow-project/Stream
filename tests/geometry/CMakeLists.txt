add_executable(test_vec test_vec.cpp)
target_link_libraries(test_vec PRIVATE geometry)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_vec.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_lbvh_coll test_lbvh_simple_collision.cpp)
target_link_libraries(test_lbvh_coll PRIVATE geometry)
if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_lbvh_simple_collision.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_lbvh_coll_rand test_lbvh_random_collision.cpp)
target_link_libraries(test_lbvh_coll_rand PRIVATE geometry)
if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_lbvh_random_collision.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_lbvh_agglo_simple test_lbvh_agglo_simple_collision.cpp)
target_link_libraries(test_lbvh_agglo_simple PRIVATE geometry)
if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_lbvh_agglo_simple_collision.cpp PROPERTIES LANGUAGE CUDA)
endif()
