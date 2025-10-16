add_executable(test_vec test_vec.cpp)
target_link_libraries(test_vec PRIVATE geometry)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_vec.cpp PROPERTIES LANGUAGE CUDA)
endif()
