if (TARGET geometry)
    target_include_directories(geometry
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/geometry/>
               $<INSTALL_INTERFACE:include/stream/geometry/>
    )

    target_include_directories(geometry
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(geometry/)
endif()

if (TARGET numerics)
    target_include_directories(numerics
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/numerics/>
               $<INSTALL_INTERFACE:include/stream/numerics/>
    )

    target_include_directories(numerics
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(numerics/)
endif()

if (TARGET mesh)
    target_include_directories(mesh
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mesh/>
               $<INSTALL_INTERFACE:include/stream/mesh/>
    )

    target_include_directories(mesh
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(mesh/)
endif()
