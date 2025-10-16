target_include_directories(geometry
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/geometry/>
           $<INSTALL_INTERFACE:include/stream/geometry/>
)

target_include_directories(geometry
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
           $<INSTALL_INTERFACE:include/stream/>
)

add_subdirectory(geometry/)
