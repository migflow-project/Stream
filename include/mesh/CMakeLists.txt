
# ===================== mesh ===========================

# Set header when installing target 
target_sources(mesh PUBLIC
    FILE_SET mesh_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.hpp
)
