
# ===================== geometry ===========================

# Set header when installing target 
target_sources(geometry PUBLIC
    FILE_SET geometry_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/bbox.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/morton.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/lbvh.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/predicates.hpp
)
