
# ===================== geometry ===========================

# Set header when installing target 
target_sources(numerics PUBLIC
    FILE_SET numerics_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/csr.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linear_system.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/solver.hpp
)
