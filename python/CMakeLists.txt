set(PYTHON_API_SRC
  __init__.py
  numerics/__init__.py
  numerics/cg.py 
  numerics/csr.py
  numerics/jacobi.py
  numerics/linsys.py
  mesh/__init__.py
  mesh/mesh.py
)

foreach(f ${PYTHON_API_SRC})
    configure_file("${f}" "${PYTHON_BUILD_DIR}/${f}" @ONLY)
endforeach()

# The 'DESTINATION' should be a location that is included in the user's PYTHON_PATH.
install(
    DIRECTORY ${PYTHON_BUILD_DIR}/
    DESTINATION "${PYTHON_INSTALL_DIR}"
    PATTERN "*.py"
)

