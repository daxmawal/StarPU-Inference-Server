if(NOT STARPU_DIR AND DEFINED ENV{STARPU_DIR})
    set(STARPU_DIR $ENV{STARPU_DIR} CACHE PATH "Root directory of StarPU installation")
endif()

if(NOT STARPU_DIR)
    message(FATAL_ERROR "Please set the STARPU_DIR environment variable or cache entry")
endif()

set(STARPU_INCLUDE_DIR "${STARPU_DIR}/include/starpu/1.4")
if(NOT EXISTS "${STARPU_INCLUDE_DIR}/starpu.h")
    message(FATAL_ERROR "Missing starpu.h in ${STARPU_INCLUDE_DIR}")
endif()

find_library(STARPU_LIBRARY
    NAMES starpu-1.4
    HINTS ${STARPU_DIR}/lib
)

if(NOT STARPU_LIBRARY)
    message(FATAL_ERROR "Could not find starpu-1.4 in ${STARPU_DIR}/lib")
endif()

add_library(StarPU::starpu SHARED IMPORTED GLOBAL)
set_target_properties(StarPU::starpu PROPERTIES
    IMPORTED_LOCATION "${STARPU_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${STARPU_INCLUDE_DIR}"
)

set(StarPU_FOUND TRUE)
set(StarPU_INCLUDE_DIR "${STARPU_INCLUDE_DIR}")
set(StarPU_LIBRARY "${STARPU_LIBRARY}")