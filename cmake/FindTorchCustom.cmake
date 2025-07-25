include_guard(GLOBAL)

find_package(Torch REQUIRED)

if(NOT TARGET Torch::Torch)
  message(
    WARNING
      "Torch::Torch target not defined by libtorch — creating custom Torch::Torch target."
  )
  add_library(Torch::Torch INTERFACE IMPORTED)
  set_property(TARGET Torch::Torch PROPERTY INTERFACE_LINK_LIBRARIES
                                            ${TORCH_LIBRARIES})
  if(Torch_INCLUDE_DIRS)
    set_property(TARGET Torch::Torch PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                              "${Torch_INCLUDE_DIRS}")
  elseif(TORCH_INCLUDE_DIRS)
    set_property(TARGET Torch::Torch PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                              "${TORCH_INCLUDE_DIRS}")
  endif()
endif()
