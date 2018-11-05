find_path(JsonCpp_INCLUDE_DIR "json/json.h"
  PATH_SUFFIXES "include"
  DOC "Specify the JsonCpp include directory here")

find_library(JsonCpp_LIBRARY
  NAMES jsoncpp
  PATHS "src/lib_json"
  DOC "Specify the JsonCpp library here")
  
set(JsonCpp_INCLUDE_DIRS ${JsonCpp_INCLUDE_DIR})
set(JsonCpp_LIBRARIES "${JsonCpp_LIBRARY}")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JsonCpp
  REQUIRED_VARS JsonCpp_LIBRARIES JsonCpp_INCLUDE_DIRS
  ${_JsonCpp_version_args})

mark_as_advanced(JsonCpp_INCLUDE_DIR JsonCpp_LIBRARY)
