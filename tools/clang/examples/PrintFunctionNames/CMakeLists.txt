set(MODULE TRUE)

set(LLVM_NO_RTTI 1)

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
