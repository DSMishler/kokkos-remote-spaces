include(CTest)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           main
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

SET(NAME KokkosRemote_TestAll)

if (KRS_ENABLE_MPISPACE)
  FILE(GLOB TEST_SRCS *.cpp)
  list(FILTER TEST_SRCS EXCLUDE REGEX ".*Test_Subview\\.cpp$")
  message(${TEST_SRCS})
else()
  FILE(GLOB TEST_SRCS *.cpp)
endif()

add_executable(${NAME} ${TEST_SRCS})

target_link_libraries(${NAME} PRIVATE Kokkos::kokkosremote)
target_link_libraries(${NAME} PRIVATE gtest_main)
