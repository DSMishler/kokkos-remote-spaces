add_executable(teams teams_benchmark.cpp)
target_link_libraries(teams PRIVATE Kokkos::kokkosremote)
target_include_directories(teams PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(teams PRIVATE KRS_ENABLE_NVSHMEM_PTR)
