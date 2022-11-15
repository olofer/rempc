cmake_minimum_required(VERSION 3.10)

project(calc_tests LANGUAGES C)

set(C_STANDARD C99)
set(C_STANDARD_REQUIRED ON)

add_executable(mcheck ${PROJECT_SOURCE_DIR}/testsrc/mcheck.c)
target_include_directories(mcheck PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(mcheck PUBLIC ${PROJECT_SOURCE_DIR}/testsrc)
target_link_libraries(mcheck PRIVATE -lm -lrt)
target_compile_options(mcheck PRIVATE -Wall -Wextra -Wpedantic -O2)

enable_testing()
add_test(NAME mcheck-1 COMMAND mcheck 40 50 30)
