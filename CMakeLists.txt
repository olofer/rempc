cmake_minimum_required(VERSION 3.10)

project(calc_tests LANGUAGES C)

set(C_STANDARD C99)
set(C_STANDARD_REQUIRED ON)

add_executable(mcheck ${PROJECT_SOURCE_DIR}/testsrc/mcheck.c)
target_include_directories(mcheck PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(mcheck PUBLIC ${PROJECT_SOURCE_DIR}/testsrc)
target_link_libraries(mcheck PRIVATE -lm -lrt)
target_compile_options(mcheck PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter -O2)

add_executable(niqcheck ${PROJECT_SOURCE_DIR}/testsrc/niqcheck.c)
target_include_directories(niqcheck PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(niqcheck PUBLIC ${PROJECT_SOURCE_DIR}/testsrc)
target_link_libraries(niqcheck PRIVATE -lm -lrt)
target_compile_options(niqcheck PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter -O2)

enable_testing()

set(MaxDim "200")
set(Tolerance "1.0e-14")

add_test(NAME smoke-mcheck COMMAND mcheck 40 50 30)
add_test(NAME smoke-mcheck-tolerance COMMAND python3  ../testsrc/mcheck.py --do-not-generate --tolerance ${Tolerance})

foreach(itr RANGE 1 10)
  add_test(NAME mcheck-${itr} COMMAND python3  ../testsrc/mcheck.py --N ${MaxDim} --tolerance ${Tolerance}) 
endforeach()
