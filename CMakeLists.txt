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

set(MaxDim "100")
set(Tolerance "1.0e-8")  # typically much better tolerance, but this is to signal actual badness

add_test(NAME smoke-mcheck COMMAND mcheck 40 50 30)
add_test(NAME smoke-mcheck-tolerance COMMAND python3  ../testsrc/mcheck.py --do-not-generate --tolerance ${Tolerance})

foreach(itr RANGE 1 10)
  add_test(NAME mcheck-${itr} COMMAND python3  ../testsrc/mcheck.py --N ${MaxDim} --tolerance ${Tolerance}) 
endforeach()

add_test(NAME smoke-niqcheck COMMAND niqcheck 50 1 20 1)

set(TestStages "25")
set(TestStageDims "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
foreach(TestStageDim ${TestStageDims})
  add_test(NAME niq-eq-${TestStageDim} COMMAND python3 ../testsrc/niqcheck.py --dmin ${TestStageDim} --dmax ${TestStageDim} --stages ${TestStages} --tolerance ${Tolerance})
endforeach()
