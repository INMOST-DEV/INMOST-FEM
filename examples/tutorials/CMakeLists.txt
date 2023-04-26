add_executable(ex1 ex1.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex1 AniFem++ example_com)

add_executable(ex2 ex2.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex2 AniFem++ example_com)

add_executable(ex3 ex3.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex3 AniFem++ example_com)

add_executable(ex4 ex4.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex4 AniFem++ example_com)

add_executable(react_diff1 react_diff1.cpp prob_args.h prob_args.cpp)
target_link_libraries(react_diff1 AniFem++ example_com)

add_executable(react_diff2 react_diff2.cpp prob_args.h prob_args.cpp)
target_link_libraries(react_diff2 AniFem++ example_com)

add_executable(stokes stokes.cpp prob_args.h prob_args.cpp)
target_link_libraries(stokes AniFem++ example_com)