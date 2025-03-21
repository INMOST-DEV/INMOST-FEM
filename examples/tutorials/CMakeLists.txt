add_executable(ex1 ex1.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex1 AniFem++ Fem_utils)

add_executable(ex2 ex2.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex2 AniFem++ Fem_utils)

add_executable(ex3 ex3.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex3 AniFem++ Fem_utils)

add_executable(ex4 ex4.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex4 AniFem++ Fem_utils)

add_executable(ex5 ex5.cpp prob_args.h prob_args.cpp)
target_link_libraries(ex5 AniFem++ Fem_utils)

add_executable(quadrature_function quadrature_function.cpp prob_args.h prob_args.cpp)
target_link_libraries(quadrature_function AniFem++ Fem_utils)

add_executable(react_diff1 react_diff1.cpp prob_args.h prob_args.cpp)
target_link_libraries(react_diff1 AniFem++ Fem_utils)

add_executable(react_diff2 react_diff2.cpp prob_args.h prob_args.cpp)
target_link_libraries(react_diff2 AniFem++ Fem_utils)

add_executable(stokes stokes.cpp prob_args.h prob_args.cpp)
target_link_libraries(stokes AniFem++ Fem_utils)

add_executable(unsteady_conv_dif unsteady_conv_dif.cpp prob_args.h prob_args.cpp)
target_link_libraries(unsteady_conv_dif AniFem++ Fem_utils)

add_executable(lin_elast lin_elast.cpp prob_args.h prob_args.cpp)
target_link_libraries(lin_elast AniFem++ Fem_utils)

add_executable(nonlin_elast nonlin_elast.cpp prob_args.h prob_args.cpp)
target_link_libraries(nonlin_elast AniFem++ Fem_utils)

add_executable(nonlin_elast_incompress nonlin_elast_incompress.cpp prob_args.h prob_args.cpp)
target_link_libraries(nonlin_elast_incompress AniFem++ Fem_utils)