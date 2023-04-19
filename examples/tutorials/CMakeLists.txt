add_executable(ex1 ex1.cpp cmd_ex1_3.cpp cmd_ex1_3.h)
target_link_libraries(ex1 AniFem++ example_com)

add_executable(ex2 ex2.cpp cmd_ex1_3.cpp cmd_ex1_3.h)
target_link_libraries(ex2 AniFem++ example_com)

add_executable(ex3 ex3.cpp cmd_ex1_3.cpp cmd_ex1_3.h)
target_link_libraries(ex3 AniFem++ example_com)

add_executable(ex4 ex4.cpp cmd_ex4.cpp cmd_ex4.h)
target_link_libraries(ex4 AniFem++ example_com)