add_executable(diffusion diffusion.cpp)
target_link_libraries(diffusion AniFem++ example_com)

add_executable(dyn_diffusion dyn_diffusion.cpp)
target_link_libraries(dyn_diffusion AniFem++ example_com stdc++fs)