//
// Created by Liogky Alexey on 05.04.2022.
//

#include "inmost.h"
#include "example_common.h"
#include "mesh_gen.h"
#include "anifem++/inmost_interface/fem.h"
#include <numeric>
#include <sys/stat.h>

// Class for parsing command line args
struct InputArgs{
    std::array<unsigned, 3> axis_sizes = {8, 8, 8};
    std::string lin_sol_db = "";
    std::string save_dir = "", save_prefix = "problem_out";
    std::string lin_sol_nm = "inner_mptiluc", lin_sol_prefix = "";
    std::string USpace = "P1";
    uint max_quad_order = 2;

    static void printArgsHelpMessage(std::ostream& out = std::cout, const std::string& prefix = "");
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const;
    friend std::ostream& operator<<(std::ostream& out, const InputArgs& p){ return p.print(out), out; }
    void parseArgs(int argc, char* argv[], bool print_messages = true);
    void parseArgs_mpi(int* argc, char** argv[]);
};