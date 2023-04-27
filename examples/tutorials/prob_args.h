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
    unsigned axis_sizes = 8;
    std::string lin_sol_db = "";
    std::string save_dir = "./", save_prefix = "problem_out";
    std::string lin_sol_nm = "inner_mptiluc", lin_sol_prefix = "";
    uint max_quad_order = 2;

    virtual uint parseArg(int argc, char* argv[], bool print_messages = true);
    virtual void print(std::ostream& out = std::cout, const std::string& prefix = "") const;

    friend std::ostream& operator<<(std::ostream& out, const InputArgs& p);
    void parseArgs(int argc, char* argv[], bool print_messages = true);
    void parseArgs_mpi(int* argc, char** argv[]);

protected:
    virtual void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "");
    virtual void setup_args();
    void printArgsHelpMessage(std::ostream& out = std::cout, const std::string& prefix = "");
};

inline std::ostream& operator<<(std::ostream& out, const InputArgs& p){ return p.print(out), out; }

struct InputArgs1Var: public InputArgs{
    std::string USpace = "P1";

    uint parseArg(int argc, char* argv[], bool print_messages = true) override ;
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override;

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override;
};

struct InputArgs2Var: public InputArgs{
    std::string USpace = "P2", PSpace = "P1";

    uint parseArg(int argc, char* argv[], bool print_messages = true) override ;
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override;

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override;
};