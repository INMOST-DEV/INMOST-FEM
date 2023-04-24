#include "prob_args.h"
#include <sstream>

using namespace INMOST;

void InputArgs::printArgsDescr(std::ostream& out, const std::string& prefix){
    out << prefix << "  -ac, --axis_cnt  IVAL[3] <Sets the number of segments of the cube [0; 1]^3 partition along every coordinate axes, default=" << axis_sizes << ">" << "\n"
        << prefix << "  -t , --target    PATH    <Directory to save results, default=\"" << save_dir  << "\">" << "\n"
        << prefix << "  -nm, --name      STR     <Prefix for saved results, default=\"" << save_prefix << "\">" << "\n"
        << prefix << "  -db, --lnslvdb   FILE    <Specify linear solver data base, default=\"" << lin_sol_db << "\">\n"
        << prefix << "  -ls, --lnslv     STR[2]  <Set linear solver name and prefix, default=\"" << lin_sol_nm << "\" \"\">\n"
        << prefix << "  -q , --qorder    DVAL    <Set maximal quadrature formulas order, default=" << max_quad_order << ">\n";
}

void InputArgs::printArgsHelpMessage(std::ostream& out, const std::string& prefix){
    out << prefix << "Help message: " << "\n"
        << prefix << " Command line options: " << "\n";
    printArgsDescr(out, prefix);
    out << prefix << "  -h , --help              <Print this message and exit>" << "\n";    
}

void InputArgs::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "mesh axis partition = " << axis_sizes << " x " << axis_sizes << " x " << axis_sizes << "\n"
        << prefix << "save_dir  = \"" << save_dir << "\" save_prefix = \"" << save_prefix << "\"\n"
        << prefix << "linsol = \"" << lin_sol_nm << "\" prefix = \"" << lin_sol_prefix  << "\" database = \"" << lin_sol_db << "\"" << "\n"
        << prefix << "maximal quadrature order = " << max_quad_order << "\n";    
}

uint InputArgs::parseArg(int argc, char* argv[], bool print_messages){
    #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
    auto is_int = [](const std::string& s) -> std::pair<bool, int>{
        std::istringstream iss(s);
        int f = -1;
        iss >> f; 
        return {iss.eof() && !iss.fail(), f};    
    }; 
    uint i = 0;
    if (strcmp(argv[i], "-ac") == 0 || strcmp(argv[i], "--axis_cnt") == 0) {
        unsigned j = 0;
        for (; j < 1 && i+1 < static_cast<uint>(argc) && is_int(argv[i+1]).first; ++j)
            axis_sizes = is_int(argv[++i]).second;
        if (j != 1) throw std::runtime_error("Waited " + std::to_string(1) + " arguments for command \"" + std::string(argv[i-j]) + "\" but found only " + std::to_string(j));  
        ++i;  
    } else if (strcmp(argv[i], "-nm") == 0 || strcmp(argv[i], "--name") == 0) {
        GETARG(save_prefix = argv[++i];)
        ++i;
    } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--target") == 0) {
        GETARG(save_dir = argv[++i];)
        if (save_dir.back() != '/') save_dir += '/';
        ++i;
    } else if (strcmp(argv[i], "-db") == 0 || strcmp(argv[i], "--lnslvdb") == 0) {
        GETARG(lin_sol_db = argv[++i];)
        ++i;
    } else if (strcmp(argv[i], "-ls") == 0 || strcmp(argv[i], "--lnslv") == 0) {
        std::array<std::string*, 2> plp = {&lin_sol_nm, &lin_sol_prefix};
        unsigned j = 0;
        for (; j < plp.size() && i+1 < static_cast<uint>(argc); ++j)
            *(plp[j]) = argv[++i];
        ++i;    
    } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--qorder") == 0) {
        GETARG(max_quad_order = is_int(argv[++i]).second;)
        ++i;
    }
    #undef GETARG

    return i;
}

void InputArgs::setup_args(){
    if (!save_dir.empty()){
        struct stat sb;

        if (!(stat(save_dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))) {
            std::cerr << ("save_dir = \"" + save_dir + "\" is not exists") << std::endl;
            exit(-2);
        }
    }
    if (!save_dir.empty() && save_dir.back() != '/') save_dir += "/";
}

void InputArgs::parseArgs(int argc, char* argv[], bool print_messages){
    std::stringstream ss;
    printArgsHelpMessage(ss);
    for (int i = 1; i < argc; i++) {
        uint cnt = parseArg(argc - i, argv + i, print_messages);
        if (cnt == 0){
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0){
                if (print_messages) std::cout << ss.str() << std::endl;
                #if defined(USE_MPI)
                    int inited = 0;
                    MPI_Initialized(&inited);
                    if (inited) MPI_Finalize();
                #endif
                exit(0);
            } else {
                if (print_messages) {
                    std::cerr << "Faced unknown command \"" << argv[i] << "\"" << "\n";
                    std::cout << ss.str() << std::endl;
                }
                exit(-1);
            }
        }
        i += (cnt-1); 
    }

    setup_args();
}

uint InputArgs1Var::parseArg(int argc, char* argv[], bool print_messages){
    #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
    uint i = 0;
    if (strcmp(argv[i], "-sp") == 0 || strcmp(argv[i], "--fem_space") == 0){
        GETARG(USpace = argv[++i];)
        return i+1; 
    } else 
        return InputArgs::parseArg(argc, argv, print_messages);
    #undef GETARG   
}
void InputArgs1Var::print(std::ostream& out, const std::string& prefix) const{
    out << prefix << "USpace = " << USpace << "\n";
    InputArgs::print(out, prefix);
}
void InputArgs1Var::printArgsDescr(std::ostream& out, const std::string& prefix){
    out << prefix << "  -sp, --fem_space STR     <Set fem space for U variable (displacement), supported \"P0\", \"P1\", \"P2\", \"P3\", \"MINI1\" (=P1+B4), \"MINI2\"(=P2+B4), \"MINI3\"(=P3+B4), default=\"" << USpace << "\">\n";
    InputArgs::printArgsDescr(out, prefix);
}

uint InputArgs2Var::parseArg(int argc, char* argv[], bool print_messages){
    #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
    uint i = 0;
    if (strcmp(argv[i], "-us") == 0 || strcmp(argv[i], "--u_space") == 0){
        GETARG(USpace = argv[++i];)
        return i+1; 
    } else if (strcmp(argv[i], "-ps") == 0 || strcmp(argv[i], "--p_space") == 0){
        GETARG(PSpace = argv[++i];)
        return i+1; 
    } else 
        return InputArgs::parseArg(argc, argv, print_messages);
    #undef GETARG   
}
void InputArgs2Var::print(std::ostream& out, const std::string& prefix) const{
    out << prefix << "U_space = " << USpace << ", p_space = " << PSpace << "\n";
    InputArgs::print(out, prefix);
}
void InputArgs2Var::printArgsDescr(std::ostream& out, const std::string& prefix){
    out << prefix << "  -us, --u_space   STR     <Set fem space for U variable (displacement), supported \"P0\", \"P1\", \"P2\", \"P3\", \"MINI1\" (=P1+B4), \"MINI2\"(=P2+B4), \"MINI3\"(=P3+B4), default=\"" << USpace << "\">\n"
        << prefix << "  -ps, --p_space   STR     <Set fem space for p variable (displacement), supported \"P0\", \"P1\", \"P2\", \"P3\", \"MINI1\" (=P1+B4), \"MINI2\"(=P2+B4), \"MINI3\"(=P3+B4), default=\"" << PSpace << "\">\n";
    InputArgs::printArgsDescr(out, prefix);
}

void InputArgs::parseArgs_mpi(int* argc, char** argv[]){
    int pRank = 0;
    #if defined(USE_MPI)
        int inited = 0;
        MPI_Initialized(&inited);
        if (!inited) MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
    #endif
    parseArgs(*argc, *argv, pRank == 0);
}