#include "cmd_ex1_3.h"

using namespace INMOST;

void InputArgs::printArgsHelpMessage(std::ostream& out, const std::string& prefix){
    out << prefix << "Help message: " << "\n"
        << prefix << " Command line options: " << "\n"
        << prefix << "  -sz, --sizes     IVAL[3] <Sets the number of segments of the cube [0; 1]^3 partition along the coordinate axes Ox, Oy, Oz, default=8 8 8>" << "\n"
        << prefix << "  -t , --target    PATH    <Directory to save results, default=\"\">" << "\n"
        << prefix << "  -nm, --name      STR     <Prefix for saved results, default=\"problem_out\">" << "\n"
        << prefix << "  -db, --lnslvdb   FILE    <Specify linear solver data base, default=\"\">\n"
        << prefix << "  -ls, --lnslv     STR[2]  <Set linear solver name and prefix, default=\"inner_mptiluc\" \"\">\n"
        << prefix << "  -h , --help              <Print this message and exit>" << "\n";
}
void InputArgs::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "mesh axis partition = " << axis_sizes[0] << " x " << axis_sizes[1] << " x " << axis_sizes[2] << "\n"
        << prefix << "save_dir  = \"" << save_dir << "\" save_prefix = \"" << save_prefix << "\"\n"
        << prefix << "linsol = \"" << lin_sol_nm << "\" prefix = \"" << lin_sol_prefix  << "\" database = \"" << lin_sol_db << "\"" << "\n";     
}
void InputArgs::parseArgs(int argc, char* argv[], bool print_messages){
    #define GETARG(X)   if (i+1 < argc) { X }\
                    else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
    auto is_double = [](const std::string& s) -> std::pair<bool, double>{
        std::istringstream iss(s);
        double f = NAN;
        iss >> f; 
        return {iss.eof() && !iss.fail(), f};    
    };       
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if (print_messages) printArgsHelpMessage();
            #if defined(USE_MPI)
                int inited = 0;
                MPI_Initialized(&inited);
                if (inited) MPI_Finalize();
            #endif
            exit(0);
        } else if (strcmp(argv[i], "-sz") == 0 || strcmp(argv[i], "--sizes") == 0) {
            unsigned j = 0;
            for (; j < axis_sizes.size() && i+1 < argc && is_double(argv[i+1]).first; ++j)
                axis_sizes[j] = is_double(argv[++i]).second;
            if (j != axis_sizes.size()) throw std::runtime_error("Waited " + std::to_string(axis_sizes.size()) + " arguments for command \"" + std::string(argv[i-j]) + "\" but found only " + std::to_string(j));    
            continue;
        } else if (strcmp(argv[i], "-nm") == 0 || strcmp(argv[i], "--name") == 0) {
            GETARG(save_prefix = argv[++i];)
            continue;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--target") == 0) {
            GETARG(save_dir = argv[++i];)
            if (save_dir.back() != '/') save_dir += '/';
            continue;
        } else if (strcmp(argv[i], "-db") == 0 || strcmp(argv[i], "--lnslvdb") == 0) {
            GETARG(lin_sol_db = argv[++i];)
            continue;
        } else if (strcmp(argv[i], "-ls") == 0 || strcmp(argv[i], "--lnslv") == 0) {
            std::array<std::string*, 2> plp = {&lin_sol_nm, &lin_sol_prefix};
            unsigned j = 0;
            for (; j < plp.size() && i+1 < argc; ++j)
                *(plp[j]) = argv[++i];    
            continue;
        } else {
            if (print_messages) {
                std::cerr << "Faced unknown command \"" << argv[i] << "\"" << "\n";
                printArgsHelpMessage();
            }
            exit(-1);
        }
    }
    #undef GETARG
    if (!save_dir.empty()){
        struct stat sb;

        if (!(stat(save_dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))) {
            std::cerr << ("save_dir = \"" + save_dir + "\" is not exists") << std::endl;
            exit(-2);
        }
    }
    if (!save_dir.empty() && save_dir.back() != '/') save_dir += "/";
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