//
// Created by Liogky Alexey on 24.03.2026.
//

/**
 * Here we show how to operate with laplace problem solver function make_laplace(...)
 * 
 * This program read mesh with label tag, mapping from label value to boundary condition
 * and solves the resulting laplace problem
 * 
 * Example of run:
 * 
 * ./laplace "--mesh_file" "ventrical_0.5.msh" "--label_tag" "GMSH_TAGS" "--only_bnd" "true" /
 * "--dirichlet" "{ 2:3->0.0; 3:4->1.0 }" /
 * "--res_tag" "Z" "--target" "vlapl.pvtk"
 * 
 * will read mesh "ventrical_0.5.msh" containing tag "GMSH_TAGS", 
 * values on tag "GMSH_TAGS" will be copyed to from all boundary elements to all its lower adjacency connections
 * (from Faces to Edges and from Edges to Nodes) to define boundary condition tag,
 * elements with label 2 will be considired as dirichlet boundary with value 0.0 
 * and elements with label 3 as dirichlet boundary with value 1.0,
 * result will be saved in "vlapl.pvtk" on tag "Z"
 **/

#include "anifem++/utils/utils.h"
#include "anifem++/interval/interval_map.h"

using namespace INMOST;

struct InputArgsProblem{
    std::string mesh_file = "mesh.pvtk";
    std::string mark_tag = "Label", result_tag = "U";
    Ani::interval_map<INMOST_DATA_INTEGER_TYPE, INMOST_DATA_REAL_TYPE> dirichlet_function;
    std::string save_file = "mesh_out.pvtk";
    std::string space = "P1";
    std::string lin_sol_db = "";
    int order = 5;
    bool is_boundary_tag = true;
    
    uint parseArg(int argc, char* argv[], bool print_messages = true) {
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        auto is_int = [](const std::string& s) -> std::pair<bool, int>{
            std::istringstream iss(s);
            int f = -1;
            iss >> f; 
            return {iss.eof() && !iss.fail(), f};    
        };
        auto get_bool = [](std::string s0)->bool{
            bool res = true;
            std::transform(s0.begin(), s0.end(), s0.begin(), [](char c){ return std::tolower(c); });
            if (s0 == "true") res = true;
            else if (s0 == "false") res = false;
            else {
                int r = stoi(s0);
                if (r == 0) res = false;
                else if (r == 1) res = true;
                else 
                    throw std::runtime_error("Expected bool value? but found = \"" + s0 + "\"");
            }
            return res;
        };
        uint i = 0;
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mesh_file") == 0) {
            GETARG(mesh_file = argv[++i];)
            ++i;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--label_tag") == 0) {
            GETARG(mark_tag = argv[++i];)
            ++i;
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--res_tag") == 0) {
            GETARG(result_tag = argv[++i];)
            ++i;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--target") == 0) {
            GETARG(save_file = argv[++i];)
            ++i;
        } else if (strcmp(argv[i], "-db") == 0 || strcmp(argv[i], "--lnslvdb") == 0) {
            GETARG(lin_sol_db = argv[++i];)
            ++i;
        } else if (strcmp(argv[i], "-sp") == 0 || strcmp(argv[i], "--fem_space") == 0){
            GETARG(space = argv[++i];)
            ++i; 
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--qorder") == 0) {
            GETARG(order = is_int(argv[++i]).second;)
            ++i;
        }  else if (strcmp(argv[i], "--only_bnd") == 0){
            GETARG( is_boundary_tag = get_bool(argv[++i]); )
            ++i;
        } else if (strcmp(argv[i], "-dc") == 0 || strcmp(argv[i], "--dirichlet") == 0){
            std::stringstream oss;
            if (argv[++i][0] != '{') throw std::runtime_error("Can't find open bracket \'{\' after " + std::string(argv[i-1]));
            bool find_end = false;
            for (uint j = 0; j < argc - i; ++j){
                auto len = strlen(argv[i+j]);
                oss << argv[i+j];
                if (len > 0 && argv[i+j][len - 1] == '}'){
                    i += j;
                    find_end = true;
                    break;
                }
            }
            if (!find_end) throw std::runtime_error("Can't find close bracket \'}\' for \"" + oss.str() + "\"");
            ++i;
            auto ss = oss.str();
            dirichlet_function = Ani::parse_interval_map<INMOST_DATA_INTEGER_TYPE, INMOST_DATA_REAL_TYPE>(ss.substr(1, ss.size()-2));
        }

        return i;
    }

    void print(std::ostream& out = std::cout, const std::string& prefix = "") const {
        out << prefix << "mesh_file = \"" << mesh_file << "\"" << "\n"
            << prefix << "label_tag = \"" << mark_tag << "\"" << "\n"
            << prefix << "only_bnd = " << (is_boundary_tag ? "true" : "false") << "\n"
            << prefix << "label -> dirichlet = { ";
        dirichlet_function.print(out);
        out << " }\n"
            << prefix << "result_tag = \"" << result_tag << "\"" << "\n"
            << prefix << "save_file = \"" << save_file << "\"" << "\n"
            << prefix << "fem_space = " << space << "\n"
            << prefix << "database = \"" << lin_sol_db << "\"" << "\n";
    }
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = ""){
        out << prefix << "  -m , --mesh_file FILE   <Set mesh file, default=\"" << mesh_file << "\">\n"
            << prefix << "  -l , --label_tag SVAL   <Set boundary label tag, default=\"" << mark_tag << "\">\n"
            << prefix << "       --only_bnd  BVAL   <Should consider label tag as label for full mesh boundary only and project data from faces if present, default=" << (is_boundary_tag ? "true" : "false") << ">\n"
            << prefix << "  -r , --res_tag   SVAL   <Set tag name to save laplace problem solution, default=\"" << result_tag << "\">\n"
            << prefix << "  -dc, --dirichlet LIST<IVAL:IVAL->DVAL>    <Set list of dirichlet labels, default=";
        dirichlet_function.print(out);
        out << ">\n"
            << prefix << "  -sp, --fem_space STR    <Set fem space for laplace variable, supported \"P0\", \"P1\", \"P2\", \"P3\", \"MINI1\" (=P1+B4), \"MINI2\"(=P2+B4), \"MINI3\"(=P3+B4), default=\"" << space << "\">\n"
            << prefix << "  -t , --target    FILE   <File name to save the result mesh, default=\"" << save_file << "\">\n"
            << prefix << "  -db, --lnslvdb   FILE   <Specify linear solver data base, default=\"" << lin_sol_db << "\">\n"
            << prefix << "  -q , --qorder    DVAL    <Set maximal quadrature formulas order, default=" << order << ">\n";
    }
    void printArgsHelpMessage(std::ostream& out = std::cout, const std::string& prefix = ""){
        out << prefix << "Help message: " << "\n"
            << prefix << " Command line options: " << "\n";
        printArgsDescr(out, prefix);
        out << prefix << "  -h , --help              <Print this message and exit>" << "\n";    
    }

    void parseArgs(int argc, char* argv[], bool print_messages = true){
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

        // setup_args();
    }
    void parseArgs_mpi(int* argc, char** argv[]){
        int pRank = 0;
        #if defined(USE_MPI)
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited) MPI_Init(argc, argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
        #endif
        parseArgs(*argc, *argv, pRank == 0);
    }
    // void setup_args(){}
};

bool is_system_inmost_element_name(const std::string& name);
void print_mesh_content(Mesh* m, bool with_statistics = true, bool hide_internal_tags = true);
Tag spread_boundary_tag(Mesh& m, const Ani::FemSpace& fem, Tag Lbl, std::function<bool(long)> is_spread_val, std::string res_tag_name);

int main(int argc, char* argv[]){
    int pRank = 0, pCount = 1;
#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(INMOST_MPI_COMM_WORLD, &pRank);
#endif
    if (pRank == 0){
        for (int i = 0; i < argc; ++i)
            std::cout << "\"" << argv[i] << "\"" << ((i == argc-1) ? "" : " ");// << std::endl;
        std::cout << "\n" << std::endl;
    }
    InputArgsProblem p;
    p.parseArgs_mpi(&argc, &argv);
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);
    if (pRank == 0) p.print();
    if (pRank == 0){
        std::ifstream f(p.mesh_file);
        if (!f.is_open())
            throw std::runtime_error("File \"" + p.mesh_file + "\" do not exists");
    }
    {
        Mesh m("m");
        if (m.isParallelFileFormat(p.mesh_file))
            m.Load(p.mesh_file);
        else if (pRank == 0)
            m.Load(p.mesh_file);
        RepartMesh(&m, pCount > 1);
        {
            bool on_every_processor = false;
            bool ignore_system_tags = true;
            for (int p = 0, pcnt = on_every_processor ? pCount : 1; p < pcnt; ++p){
                if (p == pRank) print_mesh_content(&m, on_every_processor, ignore_system_tags);
                BARRIER;
            }
        }
        print_mesh_sizes(&m);
        Tag Lbl = m.GetTag(p.mark_tag);
        if (!Lbl.isValid())
            throw std::runtime_error("Mesh have not tag \"" + p.mark_tag + "\"");
        if (Lbl.GetDataType() != DATA_INTEGER)
            throw std::runtime_error("Mesh tag \"" + p.mark_tag + "\" should have integer type");
        
        std::vector<std::string> main_tag_names;
        m.ListTagNames(main_tag_names);
        {
            auto tmp = std::move(main_tag_names);
            main_tag_names.reserve(tmp.size());
            for (auto& t: tmp) if (!is_system_inmost_element_name(t))
                main_tag_names.push_back(t);
        }

        Ani::FemSpace fem = choose_space_from_name(p.space);
        TagRealArray u0 = createFemVarTag(&m, *fem.dofMap().target<>(), p.result_tag);
        main_tag_names.push_back(u0.GetTagName());

        if (p.is_boundary_tag)  // Compute dirichlet mesh elements
            Lbl = spread_boundary_tag(m, fem, Lbl, [&dir_map = p.dirichlet_function](long v) -> bool { return dir_map.contains(v); }, "TMP_" + p.mark_tag);
        
        {   // Print number of dirichlet elements 
            INMOST::Storage::integer E[8]{0, 0, 0, 0, 0, 0, 0, 0};
            for (auto it = m.BeginElement(NODE|EDGE|FACE|CELL); it != m.EndElement(); ++it) if (it->HaveData(Lbl)) {
                auto dim = it->GetElementDimension();
                E[dim]++;
                auto l = it->Integer(Lbl);
                if (p.dirichlet_function.contains(l))
                    E[dim + 4]++;
            }
            m.Integrate(E, 8);
            if (pRank == 0) std::cout << "Labled part: #N = " << E[0] << " #E = " << E[1] << " #F = " << E[2] << " #T = " << E[3] << std::endl;
            if (pRank == 0) std::cout << "Dirichlet part: #N = " << E[4] << " #E = " << E[5] << " #F = " << E[6] << " #T = " << E[7] << std::endl;
        }

        auto is_dirichlet = [&Lbl, &p](INMOST::Element e) -> bool {
            if (!e.HaveData(Lbl)) return false;
            auto l = e.Integer(Lbl);
            return p.dirichlet_function.contains(l);
        };
        auto dirichlet_func = [&Lbl, &p](INMOST::Element e, std::array<double, 3> X, Ani::ArrayView<> dirichlet_value_out) -> bool {
            if (!e.HaveData(Lbl)) 
                return false;
            auto l = e.Integer(Lbl);
            auto it = p.dirichlet_function.find(l);
            if (it == p.dirichlet_function.end()) return false;
            auto v = it->second;
            dirichlet_value_out[0] = v;
            return true;
        };
        make_laplace(&m, fem, u0, is_dirichlet, dirichlet_func, p.order);
        if (p.is_boundary_tag) 
            m.DeleteTag(Lbl);

        if (pRank == 0) std::cout << "Save mesh into \"" << p.save_file << "\"" << std::endl;
        for (auto& t: main_tag_names)
            m.SetFileOption("Tag:" + t, "saveonly");
        m.Save(p.save_file);
    }
    
    InmostFinalize();
    
    return 0;
}

bool is_system_inmost_element_name(const std::string& name){
    return ((name.size() > 10 && name.substr(0, 10) == "PROTECTED_")
            || (name.size() > 10 && name.substr(0, 10) == "TEMPORARY_")
            || (name.size() > 18 && name.substr(0, 18) == "IGlobEnumeration::")
            || name == "OWNER_PROCESSOR" || name == "TEMPORARY_NODE_ID"
            || name == "BRIDGE" || name == "PROCESSORS_LIST"
            || name == "LAYERS" || name == "GLOBAL_ID" || (name.size() > 0 && name[0] == '_')
           );
}
template<typename IsDefined>
inline std::string InmostIsDefinedToString(IsDefined check_defined){
    std::stringstream oss;
    bool type_printed = false;
    for (ElementType e = NODE; e <= MESH; e = NextElementType(e)) if (check_defined(e)){
        oss << (type_printed ? "|" : "") << ElementTypeName(e);
        type_printed = true;
    }
    if (!type_printed)
        oss << "NONE";
    return oss.str();
};

void print_mesh_content(Mesh* m, bool with_statistics, bool hide_internal_tags){
    auto is_system_name = [](const std::string& name){ return is_system_inmost_element_name(name); };
    struct Table{
        std::vector<std::string> head;
        std::vector<std::vector<std::string>> fields;
        std::vector<std::string> delimiters;
        void print(std::ostream& out, std::string prefix = "", bool print_head = true){
            std::vector<std::size_t> field_sizes(head.size());
            for (std::size_t i = 0; i < head.size(); ++i) field_sizes[i] = head[i].size();
            for (auto& field: fields){
                if (field.size() > field_sizes.size())
                    field_sizes.resize(field.size());
                for (std::size_t i = 0; i < field.size(); ++i) field_sizes[i] = std::max(field[i].size(), field_sizes[i]);
            }
            auto print_line = [&](const std::vector<std::string>& line){
                out << prefix;
                for (std::size_t i = 0; i < field_sizes.size(); ++i){
                    std::string word = (i < line.size()) ? line[i] : (field_sizes[i] > 0 ? "-" : "");
                    word.resize(field_sizes[i], ' ');
                    std::string delimiter = (i+1 == field_sizes.size()) ? std::string("\n") : ((delimiters.size() > i) ? delimiters[i] : (delimiters.empty() ? std::string(" ") : delimiters.back()));
                    out << word << delimiter;
                }
            };
            if (print_head) print_line(head);
            for (auto& field: fields)
                print_line(field);
        }
    };
    const char* elem_mark[6]{"NODE", "EDGE", "FACE", "CELL", "ESET", "TAG"};
    std::cout << "Mesh \"" + m->GetMeshName() << "\"{\n";
    if (with_statistics){
        int nN = m->NumberOfNodes(), nE = m->NumberOfEdges(), nF = m->NumberOfFaces(), nC = m->NumberOfCells();
        int nS = m->NumberOfSets(), nT = m->NumberOfTags();
        // std::cout << "  #N = " << nN << " #E = " << nE << " #F = " << nF << " #C = " << nC << "\n";
        int elems[4]{nN, nE, nF, nC}, owned_elems[4]{0, 0, 0, 0}, ghost_elems[4]{0, 0, 0, 0};
        for (auto e = m->BeginElement(NODE|EDGE|FACE|CELL); e != m->EndElement(); ++e){
            if (e->GetStatus() == Element::Ghost) ghost_elems[e->GetElementNum()]++;
            else if (e->GetStatus() == Element::Owned) owned_elems[e->GetElementNum()]++;
        }
        for (int i = 0; i < 4; ++i){
            std::cout << "  #" << elem_mark[i] << " = " << elems[i];
            if (m->GetProcessorsNumber() > 1)
                std::cout << "(= " <<  owned_elems[i] << "+" <<  (elems[i] - ghost_elems[i] - owned_elems[i]) << "+" << ghost_elems[i] << ")";
        }
        std::cout << "  #" << elem_mark[4] << " = " << nS;
        std::cout << "  #" << elem_mark[5] << " = " << nT;
        std::cout << "\n";
    }
    std::vector<std::string> tag_names;
    m->ListTagNames(tag_names);
    int print_tags_count = 0;
    for (const auto& tname: tag_names) if (!hide_internal_tags || !is_system_name(tname)){ print_tags_count++; }
    if (print_tags_count > 0){
        std::cout << "  Tags[" << print_tags_count << "]:{\n";
        Table tab;
        tab.head = std::vector<std::string>{"NAME", "DATA_TYPE", "ETYPE", "ESPARSE", "SIZE"};
        tab.delimiters = std::vector<std::string>{": ", " "};
        for (const auto& tname: tag_names) if (!hide_internal_tags || !is_system_name(tname)){
            std::vector<std::string> field;
            Tag t = m->GetTag(tname);
            field.push_back("\"" + tname + "\"");
            field.push_back(std::string(DataTypeName(t.GetDataType())));
            field.push_back(InmostIsDefinedToString([t](INMOST::ElementType e){return t.isDefined(e); }));
            field.push_back(InmostIsDefinedToString([t](INMOST::ElementType e){return t.isSparse(e); }));
            field.push_back((t.GetSize() != ENUMUNDEF) ? std::to_string(t.GetSize()) : std::string("VARSIZED"));
            tab.fields.emplace_back(std::move(field));
        }
        tab.print(std::cout, "    ", true);
        std::cout << "  }\n";
    }
    int print_set_count = 0;
    if (m->NumberOfSets() > 0)
        for (auto s = m->BeginSet(); s != m->EndSet(); ++s) if (!hide_internal_tags || !is_system_name(s->GetName())){ print_set_count++; }
    if (print_set_count > 0){
        std::cout << "  Sets[" << print_set_count << "]:{\n";
        Table tab;
        tab.head = std::vector<std::string>{"NAME", "#N", "#E", "#F", "#C", "#S", "#M", "#CHILDS", "#SUBLINGS"};
        tab.delimiters = std::vector<std::string>{": ", " "};
        for (auto s = m->BeginSet(); s != m->EndSet(); ++s) if (!hide_internal_tags || !is_system_name(s->GetName())){
            std::vector<std::string> field(6+3);
            field[0] = "\"" + s->GetName() + "\"";
            int elems[6]{0, 0, 0, 0, 0, 0};
            for (auto it = s->Begin(); it < s->End(); ++it){
                elems[it->GetElementNum()]++;
            }
            for (int i = 0; i < 6; ++i)
                field[1+i] = std::to_string(elems[i]);
            field[7] = std::to_string(s->CountChildren());
            field[8] = std::to_string(s->CountSiblings());

            tab.fields.emplace_back(std::move(field));
        }
        tab.print(std::cout, "    ", true);
        std::cout << "  }\n";
    }
    std::cout << "  TagOptions:{\n";
    const static std::vector<std::string> s_options{"nosave", "noload", "noderivatives", "loadonly", "saveonly"};
    for (auto opt_type: s_options){
        auto opts = m->TagOptions(opt_type);
        std::string opt_type_x = "\"" + opt_type + "\"";
        opt_type_x.resize(15, ' ');
        if (opts.empty())
            std::cout << "    " << opt_type_x << " : NONE\n";
        else{
            std::cout << "    " << opt_type_x << " : { ";
            for (auto opt: opts)
                std::cout << "\"" << opt << "\", ";
            std::cout << "}\n";
        }
    }
    std::cout << "  }\n";
    const static std::vector<std::string> s_foptions{"VERBOSITY", "VTK_GRID_DIMS", "VTK_OUTPUT_FACES", "ECL_SPLIT_GLUED", "ECL_PROJECT_PERM", "ECL_COMPUTE_TRAN", "ECL_DEGENERATE", "ECL_TOPOLOGY", "ECL_PARALLEL_READ", "PMF_DUP_GID"};
    std::cout << "  FileOptions:{\n";
    for (auto opt_type: s_foptions){
        auto opt = m->GetFileOption(opt_type);
        if (!opt.empty())
            std::cout << "    \"" << opt_type << "\" : \"" << opt << "\"\n";
    }
    std::cout << "  }\n";
    std::cout << "}\n";
    std::cout << std::endl;
}

Tag spread_boundary_tag(Mesh& m, const Ani::FemSpace& fem, Tag Lbl, std::function<bool(long)> is_spread_val, std::string res_tag_name){
    auto fmask = Ani::GeomMaskToInmostElementType(fem.dofMap().GetGeomMask());
    auto tmask = (Lbl.isDefined(NODE) ? NODE : NONE) | (Lbl.isDefined(EDGE) ? EDGE : NONE) | (Lbl.isDefined(FACE) ? FACE : NONE) | (Lbl.isDefined(CELL) ? CELL : NONE) | (Lbl.isDefined(MESH) ? MESH : NONE);
    auto bmrk = m.CreateMarker();
    int pRank = m.GetProcessorRank();
    for (auto it = m.BeginElement(NODE|EDGE); it != m.EndElement(); ++it) it->RemMarker(bmrk);
    m.MarkBoundaryFaces(bmrk);
    for (auto f = m.BeginFace(); f != m.EndFace(); ++f) if (f->GetMarker(bmrk)){
        auto eds = f->getEdges();
        for (std::size_t ei = 0; ei < eds.size(); ++ei)
            eds[ei].SetMarker(bmrk);
        auto nds = f->getNodes();
        for (std::size_t ni = 0; ni < nds.size(); ++ni)
            nds[ni].SetMarker(bmrk);
    }
    bool print_boundary_statistics = true;
    if (print_boundary_statistics) {
        INMOST::Storage::integer E[3]{0, 0, 0};
        for (auto it = m.BeginElement(NODE|EDGE|FACE); it != m.EndElement(); ++it) if (it->GetMarker(bmrk)) 
            E[it->GetElementDimension()]++;
        m.Integrate(E, 3);
        if (pRank == 0) std::cout << "Mesh boundary: #N = " << E[0] << " #E = " << E[1] << " #F = " << E[2] << std::endl;
    }
    Tag ELbl = m.CreateTag(res_tag_name, DATA_INTEGER, fmask, fmask, 1);
    for (auto it = m.BeginElement(tmask & fmask); it != m.EndElement(); ++it) if (it->HaveData(Lbl)) 
        it->Integer(ELbl) = it->Integer(Lbl);
    for (ElementType et = EDGE, et_low = (NODE & fmask); et != LastElementType(); ){
        for (auto it = m.BeginElement(et); it != m.EndElement(); ++it) if (it->GetMarker(bmrk) & it->HaveData(Lbl)){
            auto val = it->Integer(Lbl);
            if (!is_spread_val(val)) continue;
            if (et_low & CELL){
                auto cs = it->getCells();
                for (std::size_t ci = 0; ci < cs.size(); ++ci) if (!cs[ci].HaveData(ELbl)){
                    cs[ci].Integer(ELbl) = val;
                }
            }
            if (et_low & FACE){
                auto fcs = it->getFaces();
                for (std::size_t fi = 0; fi < fcs.size(); ++fi) if (!fcs[fi].HaveData(ELbl)){
                    fcs[fi].Integer(ELbl) = val;
                }
            }
            if (et_low & EDGE){
                auto eds = it->getEdges();
                for (std::size_t ei = 0; ei < eds.size(); ++ei) if (!eds[ei].HaveData(ELbl)){
                    eds[ei].Integer(ELbl) = val;
                }
            }
            if (et_low & NODE){
                auto nds = it->getNodes();
                for (std::size_t ni = 0; ni < nds.size(); ++ni) if (!nds[ni].HaveData(ELbl)){
                    nds[ni].Integer(ELbl) = val;
                }
            }
        }
        et_low = (et_low | et) & fmask;
        et = NextElementType(et);
    }
    for (auto it = m.BeginElement(NODE|EDGE|FACE); it != m.EndElement(); ++it) it->RemMarker(bmrk);
    m.ReleaseMarker(bmrk);
    return ELbl;
}