/**
 * This program generates  and solves a finite element system for the Stokes problem
 *
 * \f[
 * \begin{aligned}
 *  -\mathrm{div}\ \nu &\mathrm{grad}\ u\ + &\mathrm{grad}\ p &= 0\   in\  \Omega  \\
 *   \mathrm{div}\ &               u    &                 &= 0\   in\  \Omega\\
 *                 &               u    &                 &= u_0\ on\  \Gamma_1\\
 *                 &               u    &                 &= 0\   on\  \Gamma_2\\
 *           &\frac{du}{d\mathbf{n}}\ - &               p &= 0\   on\  \Gamma_3\\
 * \end{aligned}
 * \f]
 *
 * where Omega is a backstep domain,            
 * Γ_1 is the side at x=0, Γ_3 is the side at x=1, 
 * and Γ_2 is the rest of the boundary. The non-homogeneous
 * boundary condition is 
 *
 *   u_0 = { 64*(y-0.5)*(1-y)*z*(1-z), 0, 0 }.
 * The user-defined coefficient
 *    nu(x)   - positive scalar tensor
 */

#include "prob_args.h"

using namespace INMOST;

std::unique_ptr<Mesh> generate_backstep_domain(unsigned nsteps){
    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, 2*nsteps, 2*nsteps, 2*nsteps);
    Mesh* m = mptr.get();
    m->BeginModification();
    for (auto c = m->BeginElement(NODE|EDGE|FACE|CELL); c != m->EndElement(); ++c){
        double xc[3] = {0, 0, 0};
        c->Centroid(xc);
        bool outside = true;
        for (int k = 0; k < 2; ++k)
            outside &= (xc[k] - 0.5 < -5*std::numeric_limits<double>::epsilon());
        if (outside)  
            m->Delete(c->GetHandle());  
    }
    m->ApplyModification();

    m->ResolveModification();
    m->EndModification();
    
    RepartMesh(m);
    m->AssignGlobalID(NODE|EDGE|FACE|CELL);
    
    return mptr;
}

int main(int argc, char* argv[]){
    InputArgs2Var p;
    std::string prob_name = "stokes";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);
    unsigned quad_order = p.max_quad_order;

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    std::unique_ptr<Mesh> mptr = generate_backstep_domain(p.axis_sizes);
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //generate FEM space from it's name
    FemSpace    UFem = choose_space_from_name(p.USpace)^3, 
                PFem = choose_space_from_name(p.PSpace);
    uint unf = UFem.dofMap().NumDofOnTet(), pnf = PFem.dofMap().NumDofOnTet();
    auto& u_dmap = UFem.dofMap(), &p_dmap = PFem.dofMap();
    auto mask = GeomMaskToInmostElementType(u_dmap.GetGeomMask() | p_dmap.GetGeomMask()) | FACE;

    INMOST::MarkerType boundary_face = m->CreateMarker();
    m->MarkBoundaryFaces(boundary_face);
     // Set boundary labels
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, mask, NONE, 1);
    for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it)
        it->Integer(BndLabel) = 0;
    for (auto f = m->BeginFace(); f != m->EndFace(); ++f) if (f->GetMarker(boundary_face)) {
        double xc[3] = {0, 0, 0};
        f->Centroid(xc);
        int lbl = 0;
        if (abs(xc[0] - 0) < 10*std::numeric_limits<double>::epsilon()) { //Γ_1
            lbl = 1 << 1;
        } else if (abs(xc[0] - 1) < 10*std::numeric_limits<double>::epsilon()){ //Γ_3
            lbl = 1 << 3;
        } else { // Γ_2
            lbl = 1 << 2;
        }
        f->Integer(BndLabel) = lbl;
        if (mask & NODE){
            auto nds = f->getNodes();
            for (uint ni = 0; ni != nds.size(); ++ni)
                nds[ni].Integer(BndLabel) |= lbl;
        }
        if (mask & EDGE){
            auto eds = f->getEdges();
            for (uint ei = 0; ei != eds.size(); ++ei)
                eds[ei].Integer(BndLabel) |= lbl;
        } 
    }
    m->ReleaseMarker(boundary_face,FACE);

    auto nu_tensor = [](const Coord<> &X, double *nu, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        nu[0] = 1;
        return Ani::TENSOR_SCALAR;
    };
    auto U_0 = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ 
        (void) dim; (void) user_data; 
        res[0] = 64*(X[1] - 0.5)*(1 - X[1])*X[2]*(1 - X[2]);
        res[1] = 0;
        res[2] = 0; 
        return 0;
    };

    // Define user structure to store input data used in local assembler
    struct ProbLocData{
        //save labels used for apply boundary conditions
        std::array<int, 4> nlbl = {0};
        std::array<int, 6> elbl = {0};
        std::array<int, 4> flbl = {0};
    };

    // Define memory requirements for elemental assembler
    PlainMemoryX<> mem_req;
    {
        auto grad_u = UFem.getOP(GRAD), div_u = UFem.getOP(DIV), iden_p = PFem.getOP(IDEN);
        mem_req = fem3Dtet_memory_requirements<DfuncTraits<TENSOR_SCALAR, true>>(grad_u, grad_u, quad_order);
        mem_req.extend_size(fem3Dtet_memory_requirements<DfuncTraits<TENSOR_NULL>>(iden_p, div_u, quad_order));
        mem_req.extend_size(UFem.interpolateByDOFs_mem_req());
    }
    PlainMemory<> shrt_req = mem_req.enoughPlainMemory();
    // define elemental assembler of local matrix and rhs
    std::function<void(const double**, double*, double*, double*, long*, void*)> local_assembler =
            [&U_0, &nu_tensor, order = quad_order, mem_req, shrt_req, unf, &UFem, pnf, &PFem](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data) -> void{
        PlainMemory<> _mem = shrt_req;
        _mem.ddata = w, _mem.idata = reinterpret_cast<int*>(iw);
        PlainMemoryX<> mem = mem_req;
        mem.allocateFromPlainMemory(_mem);
        uint nf = unf + pnf;
        DenseMatrix<> A(Adat, nf, nf), F(Fdat, nf, 1), B(w + shrt_req.dSize, nf, nf), G(w + shrt_req.dSize, nf, 1);
        A.SetZero(); F.SetZero(); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto grad_u = UFem.getOP(GRAD), div_u = UFem.getOP(DIV), iden_p = PFem.getOP(IDEN);
        fem3Dtet<DfuncTraits<TENSOR_SCALAR, true>>(XYZ, grad_u, grad_u, nu_tensor, B, mem, order);
        for(uint j = 0; j < unf; ++j)
            for(uint i = 0; i < unf; ++i)
                A(i, j) = B(i, j);
        fem3Dtet<DfuncTraits<TENSOR_NULL>>(XYZ, iden_p, div_u, TensorNull<>, B, mem, order);
        for(uint j = unf; j < nf; ++j)
            for(uint i = 0; i < unf; ++i) 
                A(j, i) = A(i, j) = -B(i, j - unf);  
        
        // read user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        //choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp1, sp2;
        static const int sz[3] = {4, 6, 4};
        const int* lbls[3] = {dat.nlbl.data(), dat.elbl.data(), dat.flbl.data()};
        for (int d = 0; d < 3; ++d)
            for (int i = 0; i < sz[d]; ++i)
                if (lbls[d][i] & (1 << 1))
                    sp1.set(d, i);
                else if (lbls[d][i] & (1 << 2))
                    sp2.set(d, i);  
        
        //set dirichlet condition
        if (!sp1.empty() || !sp2.empty()){
            ArrayView<> dof_values(w + shrt_req.dSize, nf);
            ArrayView<> udof_values(dof_values.data+0, unf);
            UFem.interpolateConstant(0.0, udof_values, sp2);
            UFem.interpolateByDOFs(XYZ, U_0, udof_values, sp1, mem);

            //Udofs have zero shift in vector of dofs for [U, p] variable, so
            std::array<uint, DofT::NGEOM_TYPES> U_shifts_on_elem = {0};
            //Construct DofMap on dofs of U variable inside vector [U, p]
            DofT::NestedDofMapView Ushifted(UFem.dofMap().target<>(), U_shifts_on_elem, 0U);
            applyDirByDofs(*(UFem.dofMap().target<>()), A, F, sp1|sp2, ArrayView<const double>(dof_values));
        }
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, geom_mask = UFem.dofMap().GetGeomMask()](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        if (geom_mask & DofT::NODE){
            for (unsigned i = 0; i < data.nlbl.size(); ++i) 
                data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);
        }
        if (geom_mask & DofT::EDGE){
            for (unsigned i = 0; i < data.elbl.size(); ++i) 
                data.elbl[i] = (*p.edges)[i].Integer(BndLabel);
        }
        {//required for Neumann and Robin BC
            for (unsigned i = 0; i < data.flbl.size(); ++i) 
                data.flbl[i] = (*p.faces)[i].Integer(BndLabel);
        }

        p.compute(args, &data);
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, unf+pnf, unf+pnf, shrt_req.dSize+(unf+pnf)*(unf+pnf), shrt_req.iSize));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*UFem.base()),
             Var1Helper = GenerateHelper(*PFem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        fed.PushTrialFunc(Var1Helper, "p");
        fed.PushTestFunc(Var1Helper, "phi_p");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();

    // Define tag to store result
    Tag u_tag = createFemVarTag(m, *u_dmap.target<>(), "u");
    Tag p_tag = createFemVarTag(m, *p_dmap.target<>(), "p");

    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    //assemble matrix and right-hand side
    TimerWrap m_timer_total; m_timer_total.reset();
    discr.Assemble(A, b);
    double total_assm_time = m_timer_total.elapsed();
    // Print Assembler timers
    if (pRank == 0) {
        std::cout << "#dofs = " << discr.m_enum.getMatrixSize() << std::endl; 
        std::cout << "Assembler timers:"
        #ifndef NO_ASSEMBLER_TIMERS
                  << "\n\tInit assembler: " << discr.GetTimeInitAssembleData() << "s"
                  << "\n\tInit guess    : " << discr.GetTimeInitValSet() << "s"
                  << "\n\tInit handler  : " << discr.GetTimeInitUserHandler() << "s"
                  << "\n\tLocal evaluate: " << discr.GetTimeEvalLocFunc() << "s"
                  << "\n\tLocal postproc: " << discr.GetTimePostProcUserHandler() << "s"
                  << "\n\tComp glob inds: " << discr.GetTimeFillMapTemplate() << "s"
                  << "\n\tGlobal remap  : " << discr.GetTimeFillGlobalStructs() << "s"
        #endif          
                  << "\n\tTotal         : " << total_assm_time << "s" 
                  << std::endl;
    }

    //setup linear solver and solve assembled system
    Solver solver(p.lin_sol_nm, p.lin_sol_prefix);
    solver.SetMatrix(A);
    solver.Solve(b, x);
    print_linear_solver_status(solver, prob_name, true);
    
    //copy result to the tag and save solution
    discr.SaveVar(x, 0, u_tag);
    discr.SaveVar(x, 1, p_tag);

    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}