#include "mesh_gen.h"
#include <vector>
#include <algorithm>
#include <iostream>

using namespace INMOST;
/*      (4)*-------*(6)
          /|\     /|
         /   \   / |
        /  |  \ /  |
    (5)*-------*(7)|
       |   |   |   |
       |       |   |
       |   |   |   |
       |(0)*- -|- -*(2)
       |  / \  |  /
       |       | /
       |/     \|/
    (1)*-------*(3)      */
void CreateNWTetElements(Mesh *m, ElementArray<Node> verts)
{
    // Define prism faces assuming verts are numerated in the way presented above
    const INMOST_DATA_INTEGER_TYPE ne_face_nodes1[12] = {0,1,5,  5,1,3,   1,0,3, 3,0,5};
    const INMOST_DATA_INTEGER_TYPE ne_num_nodes1[4]   = {3,3,3,3};

    const INMOST_DATA_INTEGER_TYPE ne_face_nodes2[12] = {0,3,5, 0,7,3, 5,3,7, 0,5,7};
    const INMOST_DATA_INTEGER_TYPE ne_num_nodes2[4]   = {3,3,3,3};

    const INMOST_DATA_INTEGER_TYPE ne_face_nodes3[12] = {0,7,5, 4,5,7, 0,5,4, 0,4,7};
    const INMOST_DATA_INTEGER_TYPE ne_num_nodes3[4]   = {3,3,3,3};


    const INMOST_DATA_INTEGER_TYPE sw_face_nodes1[12] = {0,3,7, 2,7,3, 0,7,2, 0,2,3};
    const INMOST_DATA_INTEGER_TYPE sw_num_nodes1[4]   = {3,3,3,3};

    const INMOST_DATA_INTEGER_TYPE sw_face_nodes2[12] = {0,7,4, 0,2,7, 2,4,7, 0,4,2};
    const INMOST_DATA_INTEGER_TYPE sw_num_nodes2[4]   = {3,3,3,3};

    const INMOST_DATA_INTEGER_TYPE sw_face_nodes3[12] = {4,6,2, 6,7,2, 4,7,6, 4,2,7};
    const INMOST_DATA_INTEGER_TYPE sw_num_nodes3[4]   = {3,3,3,3};

    m->CreateCell(verts,ne_face_nodes1,ne_num_nodes1,4); // Create north-east tetrahedron cell
    m->CreateCell(verts,ne_face_nodes2,ne_num_nodes2,4); // Create north-east tetrahedron cell
    m->CreateCell(verts,ne_face_nodes3,ne_num_nodes3,4); // Create north-east tetrahedron cell
    m->CreateCell(verts,sw_face_nodes1,sw_num_nodes1,4); // Create south-west tetrahedron cell
    m->CreateCell(verts,sw_face_nodes2,sw_num_nodes2,4); // Create south-west tetrahedron cell
    m->CreateCell(verts,sw_face_nodes3,sw_num_nodes3,4); // Create south-west tetrahedron cell
}

///Create mesh for cubic domain [0, 1]^3
///@param nx, ny, nz are numbers of steps per axis
std::unique_ptr<Mesh> GenerateParallelepiped(INMOST_MPI_Comm comm, std::array<unsigned, 3> sizes, std::array<double, 3> A, std::array<double, 3> O)
{
    unsigned procs_per_axis[3] = {1,1,1};
    int rank,size;
    std::unique_ptr<Mesh> mptr(new Mesh()); // Create a mesh to be constructed
    Mesh* m = mptr.get();
    m->SetCommunicator(comm); // Set the MPI communicator, usually MPI_COMM_WORLD

#if defined(USE_MPI)
    MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);
#endif

    rank = m->GetProcessorRank(); // Get the rank of the current process
    size = m->GetProcessorsNumber(); // Get the number of processors used in communicator comm

    // Compute the configuration of processes connection
    {
        int divsize = size;
        std::vector<int> divs;
        while( divsize > 1 )
        {
            for(int k = 2; k <= divsize; k++)
                if( divsize % k == 0 )
                {
                    divs.push_back(k);
                    divsize /= k;
                    break;
                }
        }
        unsigned elements_per_procs[3] = {sizes[0],sizes[1],sizes[2]};
        for(std::vector<int>::reverse_iterator it = divs.rbegin(); it != divs.rend(); it++)
        {
            unsigned * max = std::max_element(elements_per_procs+0,elements_per_procs+3);
            procs_per_axis[max-elements_per_procs] *= *it;
            (*max) /= *it;
        }
    }

    //rank = proc_coords[2] * procs_per_axis[0] *procs_per_axis[1] + proc_coords[1] * procs_per_axis[0] + proc_coords[0];
    unsigned proc_coords[3] = {rank % procs_per_axis[0] , rank / procs_per_axis[0] % procs_per_axis[1], rank / (procs_per_axis[0] *procs_per_axis[1]) };

    unsigned localsize[3], localstart[3], localend[3];
    unsigned avgsize[3] =
            {
                    (unsigned)ceil((double)sizes[0]/procs_per_axis[0]),
                    (unsigned)ceil((double)sizes[1]/procs_per_axis[1]),
                    (unsigned)ceil((double)sizes[2]/procs_per_axis[2])
            };

    for(int j = 0; j < 3; j++)
    {
        localstart[j] = avgsize[j] * proc_coords[j];
        if( proc_coords[j] == procs_per_axis[j] - 1 )
            localsize[j] = sizes[j] - avgsize[j] * (procs_per_axis[j]-1);
        else localsize[j] = avgsize[j];
        localend[j] = localstart[j] + localsize[j];
    }

    // Create i-j-k structure of nodes
    ElementArray<Node> newverts(m);
    newverts.reserve(localsize[0]*localsize[1]*localsize[2]);

    for(unsigned i = localstart[0]; i <= localend[0]; i++)
        for(unsigned j = localstart[1]; j <= localend[1]; j++)
            for(unsigned k = localstart[2]; k <= localend[2]; k++)
            {
                Storage::real xyz[3];
                xyz[0] = i * (A[0] - O[0]) / (sizes[0]) + O[0];
                xyz[1] = j * (A[1] - O[1]) / (sizes[1]) + O[1];
                xyz[2] = k * (A[2] - O[2]) / (sizes[2]) + O[2];
                newverts.push_back(m->CreateNode(xyz)); // Create node in the mesh with index V_ID(i,j,k)
            }
#define V_ID(x, y, z) ((x-localstart[0])*(localsize[1]+1)*(localsize[2]+1) + (y-localstart[1])*(localsize[2]+1) + (z-localstart[2]))
    // Create i-j-k structure of elements
    for(unsigned i = localstart[0]+1; i <= localend[0]; i++)
        for(unsigned j = localstart[1]+1; j <= localend[1]; j++)
            for(unsigned k = localstart[2]+1; k <= localend[2]; k++)
            {
                // Create local array of eight nodes                           /*      (4)*-------*(6)  */
                // using representation on the right figure                    /*        /|      /|     */
                ElementArray<Node> verts(m);                                   /*       /       / |     */
                verts.push_back(newverts[V_ID(i - 1,j - 1, k - 1)]); // 0      /*      /  |    /  |     */
                verts.push_back(newverts[V_ID(i - 0,j - 1, k - 1)]); // 1      /*  (5)*-------*(7)|     */
                verts.push_back(newverts[V_ID(i - 1,j - 0, k - 1)]); // 2      /*     |   |   |   |     */
                verts.push_back(newverts[V_ID(i - 0,j - 0, k - 1)]); // 3      /*     |       |   |     */
                verts.push_back(newverts[V_ID(i - 1,j - 1, k - 0)]); // 4      /*     |   |   |   |     */
                verts.push_back(newverts[V_ID(i - 0,j - 1, k - 0)]); // 5      /*     |(0)*- -|- -*(2)  */
                verts.push_back(newverts[V_ID(i - 1,j - 0, k - 0)]); // 6      /*     |  /    |  /      */
                verts.push_back(newverts[V_ID(i - 0,j - 0, k - 0)]); // 7      /*     |       | /       */
                                                                               /*     |/      |/        */
                //                                                             /*  (1)*-------*(3)      */
                CreateNWTetElements(m,verts);
            }
#undef V_ID
    m->ResolveShared(); // Resolve duplicate nodes

    return mptr;
}

///Create mesh for domain [O[0], A[0]]x[O[1], A[1]]x[O[2], A[2]]
///@param nx, ny, nz are numbers of steps per axis
std::unique_ptr<Mesh> GenerateCube(INMOST_MPI_Comm comm, unsigned nx, unsigned ny, unsigned nz, double size){
    return GenerateParallelepiped(comm, {nx, ny, nz}, {size, size, size}, {0, 0, 0});
}

#if defined(USE_MPI)
#define BARRIER MPI_Barrier(MPI_COMM_WORLD);
#else
#define BARRIER
#endif

void RepartMesh(Mesh* m){
    int pRank = 0, pCount = 0;
#if defined(USE_MPI)
    MPI_Comm_rank(INMOST_MPI_COMM_WORLD, &pRank);
    MPI_Comm_size(INMOST_MPI_COMM_WORLD, &pCount);
    m->SetCommunicator(INMOST_MPI_COMM_WORLD);
#endif
    // mesh repartition
#ifdef USE_PARTITIONER
    if(pCount >1) {
        if (pRank == 0)    std::cout<<"bef part"<<std::endl;
        Partitioner *p = new Partitioner(m);
// #ifdef USE_PARTITIONER_PARMETIS
//         p->SetMethod(Partitioner::Parmetis, Partitioner::Partition);
// #elif USE_PARTITIONER_ZOLTAN
//         p->SetMethod(Partitioner::Zoltan, Partitioner::Partition);
// #else
        p->SetMethod(Partitioner::INNER_KMEANS, Partitioner::Partition);      
// #endif
        if (pRank  == 0)    std::cout<<"eval"<<std::endl;
        p->Evaluate();
        delete p;
        BARRIER

        // prior exchange ghost to get optimization in Redistribute
        m->ExchangeGhost(1, NODE); //<- required for parallel fem
        m->Redistribute();
        BARRIER;
        m->ExchangeGhost(1, NODE); //<- required for parallel fem
        m->ReorderEmpty(CELL | FACE | EDGE | NODE);
    }
#else
    if(pCount >1) {
        if (pRank == 0) std::cerr << "ERROR: Partition is not available in current INMOST build, try compile INMOST with -DUSE_PARTITIONER=ON flag" << std::endl;
        BARRIER;
        abort();
    }    
#endif

    if (pRank  == 0 && pCount > 1) { std::cout << " repartitioning success" << std::endl; }
    BARRIER

#ifdef USE_PARTITIONER
    for (int i = 0; i < pCount && pCount > 1; ++i) {
        if (pRank == i) {
            std::cout << pRank << ": Mesh info:"
                      << " #N " << m->NumberOfNodes()
                      << " #E " << m->NumberOfEdges()
                      << " #F " << m->NumberOfFaces()
                      << " #T " << m->NumberOfCells() << std::endl;
        }
        BARRIER;
    }
    if (pCount == 1){
        std::cout<< pRank << ": Mesh info:"
                      << " #N " << m->NumberOfNodes()
                      << " #E " << m->NumberOfEdges()
                      << " #F " << m->NumberOfFaces()
                      << " #T " << m->NumberOfCells() << std::endl;
    }
#endif
}
