#include <gtest/gtest.h>
#include <algorithm>
#include "anifem++/fem/tetdofmap.h"

TEST(AniInterface, TetDofMap){
    using namespace Ani;
    using namespace Ani::DofT;
    using std::array;
    using std::vector;
    using std::map;
    struct dof_map
    {
        map<unsigned, vector<vector<uint>> > m_odfs{
            {0, vector<vector<uint>>(4)}, //node_odf
            {1, vector<vector<uint>>(6)}, //edge_unorient_odf
            {2, vector<vector<uint>>(6)}, //edge_orient_odf
            {3, vector<vector<uint>>(4)}, //face_unorient_odf
            {4, vector<vector<uint>>(4)}, //face_orient_odf
            {5, vector<vector<uint>>(1)}, //cell_odf
        };
        uint max_dof = 0;
        
        dof_map& set_dofs(std::array<uint, 6> dofs){
            for (uint i = 0; i < dofs.size(); ++i){
                for (uint l = 0; l < m_odfs[i].size(); ++l)
                    for (uint k = 0; k < dofs[i]; ++k){
                        m_odfs[i][l].push_back(max_dof++);
                    }
            }
            return *this;
        }

        dof_map(std::array<uint, 6> dofs = std::array<uint, 6>{0, 0, 0, 0, 0, 0}){
            set_dofs(dofs);
        }

        std::vector<LocalOrder> getBySparsity(const TetGeomSparsity& sp, bool preferGeomOrdering = false){
            vector<LocalOrder> res;
            array<std::array<int, 2>, 4> elem_num = {std::array<int, 2>{0, 1}, {1, 3}, {3, 5}, {5, 6}};
            for (auto pos = sp.beginPos(); pos != sp.endPos(); pos = sp.nextPos(pos)){
                for (int num = elem_num[pos.elem_dim][0]; num < elem_num[pos.elem_dim][1]; ++num){
                    auto& q = m_odfs[static_cast<unsigned>(num)][pos.elem_num];
                    for (uint i = 0; i < q.size(); ++i)
                        res.push_back(LocalOrder(q[i], NumToGeomType(num), pos.elem_num, i));
                }
            }
            if (!preferGeomOrdering)
                std::sort(res.begin(), res.end(), [](const auto& a, const auto& b){ return a.gid < b.gid; });
            return res;    
        }
    };
    auto printlo = [](const LocalOrder& lo) -> std::ostream& {
        return std::cout << lo.gid << ": { " << GeomTypeToNum(lo.etype) << ", " << static_cast<int>(lo.nelem) << ", " << lo.leid << "}";
    };
    auto check_is_same = [&printlo](std::vector<LocalOrder> mv, std::vector<LocalOrder> mtv, bool reorder = false) -> bool{
        bool isSame = true;
        if (reorder){
            std::sort(mv.begin(), mv.end(), [](const auto& a, const auto& b){ return a.gid < b.gid; });
            std::sort(mtv.begin(), mtv.end(), [](const auto& a, const auto& b){ return a.gid < b.gid; });
        }
        if (mtv.size() != mv.size()) isSame = false;
        else {
            // for (uint i = 0; i < mtv.size(); ++i) { printlo(mv[i]) << "\t"; printlo(mtv[i]) << std::endl; } 
            // std::cout << std::endl;

            for (uint i = 0; i < mtv.size() && isSame; ++i)
                if (mtv[i].getGeomOrder() != mv[i].getGeomOrder() || mtv[i].getTetOrder() != mv[i].getTetOrder()){
                    isSame = false;
                    printlo(mv[i]) << " vs "; printlo(mtv[i]) << std::endl; 
                }    
        }
        return isSame;
    };
    auto test_full_bypass = [&](auto& m, dof_map mt, std::string err_msg){
        std::vector<LocalOrder> mv, mtv;
        TetGeomSparsity sp; sp.setCell(true);
        mtv = mt.getBySparsity(sp, false);
        for (auto it = m.begin(); it != m.end(); ++it){
            mv.push_back(*it);
        }
        bool isSame = check_is_same(mv, mtv, false);
        EXPECT_TRUE(isSame) << err_msg;
    };
    auto test_ordering = [&](TetGeomSparsity sp, auto& m, dof_map mt, std::string err_msg){
        bool isSame = true;
        std::vector<LocalOrder> mv, mtv;
        mtv = mt.getBySparsity(sp, false);
        for (auto it = m.beginBySparsity(sp, false); it != m.endBySparsity(); ++it){
            // printlo(*it) << std::endl;  
            mv.push_back(*it);
        }
        isSame = check_is_same(mv, mtv, true);
        EXPECT_TRUE(isSame) << err_msg;

        mv.clear(); mtv.clear();
        mtv = mt.getBySparsity(sp, true);
        for (auto it = m.beginBySparsity(sp, true); it != m.endBySparsity(); ++it){
            // printlo(*it) << std::endl;
            mv.push_back(*it);
        }
        isSame = check_is_same(mv, mtv, true);
        EXPECT_TRUE(isSame) << err_msg;
    };

    std::array<uint, NGEOM_TYPES> arr1 = {3, 2, 1, 1, 3, 4}; 
    DofMap m1(std::make_shared<UniteDofMap>(arr1));
    dof_map m1t(arr1);
    TetGeomSparsity sp1;
    sp1.setFace(1, false);
    test_full_bypass(m1, m1t, "Problem in UniteDofMap");
    test_ordering(sp1, m1, m1t, "Problem in UniteDofMap");
    sp1.setFace(2, true);
    test_ordering(sp1, m1, m1t, "Problem in UniteDofMap");
    std::array<uint, NGEOM_TYPES> arr2 = {1, 2, 0, 1, 0, 3};
    DofMap m2(std::make_shared<UniteDofMap>(arr2));
    dof_map m2t(arr2);
    test_ordering(sp1, m2, m2t, "Problem in UniteDofMap");
    TetGeomSparsity sp2;
    sp2.setCell().setEdge(2, true);
    test_ordering(sp2, m2, m2t, "Problem in UniteDofMap");

    DofMap m3(std::make_shared<VectorDofMap>(3, m1.base()));
    dof_map m3t;
    m3t.set_dofs(arr1).set_dofs(arr1).set_dofs(arr1);
    test_full_bypass(m3, m3t, "Problem in VectorDofMap");
    test_ordering(sp1, m3, m3t, "Problem in VectorDofMap");
    test_ordering(sp2, m3, m3t, "Problem in VectorDofMap");
    VectorDofMapC<UniteDofMap> m3c(3, static_cast<UniteDofMap&>(*m1.base()));
    test_full_bypass(m3c, m3t, "Problem in VectorDofMap templated");
    test_ordering(sp1, m3c, m3t, "Problem in VectorDofMap templated");
    test_ordering(sp2, m3c, m3t, "Problem in VectorDofMap templated");

    DofMap m4(std::make_shared<ComplexDofMap>(ComplexDofMap::makeCompressed({m2.base(), m3.base()})));
    DofMap m5 = merge({m3, m2});
    dof_map m4t;
    m4t.set_dofs(arr2).set_dofs(arr1).set_dofs(arr1).set_dofs(arr1);
    test_full_bypass(m4, m4t, "Problem in ComplexDofMap");
    test_ordering(sp1, m4, m4t, "Problem in ComplexDofMap");
    test_ordering(sp2, m4, m4t, "Problem in ComplexDofMap");
    ComplexDofMapC<UniteDofMap, VectorDofMapC<UniteDofMap>> m4c;
    m4c.set<0>(static_cast<UniteDofMap&>(*m2.base()));
    m4c.set<1>(m3c);
    test_full_bypass(m4c, m4t, "Problem in ComplexDofMap templated");
    test_ordering(sp1, m4c, m4t, "Problem in ComplexDofMap templated");
    test_ordering(sp2, m4c, m4t, "Problem in ComplexDofMap templated");

    EXPECT_TRUE(m4 == m2*(m1 ^ 3)) << "Error in multiply function";
    EXPECT_TRUE(m1*m1*m1 == (m1 ^ 3)) << "Error in multiply function";
    EXPECT_TRUE(m1*m1*m2 != merge({m1, m1, m2})) << "Error in multiply function";
    EXPECT_TRUE(m2*m1*m1*m1 == m2*(m1 ^ 3)) << "Error in multiply function";
    EXPECT_TRUE(m4*m3 == m2*(m1 ^ 6)) << "Error in multiply function";
    EXPECT_TRUE(m3*m5 == (m1 ^ 6)*m2) << "Error in multiply function";
}