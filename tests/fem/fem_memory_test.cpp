//
// Created by Liogky Alexey on 01.06.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/fem_memory.h"
#include <chrono>

TEST(AniInterface, DynMem){
    using namespace Ani;
    DynMem<> wmem;

    static int rr = 0;
    auto fill_vals = [](const DynMem<>::MemPart& p){
        for (std::size_t i = 0; i < p.m_mem.dSize; ++i)
            p.m_mem.ddata[i] = rr*10000 + i;
        for (std::size_t i = 0; i < p.m_mem.iSize; ++i)
            p.m_mem.idata[i] = rr*10000 + i; 
        for (std::size_t i = 0; i < p.m_mem.mSize; ++i)
            p.m_mem.mdata[i] = DenseMatrix<>(nullptr, 0, 0);       
        rr++;
    };
    #define CHECK_CHUNK_SZ(CH, RB, IB, MB) EXPECT_TRUE(wmem.m_chunks[CH].rdata.size() == RB && wmem.m_chunks[CH].idata.size() == IB && wmem.m_chunks[CH].mdata.size() == MB)
    #define CHECK_CHUNK(CH, RB, IB, MB) EXPECT_TRUE(wmem.m_chunks.size() >= CH \
            && wmem.m_chunks[CH].rbusy == RB && wmem.m_chunks[CH].ibusy == IB && wmem.m_chunks[CH].mbusy == MB\
            && wmem.m_chunks[CH].rbusy <= wmem.m_chunks[CH].rdata.size() && wmem.m_chunks[CH].ibusy <= wmem.m_chunks[CH].idata.size() && wmem.m_chunks[CH].mbusy <= wmem.m_chunks[CH].mdata.size()\
            )
    #define CHECK_STATE(NCH, RB, IB, MB) EXPECT_TRUE(wmem.m_chunks.size() == NCH && wmem.m_chunks[NCH-1].rbusy == RB && wmem.m_chunks[NCH-1].ibusy == IB && wmem.m_chunks[NCH-1].mbusy == MB)

    {
        auto m1 = wmem.alloc(10, 20, 5); fill_vals(m1);
        CHECK_STATE(1, 10, 20, 5);
        auto m2 = wmem.alloc(3, 1, 0); fill_vals(m2);
        CHECK_STATE(2, 3, 1, 0);
        auto m3 = wmem.alloc(5, 0, 10); fill_vals(m3);
        CHECK_STATE(3, 5, 0, 10);

        m2.clear();
        CHECK_CHUNK(1, 0, 0, 0);
        auto m4 = wmem.alloc(1, 1, 0); fill_vals(m4);
        CHECK_CHUNK(1, 1, 1, 0);
        auto m5 = wmem.alloc(2, 0, 0); fill_vals(m5);
        CHECK_CHUNK(1, 3, 1, 0);
        m4.clear();
        CHECK_CHUNK(1, 3, 1, 0);
        auto m6 = wmem.alloc(1, 1, 0); fill_vals(m6);
        CHECK_STATE(4, 1, 1, 0);
        m5.clear(); 
        CHECK_CHUNK(1, 0, 0, 0);
        m3.clear();
        CHECK_CHUNK(2, 0, 0, 0);
        CHECK_STATE(4, 1, 1, 0);

        EXPECT_ANY_THROW(wmem.defragment());
    }
    wmem.defragment();
    CHECK_STATE(1, 0, 0, 0);
    CHECK_CHUNK_SZ(0, (10+3+5+1), (20+1+0+1), (5+0+10+0));
    {
        auto m1 = wmem.alloc(10, 20, 5); fill_vals(m1);
        auto m2 = wmem.alloc(3, 1, 0); fill_vals(m2);
        auto m3 = wmem.alloc(5, 0, 10); fill_vals(m3);
        CHECK_STATE(1, (10+3+5), (20+1+0), (5+0+10));
    }
    #undef CHECK_STATE
    #undef CHECK_CHUNK
    #undef CHECK_CHUNK_SZ
}