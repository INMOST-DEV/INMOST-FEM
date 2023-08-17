//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_MEMORY_PARALLEL_H
#define CARNUM_MEMORY_PARALLEL_H

#include "fem_memory.h"
#include "mutex_type.h"

namespace Ani{

///The structure for thread-safe memory allocation
template<typename ScalarType = double, typename IndexType = int, typename MutexType = mutex_none>
struct DynMemParallel{
    using Real = ScalarType;
    using Int = IndexType;

    /// Allocate memory for current thread 
    /// @return thread-safe index of internal allocated memory
    /// @note this method of memory allocation is more convenient for std::thread multi-threading
    Int alloc(){
        std::lock_guard<MutexType> lock(m_mutex);
        if (m_unused.empty()){
            m_mem.emplace_back(std::make_unique<DynMem<Real, Int>>());
            return m_mem.size() - 1;
        } else {
            Int id = m_unused.back();
            m_unused.resize(m_unused.size() - 1);
            return id;
        }
    }
    /// Release internal memory by it's index
    void release(Int mem_id){
        std::lock_guard<MutexType> lock(m_mutex);
        m_unused.push_back(mem_id);
    }
    /// Allocate memory for several threads simultaneously
    /// @param num_threads is number of threads for that memory will be allocated
    /// @return first index of internal allocated memory "id" and "id + num_threads"
    /// @note this method of memory allocation is more convenient for OpenMP multi-threading
    /// @warning Should be called only once by master thread
    std::pair<Int, Int> alloc_range(Int num_threads){
        std::lock_guard<MutexType> lock(m_mutex);
        bool reuse_mem = (m_unused.size() == m_mem.size());
        Int lsz = reuse_mem ? static_cast<Int>(m_mem.size()) : 0;
        Int st_sz = static_cast<Int>(m_mem.size()) - lsz;
        m_mem.resize(st_sz+num_threads);
        for (Int i = lsz; i < num_threads; ++i)
            m_mem[i] = std::move(std::make_unique<DynMem<Real, Int>>());
        if (reuse_mem)
            m_unused.resize(0);
        return {st_sz, st_sz+num_threads};    
    }
    ///Release memory for range of threads
    /// @param id_range include first index of internal allocated memory "id" and "id + num_threads"
    /// @note this method of memory release is more convenient for OpenMP multi-threading
    /// @warning Should be called only once by master thread
    void release_range(std::pair<Int, Int> id_range){
        std::lock_guard<MutexType> lock(m_mutex);
        auto beg = m_unused.size();
        auto num_threads = id_range.second - id_range.first;
        m_unused.resize(beg + num_threads);
        for (int i = 0; i < num_threads; ++i)
            m_unused[beg + i] = id_range.second - i - 1;
    }

    void release_all() {
        std::lock_guard<MutexType> lock(m_mutex);
        auto sz = m_mem.size();
        m_unused.resize(sz);
        for (std::size_t i = 0; i < sz; ++i)
            m_unused[i] = sz - 1 - i;
    }
    void clear(){
        std::lock_guard<MutexType> lock(m_mutex);
        m_unused.clear();
        m_mem.clear();
    }
    DynMem<Real, Int>* get(Int mem_id){ return m_mem[mem_id].get(); }


protected:
    std::vector<std::unique_ptr<DynMem<Real, Int>>> m_mem;
    std::vector<Int> m_unused;
    mutable MutexType m_mutex;
};

}
#endif //CARNUM_MEMORY_PARALLEL_H