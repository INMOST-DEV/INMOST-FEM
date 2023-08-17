//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_MUTEX_TYPE_H
#define CARNUM_MUTEX_TYPE_H

#include <thread>
#include <mutex>
#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace ThreadPar{
    enum class Type{
        NONE = 0,
        STD = 1,
    #ifdef WITH_OPENMP
        OMP = 2, 
    #endif    
    };

    template <Type tp>
    struct mutex{
        static_assert("Unknown parallel type");
        constexpr static Type parallel_type = tp;
    };
    template <>
    struct mutex<Type::STD>: public std::mutex {
        constexpr static Type parallel_type = Type::STD;
    };
    template<>
    struct mutex<Type::NONE>{
        constexpr static Type parallel_type = Type::NONE;

        void lock() {}
        bool try_lock() { return true; }
        void unlock() {}
    };

    #ifdef WITH_OPENMP
    template<>
    struct mutex<Type::OMP>{
        constexpr static Type parallel_type = Type::OMP;

        mutex() noexcept {  omp_init_lock(&m_lock); }
        void lock() { omp_set_lock(&m_lock); }
        bool try_lock() { return omp_test_lock(&m_lock); }
        void unlock() { return omp_unset_lock(&m_lock); }

        omp_lock_t m_lock;
    };
    #endif

    template<Type tp>
    inline int get_num_threads(int nthreads = -1){ static_assert("Unknown parallel type"); return -1; }
    template<> inline int get_num_threads<Type::NONE>(int nthreads){ (void)nthreads; return 1; }
#ifdef WITH_OPENMP
    template<> inline int get_num_threads<Type::OMP>(int nthreads){ return nthreads < 0 ? omp_get_max_threads() : nthreads; }
#endif
    template<> inline int get_num_threads<Type::STD>(int nthreads){ 
        if (nthreads < 0){
            unsigned nb_threads_hint = std::thread::hardware_concurrency();
            nthreads = (nb_threads_hint == 0) ? 8 : nb_threads_hint;
        }
        return nthreads; 
    }

    template<typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    inline void perform_for_on_thread(int thread_num, CycleBodyFunc f, RandAccIterator beg, RandAccIterator end, Args&&... args){
        for (auto it = beg; it != end; ++it)
            f(it, thread_num, args...);
    }

    template<Type tp, typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    struct ParallelFor{
        static_assert("Unknown parallel type");
    };
    template<typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    struct ParallelFor<Type::NONE, CycleBodyFunc, RandAccIterator, Args...>{
        static void perform(int nthreads, CycleBodyFunc f, RandAccIterator beg, RandAccIterator end, Args&&... args){
            (void) nthreads;
            perform_for_on_thread(0, f, beg, end, std::forward<Args>(args)...);
        }
    };
    #ifdef WITH_OPENMP
    template<typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    struct ParallelFor<Type::OMP, CycleBodyFunc, RandAccIterator, Args...>{
        static void perform(int nthreads, CycleBodyFunc f, RandAccIterator beg, RandAccIterator end, Args&&... args){
            assert(nthreads != 0 && "Wrong maximal desirable number of threads");
            if (nthreads == 1) { 
                perform_for_on_thread(0, f, beg, end, std::forward<Args>(args)...);
                return; 
            }
            auto n = end - beg;
            #pragma omp parallel for num_threads(get_num_threads<Type::OMP>(nthreads))
            for (decltype(n) i = 0; i < n; ++i){
                f(beg+i, omp_get_thread_num(), args...);
            }
        }
    };
    #endif
    template<typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    struct ParallelFor<Type::STD, CycleBodyFunc, RandAccIterator, Args...>{
        static void perform(int nthreads, CycleBodyFunc f, RandAccIterator beg, RandAccIterator end, Args&&... args){
            assert(nthreads != 0 && "Wrong maximal desirable number of threads");
            if (nthreads == 1) { 
                perform_for_on_thread(0, f, beg, end, std::forward<Args>(args)...);
                return; 
            }
            nthreads = get_num_threads<Type::STD>(nthreads);
            auto niters = end - beg;
            unsigned batch_size = niters / nthreads;
            unsigned batch_remainder = niters % nthreads;

            std::vector< std::thread > work_threads(nthreads - 1);
            for (int i = 0; i < nthreads - 1; ++i)
                work_threads[i] = std::thread(perform_for_on_thread, i+1,
                                beg + batch_size*i + (i < batch_remainder ? i : batch_remainder), 
                                beg + batch_size*(i+1) + ((i+1) < batch_remainder ? (i+1) : batch_remainder), 
                                args...);
            perform_for_on_thread(0, f, beg + batch_size*(nthreads - 1) + batch_remainder, end, args...);
            std::for_each(work_threads.begin(), work_threads.end(), std::mem_fn(&std::thread::join));
        }
    };

    template<Type tp, typename CycleBodyFunc, typename RandAccIterator, typename... Args>
    inline void parallel_for(int nthreads, CycleBodyFunc f, RandAccIterator beg, RandAccIterator end, Args&&... args){
        ParallelFor<tp, CycleBodyFunc,RandAccIterator, Args...>::perform(nthreads, std::forward<CycleBodyFunc>(f), std::forward<RandAccIterator>(beg), std::forward<RandAccIterator>(end), std::forward<Args>(args)...);
    }

    template<typename MutexType>
    struct ParTypeFromMutex{
        static_assert("Unknown parallel type");
        constexpr static Type value = Type::NONE;
    };
    template <Type tp>
    struct ParTypeFromMutex<mutex<tp>>{
        constexpr static Type value = tp;
    };
    template <>
    struct ParTypeFromMutex<std::mutex>{
        constexpr static Type value = Type::STD;
    };
}

using mutex_none = ThreadPar::mutex<ThreadPar::Type::NONE>;
using mutex_std = ThreadPar::mutex<ThreadPar::Type::STD>;
#ifdef WITH_OPENMP
using mutex_omp = ThreadPar::mutex<ThreadPar::Type::OMP>;
#endif

#endif //CARNUM_MUTEX_TYPE_H

