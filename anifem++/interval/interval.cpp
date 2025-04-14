//Hack to overcome that CMake less 3.19 version doesn't allow create header-only library

namespace Ani{ namespace internal_interval{    
    void internal_do_nothing() {}
}}