#ifndef TINYTENSOR_MACROS_H_
// macros.h
// Various macros used for annotating functions and classes

#define TINYTENSOR_MACROS_H_

namespace tinytensor {

#if defined(__CUDACC__)
#define TT_DEVICE      __device__
#define TT_HOST        __host__
#define TT_HOST_DEVICE __host__ __device__
#define TT_INLINE      __forceinline__
#define TT_STD_FUNC    nvstd::function
#else
#define TT_DEVICE
#define TT_HOST
#define TT_HOST_DEVICE
#define TT_INLINE   inline
#define TT_STD_FUNC std::function
#endif

}    // namespace tinytensor

#endif    // TINYTENSOR_MACROS_H_
