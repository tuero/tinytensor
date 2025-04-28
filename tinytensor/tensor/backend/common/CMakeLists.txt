target_sources(tinytensor PRIVATE
    binary.h
    dispatch.h
    distribution.h
    reduce.h
    span.h
    unary.h
    util.h

    kernel/binary.hpp
    kernel/cast.hpp
    kernel/distribution.hpp
    kernel/reduce.hpp
    kernel/unary.hpp
)
