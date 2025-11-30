// exception.h
// Exception type

#ifndef TINYTENSOR_EXCEPTION_H_
#define TINYTENSOR_EXCEPTION_H_

#include <tt/export.h>

#include <exception>
#include <iostream>
#include <string>

namespace tinytensor {

class TINYTENSOR_EXPORT TTException : public std::exception {
public:
    TTException(const std::string &message, const char *file_name, const char *function_signature, int line_number);
    [[nodiscard]] auto what() const noexcept -> const char * override;

private:
    std::string message;
};

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define TT_EXCEPTION(msg) throw TTException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__)

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define TT_ERROR(msg)                                                        \
    {                                                                        \
        try {                                                                \
            throw TTException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__); \
        } catch (const TTException &e) {                                     \
            std::cout << e.what() << std::endl;                              \
        }                                                                    \
        std::exit(1);                                                        \
    }

}    // namespace tinytensor

#endif    // TINYTENSOR_EXCEPTION_H_
