// exception.cpp
// Exception type

#include <tt/exception.h>

#include <iostream>
#include <sstream>
#include <string>

namespace tinytensor {

TTException::TTException(
    const std::string &input_message,
    const char *file_name,
    const char *function_signature,
    int line_number
) {
    std::ostringstream os;
    os << "File: " << file_name << ":" << line_number << std::endl;
    os << "In function: " << std::endl;
    os << "\t" << function_signature << std::endl;
    os << std::endl;
    os << "TinyTensor Error: " << input_message << std::endl;
    message = os.str();
}

auto TTException::what() const noexcept -> const char * {
    return message.c_str();
}

}    // namespace tinytensor
