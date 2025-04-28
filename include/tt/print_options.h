// print_options.h
// Printing options for formatting

#ifndef TINYTENSOR_PRINT_OPTIONS_H_
#define TINYTENSOR_PRINT_OPTIONS_H_

namespace tinytensor {

/**
 * Set the number of digits of precision for floating point values
 * @param precision The precision
 */
void set_print_precision(int precision);

/**
 * Get the current print precision
 * @return The print precision
 */
auto get_print_precision() -> int;

/**
 * Set the width of the printed tensor values
 * @param width The width
 */
void set_print_width(int width);

/**
 * Get the current print width
 * @return The print width
 */
auto get_print_width() -> int;

/**
 * Set the maximum number of lines each tensor can print before being summarized
 * @param max_lines The maximum lines to set
 */
void set_max_lines(int max_lines);

/**
 * Get the current maximum lines
 * @return The maximum lines
 */
auto get_max_lines() -> int;

/**
 * Set number of items on a line before summarization occurs
 * @param line_width the line width
 */
void set_print_line_width(int line_width);

/**
 * Get the current print line width
 * @return The print line width
 */
auto get_print_line_width() -> int;

/**
 * Set suppression to not use scientific notation when values too large or small
 * @param suppress Flag to suppress
 */
void set_print_suppression(bool suppress);

/**
 * Get the current print suppression
 * @return The print suppression
 */
auto get_print_suppression() -> bool;

}    // namespace tinytensor

#endif    // TINYTENSOR_PRINT_OPTIONS_H_
