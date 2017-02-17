#ifndef LOG_HPP
#define LOG_HPP

#include <cstdio>

#define LOG_INFO(format, ...) fprintf(stdout, "[%s][%d] " format, __func__, __LINE__, ## __VA_ARGS__)

#endif // LOG_HPP
