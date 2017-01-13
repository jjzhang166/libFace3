#ifndef QY_LOG_HPP
#define QY_LOG_HPP

#include <cstdio>

#define LOG_INFO(format, ...) fprintf(stderr, "[%s][%d] " format, __func__, __LINE__, ## __VA_ARGS__)

#endif // QY_LOG_HPP
