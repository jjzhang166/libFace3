#ifndef _LIBFACE_LOG_HPP
#define _LIBFACE_LOG_HPP

#include <cstdio>

#define LOG_INFO(format, ...) fprintf(stdout, "[%s][%d] " format, __func__, __LINE__, ## __VA_ARGS__)

#endif // _LIBFACE_LOG_HPP
