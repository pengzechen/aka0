#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <cstdio>

// Log Levels
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_NONE  4

// Current Log Level (Default to INFO if not defined by compiler)
#ifndef CURRENT_LOG_LEVEL
    #define CURRENT_LOG_LEVEL LOG_LEVEL_DEBUG
#endif

// Colors
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Macros - using do-while(0) to ensure they behave like single statements
#if CURRENT_LOG_LEVEL <= LOG_LEVEL_DEBUG
    #define LOGD(fmt, ...) printf(ANSI_COLOR_CYAN "[DEBUG] " fmt ANSI_COLOR_RESET "\n", ##__VA_ARGS__)
#else
    #define LOGD(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL <= LOG_LEVEL_INFO
    #define LOGI(fmt, ...) printf(ANSI_COLOR_GREEN "[INFO]  " fmt ANSI_COLOR_RESET "\n", ##__VA_ARGS__)
#else
    #define LOGI(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL <= LOG_LEVEL_WARN
    #define LOGW(fmt, ...) printf(ANSI_COLOR_YELLOW "[WARN]  " fmt ANSI_COLOR_RESET "\n", ##__VA_ARGS__)
#else
    #define LOGW(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL <= LOG_LEVEL_ERROR
    #define LOGE(fmt, ...) fprintf(stderr, ANSI_COLOR_RED "[ERROR] " fmt ANSI_COLOR_RESET "\n", ##__VA_ARGS__)
#else
    #define LOGE(fmt, ...) do {} while(0)
#endif

#endif // LOGGER_HPP
