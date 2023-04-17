
#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <mutex>

#ifdef WIN32
// terminal colors, WinApi header
 #define NOMINMAX
// #include <windows.h>
 #include "termcolor.hpp"
#endif

#define VA_ARGS(...) , ##__VA_ARGS__

#define Log(...)    Logger{false, NULL VA_ARGS(__VA_ARGS__)}
#define LogTS(...)  Logger{true, NULL VA_ARGS(__VA_ARGS__)}
#define LogR(...)   Logger{false, "\r" VA_ARGS(__VA_ARGS__)}
#define LogRTS(...) Logger{true, "\r" VA_ARGS(__VA_ARGS__)}

// Dynex colors
#ifdef WIN32

#define TEXT_DEFAULT   0
#define TEXT_BLACK     1
#define TEXT_RED       2
#define TEXT_GREEN     3
#define TEXT_YELLOW    4
#define TEXT_BLUE      5
#define TEXT_MAGENTA   6
#define TEXT_CYAN      7
#define TEXT_WHITE     8
#define TEXT_GRAY      9
#define TEXT_BRED      10
#define TEXT_BGREEN    11
#define TEXT_BYELLOW   12
#define TEXT_BBLUE     13
#define TEXT_BMAGENTA  14
#define TEXT_BCYAN     15
#define TEXT_BWHITE    16
#define TEXT_SILVER    TEXT_BWHITE
#define TEXT_NONE      -1

static std::mutex loggermutex;

class Logger {
	private:
		std::ostringstream ss;
		int use_color;
		bool add_timestamp;
		std::string prefix = "";

	public:
		Logger(bool timestamp = true, const char *str = NULL, int color = -1) {
			use_color = color;
			if (use_color < 0) {
				add_timestamp = false;
				if (str) ss << str;
				if (timestamp) {
					auto t = std::time(nullptr);
					auto tm = *std::localtime(&t);
					ss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S ");
				}
			} else {
				if (str) prefix = str;
				add_timestamp = timestamp;
			}
		}
		template <typename T>
		Logger& operator<<(T const& t) {
			ss << t;
			return *this;
		}
		Logger& operator<<(std::ostream&(*s)(std::ostream&)) {
			ss << s;
			return *this;
		}
		~Logger() {
			if (use_color >= 0) {
				loggermutex.lock();
				if (prefix != "") std::cout << prefix;
				if (add_timestamp) {
					auto t = std::time(nullptr);
					auto tm = *std::localtime(&t);
					std::cout << std::put_time(&tm, "%d-%m-%Y %H:%M:%S ");
				}
				switch(use_color) {
				case TEXT_DEFAULT: std::cout << termcolor::reset; break;
				case TEXT_BLACK: std::cout << termcolor::grey; break;
				case TEXT_RED: std::cout << termcolor::red; break;
				case TEXT_GREEN: std::cout << termcolor::green; break;
				case TEXT_YELLOW: std::cout << termcolor::yellow; break;
				case TEXT_BLUE: std::cout << termcolor::blue; break;
				case TEXT_MAGENTA: std::cout << termcolor::magenta; break;
				case TEXT_CYAN: std::cout << termcolor::cyan; break;
				case TEXT_WHITE: std::cout << termcolor::white; break;
				case TEXT_GRAY: std::cout << termcolor::bright_grey; break;
				case TEXT_BRED: std::cout << termcolor::bright_red; break;
				case TEXT_BGREEN: std::cout << termcolor::bright_green; break;
				case TEXT_BYELLOW: std::cout << termcolor::bright_yellow; break;
				case TEXT_BBLUE: std::cout << termcolor::bright_blue; break;
				case TEXT_BMAGENTA: std::cout << termcolor::bright_magenta; break;
				case TEXT_BCYAN: std::cout << termcolor::bright_cyan; break;
				case TEXT_BWHITE: std::cout << termcolor::bright_white; break;
				}
				std::cout << ss.str() << termcolor::reset;
				loggermutex.unlock();
			} else {
				std::cout << ss.str();
			}
			std::cout << std::flush;
		}
};

#else

 #define TEXT_DEFAULT   "\033[0m"
 #define TEXT_BLACK     "\033[0;30m"
 #define TEXT_RED       "\033[0;31m"
 #define TEXT_GREEN     "\033[0;32m"
 #define TEXT_YELLOW    "\033[0;33m"
 #define TEXT_BLUE      "\033[0;34m"
 #define TEXT_MAGENTA   "\033[0;35m"
 #define TEXT_CYAN      "\033[0;36m"
 #define TEXT_WHITE     "\033[0;37m"
 #define TEXT_GRAY      "\033[1;30m"
 #define TEXT_BRED      "\033[1;31m"
 #define TEXT_BGREEN    "\033[1;32m"
 #define TEXT_BYELLOW   "\033[1;33m"
 #define TEXT_BBLUE     "\033[1;34m"
 #define TEXT_BMAGENTA  "\033[1;35m"
 #define TEXT_BCYAN     "\033[1;36m"
 #define TEXT_BWHITE    "\033[1;37m"
 #define TEXT_SILVER    TEXT_BWHITE // "\033[1;315m"
 #define TEXT_NONE      NULL

class Logger {
	private:
		std::ostringstream ss;
		bool use_color = false;

	public:
		Logger(bool timestamp = true, const char *str = NULL, const char* color = NULL) {
			if (str) ss << str;
			if (timestamp) {
				auto t = std::time(nullptr);
				auto tm = *std::localtime(&t);
				ss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S ");
			}
			if (color) {
				ss << color;
				use_color = true;
			}
		}
		template <typename T>
		Logger& operator<<(T const& t) {
			ss << t;
			return *this;
		}
		Logger& operator<<(std::ostream&(*s)(std::ostream&)) {
			ss << s;
			return *this;
		}
		~Logger() {
			if (use_color) ss << TEXT_DEFAULT;
			std::cout << ss.str() << std::flush;
		}
};

#endif
