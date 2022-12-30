
#pragma once

#include <string>
#include <sstream>

// Dynex colors
#ifdef WIN32
#define TEXT_DEFAULT  ""
#define TEXT_RED      ""
#define TEXT_GREEN    ""
#define TEXT_YELLOW   ""
#define TEXT_BLUE     ""
#define TEXT_PURPLE   ""
#define TEXT_CYAN     ""
#define TEXT_WHITE    ""
#define TEXT_SILVER   ""
#else
#define TEXT_DEFAULT  "\033[0m"
#define TEXT_RED      "\033[1;31m"
#define TEXT_GREEN    "\033[1;32m"
#define TEXT_YELLOW   "\033[1;33m"
#define TEXT_BLUE     "\033[1;34m"
#define TEXT_PURPLE   "\033[1;35m"
#define TEXT_CYAN     "\033[1;36m"
#define TEXT_WHITE    "\033[1;37m"
#define TEXT_SILVER   "\033[1;315m"
#endif


#define Log   Logger{false}
#define LogTS Logger{true}


class Logger {
	private:
		std::ostringstream ss;

	public:
		Logger(bool timestamp = true) {
			if (timestamp) {
				auto t = std::time(nullptr);
				auto tm = *std::localtime(&t);
				ss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S ");
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
			std::cout << ss.str() << std::flush;
		}
};
