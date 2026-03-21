#ifndef SMILE_LOGGER_H
#define SMILE_LOGGER_H

// {{SMILE_PUBLIC_HEADER}}

#include "intarray.h"
#include "stringarray.h"
#include <cstdarg>

class DSL_loggerRedirect
{
public:
	virtual ~DSL_loggerRedirect() {}
	virtual void LogError(int code, const char* message) = 0;
	virtual DSL_loggerRedirect* Clone() = 0;
};

class DSL_logger
{
public:
	DSL_logger();
	DSL_logger(const DSL_logger& other);
	~DSL_logger();
	DSL_logger& operator=(const DSL_logger& other);

	int LogError(int errCode, const char* theMessage = NULL, const char* prefix = NULL);
	int VLogError(int errCode, const char* fmt, ...);
	int VLogError(int errCode, const char* fmt, va_list args);

	int GetLastError() const;
	const char* GetLastErrorMessage() const;
	int GetError(int index) const;
	const char* GetErrorMessage(int index) const;
	int GetNumberOfErrors() const  { return codes.GetSize(); }
	void Clear();

	// stdout/stderr can be passed as file arg
	void RedirectToFile(FILE* file, const char* format = NULL, bool close = false);
	int RedirectToFile(const char *filename, const char* format = NULL);
	void RedirectToLogger(DSL_logger& parentLogger);
	void Redirect(DSL_loggerRedirect* newRedirect);

#ifndef SMILE_NO_V2_0_COMPATIBILITY
	void Flush() { Clear(); }
#endif

private:
	void CopyRedirection(const DSL_logger& other);

	DSL_intArray codes;
	DSL_stringArray messages;
	DSL_loggerRedirect* redirection;
	bool ownedRedirection;
};

// access to global logger - will be removed in the future release
DSL_logger& DSL_errorH();

#endif
