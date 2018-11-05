#include "precomp.h"
#include "corelib/Log.h"
#include "corelib/Guards.h"
#include <time.h>


using namespace std;

/// Associate a name with each thread for nice logging.
struct ThreadLocalLogName
{
	ThreadLocalLogName(char const* _name) { name = _name; }
	thread_local static char const* name;
};

thread_local char const* ThreadLocalLogName::name;

char const* getThreadName()
{
	return ThreadLocalLogName::name ? ThreadLocalLogName::name : "Log";
}

void setThreadName(char const* _n)
{
	ThreadLocalLogName::name = _n;
}
