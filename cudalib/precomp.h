#ifndef RHMINER_CUDA_PRECOMP_H_
#define RHMINER_CUDA_PRECOMP_H_

#ifndef __CUDA_ARCH__

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <set>
#include <string>
#include <assert.h>

#include "corelib/utils.h"

#endif

#include "corelib/basetypes.h"
#include "corelib/rh_endian.h"

#ifdef BOOST_GPU_ENABLED
#error dont compile cuda with boost
#endif

#endif //RHMINER_CUDA_PRECOMP_H_
