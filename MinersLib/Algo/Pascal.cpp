#include "precomp.h"
#include "sph_sha2.h"
#include "MinersLib/Pascal/PascalCommon.h"

void PascalHashV3(void *state, const void *input)
{
  sph_sha256_context ctx_sha;
  uint32_t hash[16];

  sph_sha256_init(&ctx_sha);
  sph_sha256(&ctx_sha, input, PascalHeaderSize);
  sph_sha256_close(&ctx_sha, hash);

  sph_sha256_init(&ctx_sha);
  sph_sha256(&ctx_sha, hash, 32);
  sph_sha256_close(&ctx_sha, hash);

  memcpy(state, hash, 32);
}
