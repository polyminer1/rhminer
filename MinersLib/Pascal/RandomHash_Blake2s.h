/**
 * Based on the SPH implementation of blake2s
 * Provos Alexis - 2016
 */

#include "RandomHash_core.h"

static PLATFORM_CONST uint32_t RH_ALIGN(64) blake2s_IV[8] = {
	0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
	0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static PLATFORM_CONST uint8_t RH_ALIGN(64) blake2s_sigma[10][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
};

#define blake2s_set_lastnode( S ){	S->f[1] = ~0U; }

#define blake2s_clear_lastnode( S ){ S->f[1] = 0U; }

/* Some helper functions, not necessarily useful */

#define blake2s_set_lastblock( S )                  \
{                                                   \
	if( S->last_node ) blake2s_set_lastnode( S );   \
                                                    \
	S->f[0] = ~0U;                                  \
}                                                   \


#define blake2s_clear_lastblock( S )                 \
{                                                    \
	if( S->last_node ) blake2s_clear_lastnode( S );  \
                                                     \
	S->f[0] = 0U;                                    \
}


#define blake2s_increment_counter( S, inc ) \
{                                           \
	S->t[0] += inc;                         \
	S->t[1] += ( S->t[0] < inc );           \
}                                           \


#define load32_SSE2(src)    (*(uint32_t *)(src))

#define store32_SSE2(dst, w) {*(uint32_t *)(dst) = w;}


inline int blake2s_compress_SSE2( blake2s_state *S, const uint8_t block[BLAKE2S_BLOCKBYTES] )
{
	uint32_t m[16];
	uint32_t v[16];

    for( size_t i = 0; i < 16; ++i )
		m[i] = load32_SSE2( block + i * sizeof( m[i] ) );

    v[0] = S->h[0];
    v[1] = S->h[1];
    v[2] = S->h[2];
    v[3] = S->h[3];
    v[4] = S->h[4];
    v[5] = S->h[5];
    v[6] = S->h[6];
    v[7] = S->h[7];

	v[ 8] = blake2s_IV[0];
	v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];
	v[11] = blake2s_IV[3];
	v[12] = S->t[0] ^ blake2s_IV[4];
	v[13] = S->t[1] ^ blake2s_IV[5];
	v[14] = S->f[0] ^ blake2s_IV[6];
	v[15] = S->f[1] ^ blake2s_IV[7];

#define G(r,i,a,b,c,d) \
	do { \
		a = a + b + m[blake2s_sigma[r][2*i+0]]; \
		d = ROTR32(d ^ a, 16); \
		c = c + d; \
		b = ROTR32(b ^ c, 12); \
		a = a + b + m[blake2s_sigma[r][2*i+1]]; \
		d = ROTR32(d ^ a, 8); \
		c = c + d; \
		b = ROTR32(b ^ c, 7); \
	} while(0)

#define ROUND(r)  \
	do { \
		G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
		G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
		G(r,2,v[ 2],v[ 6],v[10],v[14]); \
		G(r,3,v[ 3],v[ 7],v[11],v[15]); \
		G(r,4,v[ 0],v[ 5],v[10],v[15]); \
		G(r,5,v[ 1],v[ 6],v[11],v[12]); \
		G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
		G(r,7,v[ 3],v[ 4],v[ 9],v[14]); \
	} while(0)
	ROUND( 0 );
	ROUND( 1 );
	ROUND( 2 );
	ROUND( 3 );
	ROUND( 4 );
	ROUND( 5 );
	ROUND( 6 );
	ROUND( 7 );
	ROUND( 8 );
	ROUND( 9 );

	for( size_t i = 0; i < 8; ++i )
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];

#undef G
#undef ROUND
	return 0;
}


int blake2s_update_SSE2( blake2s_state *S, const uint8_t *in, uint64_t inlen )
{
	while( inlen > 0 )
	{
		size_t left = S->buflen;
		size_t fill = 2 * BLAKE2S_BLOCKBYTES - left;

		if( inlen > fill )
		{
			memcpy( S->buf + left, in, fill ); 
			S->buflen += fill;
			blake2s_increment_counter( S, BLAKE2S_BLOCKBYTES );
			blake2s_compress_SSE2( S, S->buf ); 
			memcpy( S->buf, S->buf + BLAKE2S_BLOCKBYTES, BLAKE2S_BLOCKBYTES ); 
			S->buflen -= BLAKE2S_BLOCKBYTES;
			in += fill;
			inlen -= fill;
		}
		else 
		{
			memcpy(S->buf + left, in, (size_t) inlen);
			S->buflen += (size_t) inlen; 
			in += inlen;
			inlen -= inlen;
		}
	}

	return 0;
}


int blake2s_final_SSE2( blake2s_state *S, uint8_t *out, uint8_t outlen )
{
	uint8_t buffer[BLAKE2S_OUTBYTES];

	if( S->buflen > BLAKE2S_BLOCKBYTES )
	{
		blake2s_increment_counter( S, BLAKE2S_BLOCKBYTES );
		blake2s_compress_SSE2( S, S->buf );
		S->buflen -= BLAKE2S_BLOCKBYTES;
		memcpy( S->buf, S->buf + BLAKE2S_BLOCKBYTES, S->buflen );
	}

	blake2s_increment_counter( S, ( uint32_t )S->buflen );
	blake2s_set_lastblock( S );
	memset( S->buf + S->buflen, 0, 2 * BLAKE2S_BLOCKBYTES - S->buflen ); /* Padding */
	blake2s_compress_SSE2( S, S->buf );

	for( int i = 0; i < 8; ++i ) /* Output full hash to temp buffer */
		store32_SSE2( buffer + sizeof( S->h[i] ) * i, S->h[i] );

	memcpy( out, buffer, outlen );
	return 0;
}


void RandomHash_blake2s(RH_StridePtr roundInput, RH_StridePtr output, U32 bitSize)
{
    uint32_t *in = RH_STRIDE_GET_DATA(roundInput);

    RH_ALIGN(64) blake2s_state S;
	RH_ALIGN(64) blake2s_param P[1];
    const int outlen = bitSize / 8;

	P->digest_length = outlen;
	P->key_length    = 0;
	P->fanout        = 1;
	P->depth         = 1;
	store32_SSE2( &P->leaf_length, 0 ); 
    P->node_offset[0] = 0;
    P->node_offset[1] = 0;
    P->node_offset[2] = 0;
    P->node_offset[3] = 0;
    P->node_offset[4] = 0;
    P->node_offset[5] = 0;
	P->node_depth    = 0;
	P->inner_length  = 0;

#if defined(_WIN32_WINNT) || defined(__CUDA_ARCH__)
    RH_memzero_8(P->salt, sizeof( P->salt ))
    RH_memzero_8(P->personal, sizeof( P->personal ) );
#else
    memset(P->salt, 0, sizeof( P->salt ));
    memset(P->personal, 0, sizeof( P->personal ) );
#endif

    RH_memzero_of16(&S, sizeof( blake2s_state ) );    

	for( int i = 0; i < 8; ++i ) S.h[i] = blake2s_IV[i];

	uint32_t *p = ( uint32_t * )( P );

	for( size_t i = 0; i < 8; ++i )
		S.h[i] ^= load32_SSE2( &p[i] );
    
	blake2s_update_SSE2( &S, ( uint8_t * )in, RH_STRIDE_GET_SIZE(roundInput) );
	blake2s_final_SSE2( &S, RH_STRIDE_GET_DATA8(output), outlen);
    RH_STRIDE_SET_SIZE(output, outlen)
}
