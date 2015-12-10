#include "perlin.h"

#define setup(i,b0,b1,r0,r1)\
	t = vec[i] + N;\
	b0 = ((int)t) & BM;\
	b1 = (b0+1) & BM;\
	r0 = t - (int)t;\
	r1 = r0 - 1.0f;

#define hash(a)\
	a = (a + 0x7ed55d16) + (a << 12);\
	a = (a ^ 0xc761c23c) ^ (a >> 19);\
	a = (a + 0x165667b1) + (a << 5);\
	a = (a + 0xd3a2646c) ^ (a << 9);\
	a = (a + 0xfd7046c5) + (a << 3);\
	a = (a ^ 0xb55a4f09) ^ (a >> 16);

__device__ void Perlin::init_rand(int seed) {
	hash(seed);
	rng = thrust::default_random_engine(float(seed));
	unitDistrib = thrust::uniform_real_distribution<float>(0, 1);
}

__device__ float Perlin::noise1(float arg) {
	int bx0, bx1;
	float rx0, rx1, sx, t, u, v, vec[1];

	vec[0] = arg;

	if (mStart) {
		init_rand(mSeed);
		mStart = false;
		init();
	}

	setup(0, bx0,bx1, rx0,rx1);

	sx = s_curve(rx0);

	u = rx0 * g1[p[bx0]];
	v = rx1 * g1[p[bx1]];

	return lerrp(sx, u, v);
}

__device__ float Perlin::noise2(float vec[2]) {
	int bx0, bx1, by0, by1, b00, b10, b01, b11;
	float rx0, rx1, ry0, ry1, *q, sx, sy, a, b, t, u, v;
	int i, j;

	if (mStart) {
		init_rand(mSeed);
		mStart = false;
		init();
	}

	setup(0,bx0,bx1,rx0,rx1);
	setup(1,by0,by1,ry0,ry1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];

	sx = s_curve(rx0);
	sy = s_curve(ry0);

	#define at2(rx,ry) (rx * q[0] + ry * q[1])

	q = g2[b00];
	u = at2(rx0,ry0);
	q = g2[b10];
	v = at2(rx1,ry0);
	a = lerrp(sx, u, v);

	q = g2[b01];
	u = at2(rx0,ry1);
	q = g2[b11];
	v = at2(rx1,ry1);
	b = lerrp(sx, u, v);

	return lerrp(sy, a, b);
}

__device__ float Perlin::noise3(float vec[3]) {
	int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
	float rx0, rx1, ry0, ry1, rz0, rz1, *q, sy, sz, a, b, c, d, t, u, v;
	int i, j;

	if (mStart) {
		init_rand(mSeed);
		mStart = false;
		init();
	}

	setup(0, bx0,bx1, rx0,rx1);
	setup(1, by0,by1, ry0,ry1);
	setup(2, bz0,bz1, rz0,rz1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];

	t  = s_curve(rx0);
	sy = s_curve(ry0);
	sz = s_curve(rz0);

    #define at3(rx,ry,rz) ( rx * q[0] + ry * q[1] + rz * q[2] )

	q = g3[b00 + bz0] ; u = at3(rx0,ry0,rz0);
	q = g3[b10 + bz0] ; v = at3(rx1,ry0,rz0);
	a = lerrp(t, u, v);

	q = g3[b01 + bz0] ; u = at3(rx0,ry1,rz0);
	q = g3[b11 + bz0] ; v = at3(rx1,ry1,rz0);
	b = lerrp(t, u, v);

	c = lerrp(sy, a, b);

	q = g3[b00 + bz1] ; u = at3(rx0,ry0,rz1);
	q = g3[b10 + bz1] ; v = at3(rx1,ry0,rz1);
	a = lerrp(t, u, v);

	q = g3[b01 + bz1] ; u = at3(rx0,ry1,rz1);
	q = g3[b11 + bz1] ; v = at3(rx1,ry1,rz1);
	b = lerrp(t, u, v);

	d = lerrp(sy, a, b);

	return lerrp(sz, c, d);
}

__device__ void Perlin::normalize2(float v[2]) {
	float s;

	s = (float)sqrt(v[0] * v[0] + v[1] * v[1]);
	s = 1.0f/s;
	v[0] = v[0] * s;
	v[1] = v[1] * s;
}

__device__ void Perlin::normalize3(float v[3]) {
	float s;

	s = (float)sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	s = 1.0f/s;

	v[0] = v[0] * s;
	v[1] = v[1] * s;
	v[2] = v[2] * s;
}

__device__ void Perlin::init(void) {
	int i, j, k;

	for (i = 0 ; i < B ; i++) {
		p[i] = i;
		g1[i] = (float)((int(unitDistrib(rng)*(float)RAND_MAX) % (B + B)) - B) / B;
		for (j = 0 ; j < 2 ; j++) {
			g2[i][j] = (float)((int(unitDistrib(rng)*(float)RAND_MAX) % (B + B)) - B) / B;
		}
		normalize2(g2[i]);
		for (j = 0 ; j < 3 ; j++) {
			g3[i][j] = (float)((int(unitDistrib(rng)*(float)RAND_MAX) % (B + B)) - B) / B;
		}
		normalize3(g3[i]);
	}

	while (--i) {
		k = p[i];
		p[i] = p[j = int(unitDistrib(rng)*(float)RAND_MAX) % B];
		p[j] = k;
	}

	for (i = 0 ; i < B + 2 ; i++) {
		p[B + i] = p[i];
		g1[B + i] = g1[i];
		for (j = 0 ; j < 2 ; j++) {
			g2[B + i][j] = g2[i][j];
		}
		for (j = 0 ; j < 3 ; j++) {
			g3[B + i][j] = g3[i][j];
		}
	}
}


__device__ float Perlin::perlin_noise_2D(float vec[2]) {
	int terms = mOctaves;
	float freq = mFrequency;
	float result = 0.0f;
	float amp = mAmplitude;
	
	vec[0] *= freq;
	vec[1] *= freq;
	
	for(int i=0; i < terms; i++) {
		result += noise2(vec)*amp;
		vec[0] *= 2.0f;
		vec[1] *= 2.0f;
		amp*=0.5f;
	}
	return result;
}

__device__ float Perlin::perlin_noise_3D(float vec[3]) {
	int terms = mOctaves;
	float freq = mFrequency;
	float result = 0.0f;
	float amp = mAmplitude;

	vec[0] *= freq;
	vec[1] *= freq;
	vec[2] *= freq;

	for(int i=0; i<terms; i++ ) {
		result += noise3(vec)*amp;
		vec[0] *= 2.0f;
		vec[1] *= 2.0f;
		vec[2] *= 2.0f;
		amp*=0.5f;
	}
	return result;
}

__device__ Perlin::Perlin(int octaves,float freq,float amp,int seed) {
	mOctaves = octaves;
	mFrequency = freq;
	mAmplitude = amp;
	mSeed = seed;
	mStart = true;
}