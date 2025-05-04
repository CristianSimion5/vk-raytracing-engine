#ifndef RANDOM
#define RANDOM

#include "globals.glsl"

uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0u;

  for(uint n = 0u; n < 16u; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}

vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(r1);

  vec3 direction = vec3(cos(2 * M_PI * r2) * sq, sin(2 * M_PI * r2) * sq, sqrt(1 - r1));
  direction      = direction.x * x + direction.y * y + direction.z * z;

  return direction;
}

void createCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  if(abs(N.x) > abs(N.y))
    Nt = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
  else
    Nt = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
  Nb = cross(N, Nt);
}

vec3 samplingNDF_GGXTR(inout uint seed, float alpha2)
{
    float r1 = rnd(seed);
    float r2 = rnd(seed);

    // https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
    float cosTheta = sqrt((1.0f - r2) / ((alpha2 - 1.0f) * r2 + 1.0f));
    float sinTheta = clamp(sqrt(1.0f - cosTheta * cosTheta), 0.0f, 1.0f);
    float phi = r1 * 2.0f * M_PI;
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);

    // Polar to cartesian, radius = 1 and angles theta and phi
    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

#endif // RANDOM