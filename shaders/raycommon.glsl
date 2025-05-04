#ifndef RAYCOMMON
#define RAYCOMMON

#include "host_device.h"

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

struct hitPayload
{
    vec3 hitValue;
    uint seed;
    uint depth;
    vec3 rayOrigin;
    vec3 rayDirection;
    vec3 weight;
    bool isSpecular;
    float lightDist;
    vec3 shadowRayDir;
};

struct shadowPayload
{
    bool isHit;
    uint seed;
    uint depth;
};

#endif // RAYCOMMON