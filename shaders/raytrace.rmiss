#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "raycommon.glsl"
#include "host_device.h"

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main()
{
    // hitValue = vec3(0.0, 0.1, 0.3);
    if(prd.depth == 0)
        prd.hitValue = pcRay.clearColor.xyz * 0.8;
    else
        prd.hitValue = vec3(0.01f);//vec3(0.01);
    prd.depth = 100;
}