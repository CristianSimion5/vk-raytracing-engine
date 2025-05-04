#ifndef COMMON_LAYOUTS
#define COMMON_LAYOUTS

#include "host_device.h"

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

layout(buffer_reference, scalar) readonly buffer GltfMaterials { GltfPBRMaterial m[]; };
layout(buffer_reference, scalar) readonly buffer GltfLights    { GltfLight       l[]; };

layout(binding = eSceneDesc, set = 0) readonly buffer SceneDesc_ { SceneDesc sceneDesc; };
layout(binding = eTextures, set = 0) uniform sampler2D[] textureSamplers;

#endif