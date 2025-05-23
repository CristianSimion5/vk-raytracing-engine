/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE

#ifdef __cplusplus
#include "nvmath/nvmath.h"
// GLSL Type
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = unsigned int;
#else
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

START_BINDING(SceneBindings)
  eGlobals   = 0,  // Global uniform containing camera matrices
  eSceneDesc = 1,
  eTextures  = 2  // Access to textures
END_BINDING();

START_BINDING(RtxBindings)
  eTlas       = 0,  // Top-level acceleration structure
  eOutImage   = 1,  // Ray tracer output image
  ePrimLookup = 2,  // Lookup of objects
  ePosMap     = 3,
  eNormMap    = 4,
  eAccumMap   = 5,
  eRoughMap   = 6,
  eInMV       = 7,
  eInNormRough= 8,
  eInViewZ    = 9,
  eInRadHitD  = 10
END_BINDING();
// clang-format on


// Uniform buffer set at each frame
struct GlobalUniforms
{
  mat4 viewProj;     // Camera view * projection
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix
};

// Push constant structure for the raster
struct PushConstantRaster
{
  mat4  modelMatrix;  // matrix of the instance
  mat4  inverseTransposeMatrix;
  mat4  viewMatrix;
  uint  objIndex;
  int   materialId;
  int   lightsCount;
};


// Push constant structure for the ray tracer
struct PushConstantRay
{
  vec4 clearColor;
  int  frame;
  int  lightsCount;
  int  samples;
  int  depth;
  int  useShadows;
  int  useAO;
  int  useGI;
};

struct PrimMeshInfo
{
  uint indexOffset;
  uint vertexOffset;
  int  materialIndex;
};

struct SceneDesc
{
  uint64_t vertexAddress;
  uint64_t normalAddress;
  uint64_t tangentAddress;
  uint64_t uvAddress;
  uint64_t indexAddress;
  uint64_t materialAddress;
  uint64_t lightAddress;
  uint64_t primInfoAddress;
};

struct GltfPBRMaterial
{
  vec4  pbrBaseColorFactor;
  int   pbrBaseColorTexture;
  float metallicFactor;
  float roughnessFactor;
  int   metallicRoughnessTexture;
  int   normalTexture;
  vec3  emissiveFactor;
  int   emissiveTexture;
};

struct GltfLight
{
    vec3  position;
    vec3  color;
    float intensity;
    int   type;
};

#endif
