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

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"
#include "common_layouts.glsl"
#include "gltf.glsl"
#include "globals.glsl"

layout(push_constant) uniform _PushConstantRaster
{
  PushConstantRaster pcRaster;
};

// clang-format off
// Incoming 
layout(location = 1) in vec3 i_worldPos;
layout(location = 2) in vec3 i_worldNrm;
layout(location = 3) in vec3 i_worldTag;
layout(location = 4) in vec3 i_worldBin;
layout(location = 5) in vec3 i_viewDir;
layout(location = 6) in vec2 i_texCoord;
//layout(location = 7) in mat3 i_tbn;
// Outgoing
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_position;
layout(location = 2) out vec4 o_normal;
layout(location = 3) out vec2 o_roughness;
layout(location = 4) out vec4 o_motionVector;
layout(location = 5) out vec4 o_normalRoughness;
layout(location = 6) out float o_viewZ;
layout(location = 7) out vec4 o_diffRadianceHitD;

layout(binding = 0) uniform _GlobalUniforms
{
  GlobalUniforms uni;
};

// clang-format on


// Generates tangent using dFdx/y functions
mat3 getTBN()
{
    vec3 pos_dx = dFdx(i_worldPos);
    vec3 pos_dy = dFdy(i_worldPos);
    vec2 tex_dx = dFdx(i_texCoord);
    vec2 tex_dy = dFdy(i_texCoord);

    if (length(tex_dx) + length(tex_dy) <= 1e-6) {
        tex_dx = vec2(1.0, 0.0);
        tex_dy = vec2(0.0, 1.0);
    }

    vec3 t_ = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

    vec3 ng = normalize(i_worldNrm);

    vec3 t = normalize(t_ - ng * dot(ng, t_));
    vec3 b = normalize(cross(ng, t));
    
    if (gl_FrontFacing == false)
    {
        t *= -1.0;
        b *= -1.0;
        ng *= -1.0;
    }
    
    return mat3(t, b, ng);
}

vec3 getNormal(int normTexId)
{
  vec3 N = normalize(i_worldNrm);

  if (normTexId > -1)
  {
    vec3 T = normalize(i_worldTag);  // use the interpolated tangent
    vec3 B = normalize(i_worldBin); // use the interpolated binormal
    // Gram-Schmidt
    T = normalize(T - dot(T, N) * N);
    B = normalize(B - dot(B, N) * N - dot(B, T) * T);
    //vec3 B = cross(N, T); // generate binormal in fragment shader, needs correct tangent.w value for sign
    mat3 tbn = mat3(T, B, N);
    //mat3 tbn = getTBN(); // generate tangent and binormal using surface derivatives, can show incorrect results if the tangents in the buffer are not "ordinary"

    vec3 nrm = texture(textureSamplers[nonuniformEXT(normTexId)], i_texCoord).xyz * 2.0f - 1.0f;
    nrm = normalize(nrm);
    nrm = normalize(tbn * nrm);
    
    N = nrm;
  }

  return N;
}


void main()
{
  // Material of the object
  GltfMaterials   gltfMat = GltfMaterials(sceneDesc.materialAddress);
  GltfPBRMaterial mat     = gltfMat.m[pcRaster.materialId];
  GltfLights      lights  = GltfLights(sceneDesc.lightAddress);

  vec3 N = getNormal(mat.normalTexture);

  vec3 baseColor = pbrGetBaseColor(mat, i_texCoord);
  float metalness, roughness;
  pbrGetMetallicRoughness(mat, i_texCoord, metalness, roughness);

  o_motionVector = vec4(0.0f);
  o_normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(N, roughness, float(pcRaster.materialId));
  o_viewZ = (pcRaster.viewMatrix * vec4(i_worldPos, 1.0f)).z;
  o_diffRadianceHitD = vec4(0.0f);
  
  vec3 albedo = (1.0f - metalness) * baseColor;
  o_roughness = vec2(roughness, metalness);

  o_position.xyz = i_worldPos;
  o_normal.xyz   = N;

  // Encode albedo in each unused alpha component
  o_color.w    = albedo.r;
  o_position.w = albedo.g;
  o_normal.w   = albedo.b;

  //vec3 N = normalize(i_worldNrm);
  //o_color = vec4(N * 0.5f + 0.5f, 1);  
  //o_color = vec4(vec2(1.0, 1.0) - i_texCoord, 0, 1);  
  //return;

  // Vector toward light
  /*vec3  L;
  float lightIntensity = pcRaster.lightIntensity;
  if(pcRaster.lightType == 0)
  {
    vec3  lDir     = pcRaster.lightPosition - i_worldPos;
    float d        = length(lDir);
    lightIntensity = pcRaster.lightIntensity / (d * d);
    L              = normalize(lDir);
  }
  else
  {
    L = normalize(pcRaster.lightPosition);
  }


  // Diffuse
  vec3 diffuse = computePhongDiffuse(mat, L, N);
  if(mat.pbrBaseColorTexture > -1)
  {
    uint txtId      = mat.pbrBaseColorTexture;
    vec3 diffuseTxt = texture(textureSamplers[nonuniformEXT(txtId)], i_texCoord).xyz;
    diffuse *= diffuseTxt;
  }

  // Specular
  vec3 specular = computePhongSpecular(mat, i_viewDir, L, N);
  */

  // omega_i (incoming light) = L = light_pos - world_pos
  // omega_o (outgoing light) = V = eye_pos - world_pos
  vec3 V = normalize(-i_viewDir);
  vec3 color = vec3(0.0f);

  vec3  emittance = mat.emissiveFactor;
  if (mat.emissiveTexture > -1) 
    emittance *= texture(textureSamplers[nonuniformEXT(mat.emissiveTexture)], i_texCoord).xyz;
  for (int i = 0; i < pcRaster.lightsCount; i++)
  {
    GltfLight light = lights.l[i];
    vec3 L = normalize(light.position - i_worldPos);
    vec3 lightIntensity = light.color * light.intensity;
    if(light.type == 0)
    {
        vec3 lDir = light.position - i_worldPos;
        float d = length(lDir);
        lightIntensity /= (d * d);
    }
    else
    {
        L = normalize(light.position);
    }
    vec3 H = normalize(L + V);
    float cosTheta = max(dot(L, N), 0.0f);
  
    if (cosTheta > 0.0f)
        color += computePBR_BRDF(N, V, L, H, mat, i_texCoord) * lightIntensity * cosTheta;
  }
  o_color.rgb = emittance + color;
  /*float cosTheta = max(dot(L, N), 0.0f);
  vec3 baseColor = pbrGetBaseColor(mat, i_texCoord);
  
  float metalness, roughness;
  pbrGetMetallicRoughness(mat, i_texCoord, metalness, roughness);
  vec3 emissive = pbrGetEmissive(mat, i_texCoord);
  
  //roughness = max(roughness, 0.001);
  float alpha = roughness * roughness;
  float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
  
  if (light.type == 0)
  {

      float D = getNDF_GGXTR(N, H, alpha);
      float G = getG_Smith(N, V, L, k);
      vec3 F0 = vec3(0.04f);
      F0 = mix(F0, baseColor, metalness);
      vec3 F = getF_Schlick(H, V, F0);
      float down = 4.0f * abs(dot(V, N)) * abs(dot(L,N)) + 1e-4f;
      vec3 f_cook_torrance = D * F * G / down;
  
      // The specular coefficient kS is F;
      vec3 kD = vec3(1.0f) - F;
      //kD *= 1.0f - metalness;   // = mix(vec3(1.0f), vec3(0.0f), metalness);
      vec3 c_diff = mix(baseColor, vec3(0.0f), metalness);
      vec3 f_lambert =  c_diff * M_INV_PI;
      vec3 diffuse = kD * f_lambert; //getPBRDiffuse();

      //specular is cook-torrance
      vec3 BRDF = (diffuse + f_cook_torrance) * lightIntensity * cosTheta;
      //o_color = vec4(vec3(BRDF + emissive), 1);
  }
  else
  {
    const vec3 black = vec3(0.0f);
    vec3 c_diff = mix(baseColor, black, metalness);
    vec3 F0 = mix(vec3(0.04f), baseColor, metalness);

    vec3 F = F0 + (1 - F0) * pow(1 - abs(dot(V,H)), 5);

    vec3 f_diffuse = (1 - F) * (1.0 / M_PI) * c_diff; 
    vec3 f_specular = F * getNDF_GGXTR(N, H, alpha) * getG_Smith(N, V, L, k) / (4 * abs(dot(V, N)) * abs(dot(L, N)));
    vec3 material_brdf = f_diffuse + f_specular;

    o_color = vec4(material_brdf * light.intensity * cosTheta + emissive, 1);
  }
  */
  // Result
  //o_color = vec4(lightIntensity * (diffuse + specular), 1);
}
