#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "gltf.glsl"
#include "random.glsl"
#include "raycommon.glsl"
#include "host_device.h"

// Barycentric coordinates
hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT hitPayload prd;
//layout(location = 1) rayPayloadEXT shadowPayload prdShadow;

layout(set = 1, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = ePrimLookup) readonly buffer _InstanceInfo {PrimMeshInfo primInfo[];};

layout(buffer_reference, scalar) readonly buffer Vertices  { vec3 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uint i[]; };
layout(buffer_reference, scalar) readonly buffer Normals   { vec3 n[]; };
layout(buffer_reference, scalar) readonly buffer Tangents  { vec4 tg[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords { vec2 t[]; };

#include "common_layouts.glsl"

void main()
{
  // ivec3 ind = indices.i[gl_PrimitiveID];
  PrimMeshInfo pinfo = primInfo[gl_InstanceCustomIndexEXT];

  uint indexOffset  = pinfo.indexOffset + (3 * gl_PrimitiveID);
  uint vertexOffset = pinfo.vertexOffset;
  uint matIndex     = max(0, pinfo.materialIndex);

  //Materials gltfMat   = GltfMaterials(sceneDesc.materialAddress);
  Vertices  vertices  = Vertices(sceneDesc.vertexAddress);
  Indices   indices   = Indices(sceneDesc.indexAddress);
  Normals   normals   = Normals(sceneDesc.normalAddress);
  Tangents  tangents  = Tangents(sceneDesc.tangentAddress);
  TexCoords texCoords = TexCoords(sceneDesc.uvAddress);
  GltfMaterials materials = GltfMaterials(sceneDesc.materialAddress);
  GltfLights lights = GltfLights(sceneDesc.lightAddress);

  ivec3 triangleIndex = ivec3(indices.i[indexOffset + 0], indices.i[indexOffset + 1], indices.i[indexOffset + 2]);
  triangleIndex += ivec3(vertexOffset);

  const vec3 v0 = vertices.v[triangleIndex.x];
  const vec3 v1 = vertices.v[triangleIndex.y];
  const vec3 v2 = vertices.v[triangleIndex.z];

  const vec3 n0 = normals.n[triangleIndex.x];
  const vec3 n1 = normals.n[triangleIndex.y];
  const vec3 n2 = normals.n[triangleIndex.z];

  const vec3 tg0 = tangents.tg[triangleIndex.x].xyz;
  const vec3 tg1 = tangents.tg[triangleIndex.y].xyz;
  const vec3 tg2 = tangents.tg[triangleIndex.z].xyz;

  const vec2 uv0 = texCoords.t[triangleIndex.x];
  const vec2 uv1 = texCoords.t[triangleIndex.y];
  const vec2 uv2 = texCoords.t[triangleIndex.z];

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  
  // vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  const vec3 pos      = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));
  const vec3 nrm      = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
  const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));
  const vec3 tag      = normalize(tg0 * barycentrics.x + tg1 * barycentrics.y + tg2 * barycentrics.z);
  vec3 worldTag       = normalize(vec3(tag * gl_WorldToObjectEXT));
  worldTag            = normalize(worldTag - dot(worldTag, worldNrm) * worldNrm);
  vec3 worldBin       = tangents.tg[triangleIndex.x].w * cross(worldNrm, worldTag);
  const vec2 texCoord = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
  
  GltfPBRMaterial   mat       = materials.m[matIndex];
  vec3              emittance = vec3(0.0f);
  if (prd.depth == 0 || prd.isSpecular)
  {
    emittance = mat.emissiveFactor;
    if (mat.emissiveTexture > -1) 
      emittance *= texture(textureSamplers[nonuniformEXT(mat.emissiveTexture)], texCoord).xyz;
  }
  //emittance = vec3(0);
  //if (matIndex == 4 || matIndex == 0)//|| matIndex == 14 || matIndex == 15 || matIndex == 16)
  //  emittance = vec3(10);

  vec3 tangent, binormal;
  tangent = worldTag;
  binormal = worldBin;
  //createCoordinateSystem(worldNrm, tangent, binormal);
  vec3 texNormal = worldNrm;
  vec3 ffnormal  = dot(worldNrm, prd.rayDirection) <= 0.0 ? worldNrm : -worldNrm;
  mat3 TBN = mat3(tangent, binormal, texNormal);
  if (mat.normalTexture > -1)
  {
    texNormal = normalize(texture(textureSamplers[nonuniformEXT(mat.normalTexture)], texCoord).xyz * 2.0f - 1.0f);
    texNormal = normalize(TBN * texNormal);
    createCoordinateSystem(texNormal, tangent, binormal);
    TBN = mat3(tangent, binormal, texNormal);
  }
  //prd.hitValue = texNormal;//worldPos; //emittance
  //prd.depth = 100;
  //return;

  vec3 baseColor = pbrGetBaseColor(mat, texCoord);
  float metalness, roughness;
  pbrGetMetallicRoughness(mat, texCoord, metalness, roughness);
  //vec3 F0 = vec3(0.04f);
  //F0 = mix(F0, baseColor, metalness);
  //vec3 F = getF_Schlick(H, V, F0);

  vec3 rayOrigin = worldPos;
  vec3 rayDirection;
  
  float pdf;
  vec3 BRDF;
  vec3 V = normalize(-gl_WorldRayDirectionEXT);
  vec3 N = texNormal;
  
  //float ratio = (1.0f - metalness) * (1.0f - roughness) + roughness;
  float ratio = 0.5f * (1.0f - metalness);
  roughness = clamp(roughness, 0.01f, 0.99f);
  metalness = clamp(metalness, 0.01f, 0.99f);
  float r1 = rnd(prd.seed);
  if (r1 < ratio)
  {
    // Sample diffuse (lambertian)
    prd.isSpecular = false;

    // Sample shadow ray + direct lighting
    int random_index = int(rnd(prd.seed) * float(pcRay.lightsCount));    // TODO: check possiblity of out of bounds index

    GltfLight light = lights.l[random_index];
    vec3 lightDir = light.position - worldPos;
    float lightDistance = length(lightDir);
    vec3 L = normalize(lightDir);

    prd.lightDist = lightDistance;
    prd.shadowRayDir = L;
    
    /*vec3  shadowRayDir =  L;
    float tMin   = 0.1f;
    float tMax   = lightDistance - tMin;
    uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    
    prdShadow.isHit = true;
    traceRayEXT(topLevelAS,
        flags,
        0xFF,
        0,
        0,
        1, // miss index
        rayOrigin,
        tMin,
        shadowRayDir,
        tMax,
        1 // payload
    );
    */
    if (dot(L, texNormal) <= 0 /*|| prdShadow.isHit*/)
        emittance += vec3(0);
    else
    {
        vec3 Li;
        float cosTheta;
        vec3 BRDF = directLight(light, worldPos, texNormal, V, mat, texCoord, Li, cosTheta);
        emittance += pcRay.lightsCount * BRDF * Li * cosTheta;//texNormal * 0.5f + 0.5f;
    }
    // Sample indirect light
    rayDirection = normalize(samplingHemisphere(prd.seed, tangent, binormal, texNormal));
    pdf = ratio * dot(rayDirection, texNormal) * M_INV_PI;
    //if (dot(rayDirection, worldNrm) < 0)
    //{
    //  cosTheta = 0;
    //  prd.depth = 100;
    //}
    BRDF = (1.0f - metalness) * baseColor * M_INV_PI;// mat.pbrBaseColorFactor.xyz;
    //if (mat.pbrBaseColorTexture > - 1)
    //  BRDF *= texture(textureSamplers[nonuniformEXT(mat.pbrBaseColorTexture)], texCoord).xyz;
  }
  else
  {
    // Sample specular
    prd.isSpecular = true;
    float alpha = roughness * roughness;
    vec3 H = normalize(TBN * samplingNDF_GGXTR(prd.seed, alpha * alpha));
    vec3 L = normalize(reflect(-V, H));
    rayDirection = L;

    // For extremely specular surfaces (roughness == 0) it generates really high values
    //float pdf_H = getNDF_GGXTR(N, H, alpha);
    //pdf = (1.0f - ratio) * pdf_H * dot(N, H) / (4.0f * dot(L, H) + 1e-4);

    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, baseColor, metalness);
    //BRDF = getSpecularBRDF_Cook_Torrance(N, H, V, L, F0, roughness, metalness);

    pdf = 1.0f;
    BRDF = getSpecularBRDF_over_pdf_Cook_Torrance(N, H, V, L, F0, roughness, metalness, ratio);
  }
  float cosTheta = dot(rayDirection, texNormal);
  
  
  //prd.depth = 100;
  //if (tangents.tg[triangleIndex.x].w == tangents.tg[triangleIndex.y].w && tangents.tg[triangleIndex.x].w == tangents.tg[triangleIndex.z].w)
  //  emittance = vec3(tangents.tg[triangleIndex.x].w * barycentrics.x + tangents.tg[triangleIndex.y].w * barycentrics.y + tangents.tg[triangleIndex.z].w * barycentrics.z) * 0.5f + 0.5f;//vec3(1.0f);


  prd.rayOrigin    = rayOrigin;
  prd.rayDirection = rayDirection;
  prd.hitValue     = emittance; //emittance
  prd.weight       = BRDF * cosTheta / pdf;
  return;

  /*

  vec3 incoming = prd.hitValue;
    
  prd.hitValue = emittance + (BRDF * incoming * cos_theta / p);
  
  vec3 specular = vec3(0);
  float attenuation = 1;

  if (dot(worldNrm, L) > 0)
  {
    float tMin   = 0.001;
    float tMax   = lightDistance;
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3  rayDir = L;
    uint  flags  =
         gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | 
        gl_RayFlagsSkipClosestHitShaderEXT;
    //isShadowed = true;
    prdShadow.isHit = true;
    prdShadow.seed  = prd.seed;
    traceRayEXT(topLevelAS,
        flags,
        0xFF,
        0,
        0,
        1,
        origin,
        tMin,
        rayDir,
        tMax,
        1
    );
    prd.seed = prdShadow.seed;

    if (prdShadow.isHit)
    //if(isShadowed)
    {
      attenuation = 0.3;
    }
    else
    {
      specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, worldNrm);
    }
  }

  prd.hitValue = vec3(lightIntensity * attenuation * (diffuse + specular));*/
}
