#include "host_device.h"
#include "common_layouts.glsl"
#include "globals.glsl"

#extension GL_EXT_nonuniform_qualifier : enable

vec3 computePhongDiffuse(GltfPBRMaterial mat, vec3 lightDir, vec3 normal)
{
  float dotNL = max(dot(normal, lightDir), 0.0);
  return mat.pbrBaseColorFactor.xyz * dotNL;
}

vec3 computePhongSpecular(GltfPBRMaterial mat, vec3 viewDir, vec3 lightDir, vec3 normal)
{
  const float kPi        = 3.14159265;
  const float kShininess = 60.0;

  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  vec3        V                   = normalize(-viewDir);
  vec3        R                   = reflect(-lightDir, normal);
  float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  return vec3(specular);
}

vec3 pbrGetBaseColor(GltfPBRMaterial mat, vec2 texCoord)
{
  vec3 color = mat.pbrBaseColorFactor.xyz;
  if (mat.pbrBaseColorTexture > - 1)
    color *= texture(textureSamplers[nonuniformEXT(mat.pbrBaseColorTexture)], texCoord).rgb;
  return color;
}

void pbrGetMetallicRoughness(GltfPBRMaterial mat, vec2 texCoord, out float metallicFactor, out float roughnessFactor)
{
  metallicFactor = mat.metallicFactor;
  roughnessFactor = mat.roughnessFactor;
  if (mat.metallicRoughnessTexture > - 1)
  {
    vec3 metallicRoughness = texture(textureSamplers[nonuniformEXT(mat.metallicRoughnessTexture)], texCoord).rgb;
    // roughness encoded in green channel, metalness encoded in blue channel
    roughnessFactor *= metallicRoughness.g;
    metallicFactor *= metallicRoughness.b;
  }
}

vec3 pbrGetEmissive(GltfPBRMaterial mat, vec2 texCoord)
{
  vec3 emittance = mat.emissiveFactor;
  if (mat.emissiveTexture > - 1)
    emittance *= texture(textureSamplers[nonuniformEXT(mat.emissiveTexture)], texCoord).rgb;
  return emittance;
}

float getNDF_GGXTR(vec3 N, vec3 H, float alpha)
{
    float a2 = alpha * alpha;
    float NH = dot(N, H);
    if (NH <= 0.0f)
        return 0.0f;

    float NH2 = NH * NH;
    float d = NH2 * (a2 - 1.0f) + 1.0f;

    return a2 * M_INV_PI / (d * d + 1e-4);
}

float getG_SchlickGGX(float NV, float k)
{
    return NV / (NV * (1.0f - k) + k);
}

float getG_Smith(vec3 N, vec3 V, vec3 L, float k)
{
    float NV = abs(dot(N, V));
    float NL = abs(dot(N, L));
    return getG_SchlickGGX(NV, k) * getG_SchlickGGX(NL, k);
}

vec3 getF_Schlick(vec3 H, vec3 V, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - abs(dot(H, V)), 5.0f);
}

vec3 getSpecularBRDF_Cook_Torrance(vec3 N, vec3 H, vec3 V, vec3 L, vec3 F0, float roughness, float metalness)
{
    float alpha = roughness * roughness;
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    
    float D = getNDF_GGXTR(N, H, alpha);
    float G = getG_Smith(N, V, L, k);
    vec3  F = getF_Schlick(H, V, F0);

    float down = 4.0f * abs(dot(V, N)) * abs(dot(L,N)) + 1e-4f;
    return  D * F * G / down;
}

vec3 getSpecularBRDF_over_pdf_Cook_Torrance(vec3 N, vec3 H, vec3 V, vec3 L, vec3 F0, float roughness, float metalness, float ratio)
{
    float alpha = roughness * roughness;
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    
    float pdf = (1.0f - ratio) * dot(N, H) / (4.0f * dot(L, H) + 1e-4);
    float G = getG_Smith(N, V, L, k);
    vec3  F = getF_Schlick(H, V, F0);

    float down = 4.0f * abs(dot(V, N)) * abs(dot(L,N)) + 1e-4f;
    return  (F * G / down) / pdf;
}

vec3 computePBR_BRDF(vec3 N, vec3 V, vec3 L, vec3 H, GltfPBRMaterial mat, vec2 texCoord)
{
    vec3 baseColor = pbrGetBaseColor(mat, texCoord);
    float metalness, roughness;
    pbrGetMetallicRoughness(mat, texCoord, metalness, roughness);
    vec3 emissive = pbrGetEmissive(mat, texCoord);
    
    // Specular
    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, baseColor, metalness);
    vec3 F = getF_Schlick(H, V, F0);
    vec3 f_cook_torrance = getSpecularBRDF_Cook_Torrance(N, H, V, L, F0, roughness, metalness);

    // The specular coefficient kS is F;
    vec3 kD = vec3(1.0f) - F;
    kD *= 1.0f - metalness; // a more metallic object should have not have much color
    vec3 f_lambert =  baseColor * M_INV_PI;
    vec3 diffuse = kD * f_lambert;

    vec3 BRDF = diffuse + f_cook_torrance;
    //vec3 Lo = /*emissive + */ BRDF;

    return BRDF;
}

vec3 directLight(GltfLight light, vec3 P, vec3 N, vec3 V, GltfPBRMaterial mat, vec2 texCoord, out vec3 Li, out float cosTheta)
{
    if (light.type == 0)    // point light
    {
        vec3 Ldir = light.position - P;
        float d = length(Ldir);
        vec3 L = Ldir / d;
        vec3 H = normalize(L + V);
        float attenuation = d * d;
        Li = light.color * light.intensity / attenuation;

        cosTheta = max(dot(L, N), 0.0f);
        if (cosTheta > 0.0f)
            return computePBR_BRDF(N, V, L, H, mat, texCoord);
        //return vec3(H);
    }

    return vec3(0.0f);
}

// Oct packing
vec2 _NRD_EncodeUnitVector( vec3 v, const bool bSigned )
{
    v /= dot( abs( v ), vec3(1.0f) );

    vec2 octWrap = ( 1.0 - abs( v.yx ) ) * ( step( 0.0, v.xy ) * 2.0 - 1.0 );
    v.xy = v.z >= 0.0 ? v.xy : octWrap;

    return bSigned ? v.xy : v.xy * 0.5 + 0.5;
}

vec4 NRD_FrontEnd_PackNormalAndRoughness( vec3 N, float roughness, float materialID )
{
    vec4 p;

    p.xy = _NRD_EncodeUnitVector( N, false );
    p.z = roughness;
    p.w = clamp( materialID / 3.0f, 0.0f, 1.0f );

    return p;
}

vec3 _NRD_DecodeUnitVector( vec2 p, const bool bSigned, const bool bNormalize )
{
    p = bSigned ? p : ( p * 2.0 - 1.0 );

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
    float t = clamp( -n.z , 0.0f, 1.0f);
    n.xy -= t * ( step( 0.0, n.xy ) * 2.0 - 1.0 );

    return bNormalize ? normalize( n ) : n;
}

vec4 NRD_FrontEnd_UnpackNormalAndRoughness( vec4 p, out float materialID )
{
    vec4 r;
    r.xyz = _NRD_DecodeUnitVector( p.xy, false, false );
    r.w = p.z;

    materialID = p.w;
 
    r.xyz = normalize( r.xyz );

    return r;
}

vec3 _NRD_LinearToYCoCg( vec3 color )
{
    float Y = dot( color, vec3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, vec3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, vec3( -0.25, 0.5, -0.25 ) );

    return vec3( Y, Co, Cg );
}

vec3 _NRD_YCoCgToLinear( vec3 color )
{
    float t = color.x - color.z;

    vec3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;

    return max( r, 0.0 );
}

#define NRD_FP16_MIN     1e-7 // min allowed hitDist (0 = no data)
#define NRD_FP16_MAX     65504.0

vec4 REBLUR_FrontEnd_PackRadianceAndNormHitDist( vec3 radiance, float normHitDist, bool sanitize )
{
    if( sanitize )
    {
        //radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0, NRD_FP16_MAX );
        radiance = (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)) || (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) 
            ? vec3(0) : clamp( radiance, 0, NRD_FP16_MAX );
        normHitDist = ( isnan( normHitDist ) || isinf( normHitDist ) ) ? 0 : clamp( normHitDist , 0.0f, 1.0f);
    }

    // "0" is reserved to mark "no data" samples, skipped due to probabilistic sampling
    if( normHitDist != 0 )
        normHitDist = max( normHitDist, NRD_FP16_MIN );

    radiance = _NRD_LinearToYCoCg( radiance );

    return vec4( radiance, normHitDist );
}

vec4 REBLUR_BackEnd_UnpackRadianceAndNormHitDist( vec4 data )
{
    data.xyz = _NRD_YCoCgToLinear( data.xyz );

    return data;
}

// Hit distance normalization
float _REBLUR_GetHitDistanceNormalization( float viewZ, vec4 hitDistParams, float roughness)
{
    return ( hitDistParams.x + abs( viewZ ) * hitDistParams.y ) * mix( 1.0, hitDistParams.z, 
        clamp( exp2( hitDistParams.w * roughness * roughness ), 0.0f, 1.0f ) );
}

float REBLUR_FrontEnd_GetNormHitDist( float hitDist, float viewZ, vec4 hitDistParams, float roughness)
{
    float f = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, roughness );

    return clamp( hitDist / f, 0.0f, 1.0f);
}

// Scales normalized hit distance back to real length
float REBLUR_GetHitDist( float normHitDist, float viewZ, vec4 hitDistParams, float roughness )
{
    float scale = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, roughness );

    return normHitDist * scale;
}