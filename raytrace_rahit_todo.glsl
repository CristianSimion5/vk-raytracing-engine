#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "random.glsl"
#include "raycommon.glsl"

#ifdef PAYLOAD_0
layout(location = 0) rayPayloadInEXT hitPayload prd;
#elif defined(PAYLOAD_1)
layout(location = 1) rayPayloadInEXT shadowPayload prd;
#endif

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; };
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; };
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; };
layout(buffer_reference, scalar) buffer MatIndices {int i[]; };
layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;

void main()
{
    ObjDesc objResource   = objDesc.i[gl_InstanceCustomIndexEXT];
    MatIndices matIndices = MatIndices(objResource.materialIndexAddress);
    Materials materials   = Materials(objResource.materialAddress);

    int               matIdx = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat    = materials.m[matIdx];

    if (mat.illum != 4)
      return;   // material is opaque

    if (mat.dissolve == 0.0)
        ignoreIntersectionEXT;
    else if (rnd(prd.seed) > mat.dissolve)
        ignoreIntersectionEXT;
}