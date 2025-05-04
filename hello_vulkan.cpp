/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sstream>


#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

 //#include "obj_loader.h"
//#include "stb_image.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/gltfscene.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
    AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
    m_alloc.init(instance, device, physicalDevice);
    m_debug.setup(m_device);
    m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
    // Prepare new UBO contents on host.
    const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
    GlobalUniforms hostUBO = {};
    const auto& view = CameraManip.getMatrix();
    const auto& proj = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
    // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

    hostUBO.viewProj = proj * view;
    hostUBO.viewInverse = nvmath::invert(view);
    hostUBO.projInverse = nvmath::invert(proj);

    // UBO on the device, and what stages access it.
    VkBuffer deviceUBO = m_bGlobals.buffer;
    auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

    // Ensure that the modified UBO is not visible to previous frames.
    VkBufferMemoryBarrier beforeBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    beforeBarrier.buffer = deviceUBO;
    beforeBarrier.offset = 0;
    beforeBarrier.size = sizeof(hostUBO);
    vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
        nullptr, 1, &beforeBarrier, 0, nullptr);


    // Schedule the host-to-device upload. (hostUBO is copied into the cmd
    // buffer so it is okay to deallocate when the function returns).
    vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

    // Making sure the updated UBO will be visible.
    VkBufferMemoryBarrier afterBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    afterBarrier.buffer = deviceUBO;
    afterBarrier.offset = 0;
    afterBarrier.size = sizeof(hostUBO);
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
        nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
    auto nbTxt = static_cast<uint32_t>(m_textures.size());

    // Camera matrices
    m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    // Obj descriptions
    /*m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                       | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);*/
    // Textures
    m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt,
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    // Scene
    m_descSetLayoutBind.addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
    m_descPool = m_descSetLayoutBind.createPool(m_device, 1);
    m_descSet = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
    std::vector<VkWriteDescriptorSet> writes;

    // Camera matrices and scene description
    VkDescriptorBufferInfo dbiUnif{ m_bGlobals.buffer, 0, VK_WHOLE_SIZE };
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

    /*VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));*/

    VkDescriptorBufferInfo sceneDesc{ m_sceneDesc.buffer, 0, VK_WHOLE_SIZE };
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eSceneDesc, &sceneDesc));

    // All texture samplers
    std::vector<VkDescriptorImageInfo> diit;
    for (auto& texture : m_textures)
    {
        diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

    // Writing the information
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
    VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster) };

    // Creating the Pipeline Layout
    VkPipelineLayoutCreateInfo createInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    createInfo.setLayoutCount = 1;
    createInfo.pSetLayouts = &m_descSetLayout;
    createInfo.pushConstantRangeCount = 1;
    createInfo.pPushConstantRanges = &pushConstantRanges;
    vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


    // Creating the Pipeline
    std::vector<std::string>                paths = defaultSearchPaths;
    nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
    gpb.depthStencilState.depthTestEnable = true;
    gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    // BUFFER: ADD HERE
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    // DENOISER: ADD HERE
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());
    gpb.addBlendAttachmentState(gpb.makePipelineColorBlendAttachmentState());

    gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
    gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
    gpb.addBindingDescriptions({ {0, sizeof(nvmath::vec3f)}, {1, sizeof(nvmath::vec3f)}, {2, sizeof(nvmath::vec4f)}, {3, sizeof(nvmath::vec2f)} });
    gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {2, 2, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
      {3, 3, VK_FORMAT_R32G32_SFLOAT, 0},
        });

    m_graphicsPipeline = gpb.createPipeline();
    m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

void HelloVulkan::loadGltfMaterials(const VkCommandBuffer& cmdBuf, VkBufferUsageFlags flags)
{
    std::vector<GltfPBRMaterial> shadeMaterials;
    for (auto& m : m_gltfScene.m_materials)
    {
        shadeMaterials.emplace_back(GltfPBRMaterial{
            m.baseColorFactor,
            m.baseColorTexture,
            m.metallicFactor,
            m.roughnessFactor,
            m.metallicRoughnessTexture,
            m.normalTexture,
            m.emissiveFactor,
            m.emissiveTexture 
        });
    }
    m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, flags);
}

void HelloVulkan::loadGltfLights(const VkCommandBuffer& cmdBuf, VkBufferUsageFlags flags)
{
    std::vector<GltfLight> lights;
    std::map<std::string, int> tinygltfToInt;
    // TODO: check why { std::string, int } does not work
    tinygltfToInt.emplace(std::make_pair("point", 0));
    tinygltfToInt.emplace(std::make_pair("directional", 1));
    tinygltfToInt.emplace(std::make_pair("spot", 2));

    for (auto& l : m_gltfScene.m_lights)
    {
        nvmath::vec3f color(l.light.color[0], l.light.color[1], l.light.color[2]);
        lights.emplace_back(GltfLight{
            nvmath::vec3f(l.worldMatrix.col(3)),
            color,
            static_cast<float>(l.light.intensity),
            tinygltfToInt[l.light.type]
        });
    }
     
    // Allocate a dummy light since buffer should not be empty
    if (lights.empty())
    {
        //lights.emplace_back(GltfLight{
        //    nvmath::vec3f(9.5f, 5.0f, 3.0f),    // position
        //    nvmath::vec3f(1.0f),    // color
        //    10.0f,                  // intensity
        //    0                       // type, default point light
        //});
        lights.emplace_back(GltfLight{
            nvmath::vec3f(1.0f, 5.0f, -1.33f),    // position
            nvmath::vec3f(1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        //lights.emplace_back(GltfLight{
        //    nvmath::vec3f(-9.3, 6.4, -4),    // position
        //    nvmath::vec3f(1.0f),    // color
        //    10.0f,                  // intensity
        //    0                       // type, default point light
        //    });
        //lights.emplace_back(GltfLight{
        //    nvmath::vec3f(-10.5, 0.75, -4.64),    // position
        //    nvmath::vec3f(1.0f),    // color
        //    10.0f,                  // intensity
        //    0                       // type, default point light
        //    });
        //lights.emplace_back(GltfLight{
        //    nvmath::vec3f(8.63, 1.11, 1.71),    // position
        //    nvmath::vec3f(1.0f),    // color
        //    10.0f,                  // intensity
        //    0                       // type, default point light
        //    });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(0, 3, 67),    // position
            nvmath::vec3f(1.0f, 0.01f, 0.1f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(-1.3, 7.62, 59),    // position
            nvmath::vec3f(1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(2.4, 2.05, 40.6),    // position
            nvmath::vec3f(1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(-0.33, 6.85, 30),    // position
            nvmath::vec3f(1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(-6.2, 9.6, 20.18),    // position
            nvmath::vec3f(1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(-0.23, 6.93, 12.21),    // position
            nvmath::vec3f(1.0, 1.0f, 0.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
        lights.emplace_back(GltfLight{
            nvmath::vec3f(0.24, 3.03, 49.94),    // position
            nvmath::vec3f(0.0f, 0.0f, 1.0f),    // color
            50.0f,                  // intensity
            0                       // type, default point light
            });
    }
    m_lightBuffer = m_alloc.createBuffer(cmdBuf, lights, flags);
    m_pcRaster.lightsCount = lights.size();
    m_pcRay.lightsCount = lights.size();
}

void HelloVulkan::loadGltfScene(const std::string& filename)
{
    tinygltf::Model    tmodel;
    tinygltf::TinyGLTF tcontext;
    std::string        warn, error;

    if (nvh::endsWith(filename, ".gltf"))
    {
        if (!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
            assert(!"Error while loading gltf scene");
    }
    else
    {
        if (!tcontext.LoadBinaryFromFile(&tmodel, &error, &warn, filename))
            assert(!"Error while loading binary scene");
    }

    m_gltfScene.importMaterials(tmodel);
    m_gltfScene.importDrawableNodes(tmodel,
        nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);

    nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
    VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

    VkBufferUsageFlags flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags rayTracingFlags = flags | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
    m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
    m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags);
    m_tangentBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_tangents, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags);
    m_uvBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags);

    // Load the materials
    loadGltfMaterials(cmdBuf, flags);
    loadGltfLights(cmdBuf, flags);

    std::vector<PrimMeshInfo> primLookup;
    for (auto& primMesh : m_gltfScene.m_primMeshes)
    {
        primLookup.push_back({ primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex });
    }
    m_primInfo = m_alloc.createBuffer(cmdBuf, primLookup, flags);

    SceneDesc sceneDesc;
    sceneDesc.vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
    sceneDesc.indexAddress = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);
    sceneDesc.normalAddress = nvvk::getBufferDeviceAddress(m_device, m_normalBuffer.buffer);
    sceneDesc.tangentAddress = nvvk::getBufferDeviceAddress(m_device, m_tangentBuffer.buffer);
    sceneDesc.uvAddress = nvvk::getBufferDeviceAddress(m_device, m_uvBuffer.buffer);
    sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
    sceneDesc.lightAddress = nvvk::getBufferDeviceAddress(m_device, m_lightBuffer.buffer);
    sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
    m_sceneDesc = m_alloc.createBuffer(cmdBuf, sizeof(SceneDesc), &sceneDesc, flags);

    createTextureImages(cmdBuf, tmodel);
    cmdBufGet.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();

    NAME_VK(m_vertexBuffer.buffer);
    NAME_VK(m_indexBuffer.buffer);
    NAME_VK(m_normalBuffer.buffer);
    NAME_VK(m_tangentBuffer.buffer);
    NAME_VK(m_uvBuffer.buffer);
    NAME_VK(m_materialBuffer.buffer);
    NAME_VK(m_lightBuffer.buffer);
    NAME_VK(m_primInfo.buffer);
    NAME_VK(m_sceneDesc.buffer);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
    m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}


//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//


// Albedo and Emissive maps should be in SRGB, searches for correspondence between gltf image-texture-material
// TODO: make this function more efficient and more robust
// There is already a function in nvpro (gltf_scene_vk.cpp) that does this
VkFormat getImageFormat(size_t i, const tinygltf::Model& gltfModel)
{
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    int texId = -1;
    for (size_t j = 0; j < gltfModel.textures.size(); j++)
    {
        if (gltfModel.textures[j].source == i)
        {
            texId = j;
            break;
        }
    }
    if (texId > -1)
    {
        for (const auto& material : gltfModel.materials)
        {
            if (material.pbrMetallicRoughness.baseColorTexture.index == texId ||
                material.emissiveTexture.index == texId)
            {
                format = VK_FORMAT_R8G8B8A8_SRGB;
                break;
            }
        }
    }

    return format;
}

void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
    // TODO: sampler data should be take from gltfModel
    VkSamplerCreateInfo samplerCreateInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.anisotropyEnable = VK_TRUE;
    samplerCreateInfo.maxAnisotropy = 4;
    samplerCreateInfo.maxLod = FLT_MAX;

    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    auto addDefaultTexture = [this]() {
        // Make dummy image(1,1), needed as we cannot have an empty array
        nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
        std::array<uint8_t, 4>   white = { 255, 255, 255, 255 };

        VkSamplerCreateInfo sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(VkExtent2D{ 1, 1 }), sampler));
        m_debug.setObjectName(m_textures.back().image, "dummy");
    };

    if (gltfModel.images.empty())
    {
        addDefaultTexture();
        return;
    }

    //m_textures.reserve(gltfModel.images.size());
    std::vector<nvvk::Image> images;
    std::vector<VkImageViewCreateInfo> imageViewInfos;

    images.reserve(gltfModel.images.size());
    imageViewInfos.reserve(gltfModel.images.size());
    for (size_t i = 0; i < gltfModel.images.size(); i++)
    {
        auto& gltfimage = gltfModel.images[i];
        void* buffer = &gltfimage.image[0];
        VkDeviceSize bufferSize = gltfimage.image.size();
        auto         imgSize = VkExtent2D{ (uint32_t)gltfimage.width, (uint32_t)gltfimage.height };

        if (bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
        {
            addDefaultTexture();
            continue;
        }

        format = getImageFormat(i, gltfModel);

        VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

        //nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
        images.emplace_back(m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo));
        nvvk::cmdGenerateMipmaps(cmdBuf, images[i].image, format, imgSize, imageCreateInfo.mipLevels);
        //VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        
        imageViewInfos.emplace_back(nvvk::makeImageViewCreateInfo(images[i].image, imageCreateInfo));
    }

    m_textures.reserve(gltfModel.textures.size());
    for (size_t i = 0; i < gltfModel.textures.size(); i++)
    {
        int imgIdx = gltfModel.textures[i].source;
        m_textures.emplace_back(m_alloc.createTexture(images[imgIdx], imageViewInfos[imgIdx], samplerCreateInfo));

        m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)));
    }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

    m_alloc.destroy(m_bGlobals);

    m_alloc.destroy(m_vertexBuffer);
    m_alloc.destroy(m_normalBuffer);
    m_alloc.destroy(m_tangentBuffer);
    m_alloc.destroy(m_uvBuffer);
    m_alloc.destroy(m_indexBuffer);
    m_alloc.destroy(m_materialBuffer);
    m_alloc.destroy(m_lightBuffer);
    m_alloc.destroy(m_primInfo);
    m_alloc.destroy(m_sceneDesc);

    for (auto& t : m_textures)
    {
        m_alloc.destroy(t);
    }

    //#Post
    m_alloc.destroy(m_offscreenColor);
    m_alloc.destroy(m_offscreenDepth);
    //BUFFER: destroy here
    m_alloc.destroy(m_positionTexture);
    m_alloc.destroy(m_normalTexture);
    m_alloc.destroy(m_accumulatedTexture);
    m_alloc.destroy(m_roughnessMap);
    // Denoiser
    m_alloc.destroy(m_inMV.texture);
    m_alloc.destroy(m_inNormalRoughness.texture);
    m_alloc.destroy(m_inViewZ.texture);
    m_alloc.destroy(m_inDiffRadianceHitDist.texture);
    m_alloc.destroy(m_outDiffRadianceHitDist.texture);

    vkDestroyPipeline(m_device, m_postPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
    vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
    vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

    m_rtBuilder.destroy();
    vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);
    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);

    vkDestroyPipeline(m_device, m_rtPipeline2, nullptr);
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout2, nullptr);

    //m_alloc.destroy(m_rtSBTBuffer);
    m_sbtWrapper.destroy();
    m_sbtWrapper2.destroy();

    m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterizeGltf(const VkCommandBuffer& cmdBuf)
{
    std::vector<VkDeviceSize> offsets = { 0, 0, 0, 0 };

    m_debug.beginLabel(cmdBuf, "Rasterize");

    // Dynamic Viewport
    setViewport(cmdBuf);

    // Drawing all triangles
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    std::vector<VkBuffer> vertexBuffers = { m_vertexBuffer.buffer, m_normalBuffer.buffer, m_tangentBuffer.buffer, m_uvBuffer.buffer };
    vkCmdBindVertexBuffers(cmdBuf, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets.data());
    vkCmdBindIndexBuffer(cmdBuf, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    m_pcRaster.viewMatrix = CameraManip.getMatrix();
    for (auto& node : m_gltfScene.m_nodes)
    {
        auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

        m_pcRaster.modelMatrix = nvmath::scale_mat4(nvmath::vec3f(1.0f)) * node.worldMatrix;
        m_pcRaster.inverseTransposeMatrix = nvmath::transpose(nvmath::inverse(nvmath::scale_mat4(nvmath::vec3f(1.0f)) * node.worldMatrix));
        m_pcRaster.objIndex = node.primMesh;
        m_pcRaster.materialId = primitive.materialIndex;
        vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
            sizeof(PushConstantRaster), &m_pcRaster);
        vkCmdDrawIndexed(cmdBuf, primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
    }

    m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
    resetFrame();
    createOffscreenRender();
    updatePostDescriptorSet();
    updateRtDescriptorSet();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
    m_alloc.destroy(m_offscreenColor);
    m_alloc.destroy(m_offscreenDepth);
    // BUFFER: DESTROY HERE
    m_alloc.destroy(m_positionTexture);
    m_alloc.destroy(m_normalTexture);
    m_alloc.destroy(m_accumulatedTexture);
    m_alloc.destroy(m_roughnessMap);
    // Denoising buffers
    m_alloc.destroy(m_inMV.texture);
    m_alloc.destroy(m_inNormalRoughness.texture);
    m_alloc.destroy(m_inViewZ.texture);
    m_alloc.destroy(m_inDiffRadianceHitDist.texture);
    m_alloc.destroy(m_outDiffRadianceHitDist.texture);

    // Creating the color image
    {
        auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT);


        nvvk::Image           image = m_alloc.createImage(colorCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        VkSamplerCreateInfo   sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        m_offscreenColor = m_alloc.createTexture(image, ivInfo, sampler);
        m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Additional color images for GBuffer
    {
        auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT);

        //BUFFER: ADD HERE
        nvvk::Image           image = m_alloc.createImage(colorCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        VkSamplerCreateInfo   sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        m_positionTexture = m_alloc.createTexture(image, ivInfo, sampler);
        m_positionTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image = m_alloc.createImage(colorCreateInfo);
        ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        m_normalTexture = m_alloc.createTexture(image, ivInfo, sampler);
        m_normalTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image = m_alloc.createImage(colorCreateInfo);
        ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        m_accumulatedTexture = m_alloc.createTexture(image, ivInfo, sampler);
        m_accumulatedTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        auto roughnessCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT);

        image = m_alloc.createImage(roughnessCreateInfo);
        ivInfo = nvvk::makeImageViewCreateInfo(image.image, roughnessCreateInfo);
        m_roughnessMap = m_alloc.createTexture(image, ivInfo, sampler);
        m_roughnessMap.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // denoiser:
    VkSamplerCreateInfo   sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    // Denoising buffers
    {
        auto motionVectorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image = m_alloc.createImage(motionVectorCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, motionVectorCreateInfo);
        m_inMV.texture = m_alloc.createTexture(image, ivInfo, sampler);
        m_inMV.ivInfo = ivInfo; // ivInfo.subresourceRange.aspectMask;
        m_inMV.resourceType = nrd::ResourceType::IN_MV;
        m_inMV.format = nri::ConvertVKFormatToNRI(ivInfo.format); // nri::Format::RGBA16_SFLOAT;
    }
    {
        auto normalRoughnessCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image = m_alloc.createImage(normalRoughnessCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, normalRoughnessCreateInfo);
        m_inNormalRoughness.texture = m_alloc.createTexture(image, ivInfo, sampler);
        m_inNormalRoughness.ivInfo = ivInfo;
        m_inNormalRoughness.resourceType = nrd::ResourceType::IN_NORMAL_ROUGHNESS;
        m_inNormalRoughness.format = nri::ConvertVKFormatToNRI(ivInfo.format); //nri::Format::R10_G10_B10_A2_UNORM
    }
    {
        auto viewZCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image = m_alloc.createImage(viewZCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, viewZCreateInfo);
        m_inViewZ.texture = m_alloc.createTexture(image, ivInfo, sampler);
        m_inViewZ.ivInfo = ivInfo; // ivInfo.subresourceRange.aspectMask;
        m_inViewZ.resourceType = nrd::ResourceType::IN_VIEWZ;
        m_inViewZ.format = nri::ConvertVKFormatToNRI(ivInfo.format);
    }
    {
        auto inDiffRadCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image = m_alloc.createImage(inDiffRadCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, inDiffRadCreateInfo);
        m_inDiffRadianceHitDist.texture = m_alloc.createTexture(image, ivInfo, sampler);
        m_inDiffRadianceHitDist.ivInfo = ivInfo; // ivInfo.subresourceRange.aspectMask;
        m_inDiffRadianceHitDist.resourceType = nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST;
        m_inDiffRadianceHitDist.format = nri::ConvertVKFormatToNRI(ivInfo.format);
    }
    {
        auto outDiffRadCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image = m_alloc.createImage(outDiffRadCreateInfo);
        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, outDiffRadCreateInfo);
        m_outDiffRadianceHitDist.texture = m_alloc.createTexture(image, ivInfo, sampler);
        m_outDiffRadianceHitDist.ivInfo = ivInfo; // ivInfo.subresourceRange.aspectMask;
        m_outDiffRadianceHitDist.resourceType = nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST;
        m_outDiffRadianceHitDist.format = nri::ConvertVKFormatToNRI(ivInfo.format);
    }
    

    // Creating the depth buffer
    auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    {
        nvvk::Image image = m_alloc.createImage(depthCreateInfo);


        VkImageViewCreateInfo depthStencilView{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthStencilView.format = m_offscreenDepthFormat;
        depthStencilView.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        depthStencilView.image = image.image;

        m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
    }

    // Setting the image layout for both color and depth
    {
        nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
        auto              cmdBuf = genCmdBuf.createCommandBuffer();
        nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        // BUFFER: ADD HERE
        nvvk::cmdBarrierImageLayout(cmdBuf, m_positionTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_normalTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_accumulatedTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_roughnessMap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // DENOISER:
        nvvk::cmdBarrierImageLayout(cmdBuf, m_inMV.texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_inNormalRoughness.texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_inViewZ.texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_inDiffRadianceHitDist.texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        //nvvk::cmdBarrierImageLayout(cmdBuf, m_outDiffRadianceHitDist.texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        genCmdBuf.submitAndWait(cmdBuf);
    }

    // Creating a renderpass for the offscreen
    if (!m_offscreenRenderPass)
    {
        m_offscreenRenderPass = nvvk::createRenderPass(m_device, 
            { m_offscreenColorFormat, m_offscreenColorFormat, m_offscreenColorFormat, VK_FORMAT_R16G16_SFLOAT,         //BUFFER: ADD HERE
            m_inMV.ivInfo.format, m_inNormalRoughness.ivInfo.format, m_inViewZ.ivInfo.format, m_inDiffRadianceHitDist.ivInfo.format },    //DENOISER: ADD HERE
            m_offscreenDepthFormat, 1, true,
            true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    }


    // Creating the frame buffer for offscreen
    std::vector<VkImageView> attachments = { m_offscreenColor.descriptor.imageView, 
        //BUFFER: ADD HERE
        m_positionTexture.descriptor.imageView, 
        m_normalTexture.descriptor.imageView,
        m_roughnessMap.descriptor.imageView,
        //DENOISER: ADD HERE
        m_inMV.texture.descriptor.imageView,
        m_inNormalRoughness.texture.descriptor.imageView,
        m_inViewZ.texture.descriptor.imageView,
        m_inDiffRadianceHitDist.texture.descriptor.imageView,
        m_offscreenDepth.descriptor.imageView,
    };

    vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
    VkFramebufferCreateInfo info{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    info.renderPass = m_offscreenRenderPass;
    info.attachmentCount = attachments.size();
    info.pAttachments = attachments.data();
    info.width = m_size.width;
    info.height = m_size.height;
    info.layers = 1;
    vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
    // Push constants in the fragment shader
    VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantPost) };

    // Creating the pipeline layout
    VkPipelineLayoutCreateInfo createInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    createInfo.setLayoutCount = 1;
    createInfo.pSetLayouts = &m_postDescSetLayout;
    createInfo.pushConstantRangeCount = 1;
    createInfo.pPushConstantRanges = &pushConstantRanges;
    vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


    // Pipeline: completely generic, no vertices
    nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
    pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
    pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    m_postPipeline = pipelineGenerator.createPipeline();
    m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
    m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_postDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
    m_postDescPool = m_postDescSetLayoutBind.createPool(m_device);
    m_postDescSet = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.emplace_back(m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor));
    writeDescriptorSets.emplace_back(m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_accumulatedTexture.descriptor));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
    m_debug.beginLabel(cmdBuf, "Post");

    setViewport(cmdBuf);

    m_pcPost.aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
    m_pcPost.useGI = m_pcRay.useGI;
    vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantPost), &m_pcPost.aspectRatio);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
    vkCmdDraw(cmdBuf, 3, 1, 0, 0);


    m_debug.endLabel(cmdBuf);
}


// Ray tracing
void HelloVulkan::initRayTracing()
{
    VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

    m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
    m_sbtWrapper.setup(m_device, m_graphicsQueueIndex, &m_alloc, m_rtProperties);
    m_sbtWrapper2.setup(m_device, m_graphicsQueueIndex, &m_alloc, m_rtProperties);

    m_pcRay.samples = 1;
    m_pcRay.depth = 3;
    m_pcRay.useShadows = true;
    m_pcRay.useAO = true;
    m_pcRay.useGI = false;
    m_pcPost.viewAccumulated = false;
    m_pcPost.rtMode = 0;
    m_pcPost.useGI = m_pcRay.useGI;
}

//auto HelloVulkan::objectToVkGeometryKHR(const ObjModel& model) 
//{
//  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
//  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
//
//  uint32_t maxPrimitiveCount = model.nbIndices / 3;
//
//  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
//  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
//  triangles.vertexData.deviceAddress = vertexAddress;
//  triangles.vertexStride             = sizeof(VertexObj);
//  triangles.indexType                = VK_INDEX_TYPE_UINT32;
//  triangles.indexData.deviceAddress  = indexAddress;
//  triangles.transformData            = {};
//  triangles.maxVertex                = model.nbVertices;
//
//  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
//  asGeom.geometryType           = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
//  asGeom.geometry.triangles     = triangles;
//  asGeom.flags                  = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
//
//  VkAccelerationStructureBuildRangeInfoKHR offset;
//  offset.firstVertex     = 0;
//  offset.primitiveCount  = maxPrimitiveCount;
//  offset.primitiveOffset = 0;
//  offset.transformOffset = 0;
//
//  nvvk::RaytracingBuilderKHR::BlasInput input;
//  input.asGeometry.emplace_back(asGeom);
//  input.asBuildOffsetInfo.emplace_back(offset);
//
//  return input;
//}

auto HelloVulkan::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
    VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
    VkDeviceAddress indexAddress = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);

    uint32_t maxPrimitiveCount = prim.indexCount / 3;

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(nvmath::vec3f);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;
    triangles.transformData = {};
    triangles.maxVertex = prim.vertexCount;

    VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.geometry.triangles = triangles;
    asGeom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

    VkAccelerationStructureBuildRangeInfoKHR offset;
    offset.firstVertex = prim.vertexOffset;
    offset.primitiveCount = maxPrimitiveCount;
    offset.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
    offset.transformOffset = 0;

    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
}

//void HelloVulkan::createBottomLevelAS() 
// {
//   std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
//   allBlas.reserve(m_objModel.size());
//   for(const auto& obj : m_objModel)
//   {
//     auto blas = objectToVkGeometryKHR(obj);
//     allBlas.emplace_back(blas);
//   }
//   m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
// }

void HelloVulkan::createBottomLevelASGltf()
{
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    allBlas.reserve(m_gltfScene.m_primMeshes.size());
    for (const auto& primMesh : m_gltfScene.m_primMeshes)
    {
        auto geo = primitiveToGeometry(primMesh);
        allBlas.emplace_back(geo);
    }
    m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//void HelloVulkan::createTopLevelAs() 
// {
//   std::vector<VkAccelerationStructureInstanceKHR> tlas;
//   tlas.reserve(m_instances.size());
//   for(const HelloVulkan::ObjInstance& inst : m_instances)
//   {
//     VkAccelerationStructureInstanceKHR rayInst{};
//     rayInst.transform                              = nvvk::toTransformMatrixKHR(inst.transform);
//     rayInst.instanceCustomIndex                    = inst.objIndex;
//     rayInst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(inst.objIndex);
//     rayInst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
//     rayInst.mask                                   = 0xFF;
//     rayInst.instanceShaderBindingTableRecordOffset = 0;
//     tlas.emplace_back(rayInst);
//   }
//   m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
// }

void HelloVulkan::createTopLevelAsGltf()
{
    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(m_gltfScene.m_nodes.size());
    for (auto& node : m_gltfScene.m_nodes)
    {
        VkAccelerationStructureInstanceKHR rayInst{};
        rayInst.transform = nvvk::toTransformMatrixKHR(node.worldMatrix);
        rayInst.instanceCustomIndex = node.primMesh;
        rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
        rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        rayInst.mask = 0xFF;
        rayInst.instanceShaderBindingTableRecordOffset = 0;
        tlas.emplace_back(rayInst);
    }
    m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

void HelloVulkan::createRtDescriptorSet()
{
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    // Gltf change
    m_rtDescSetLayoutBind.addBinding(RtxBindings::ePrimLookup, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
    
    // BUFFER: ADD HERE
    m_rtDescSetLayoutBind.addBinding(RtxBindings::ePosMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eNormMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eAccumMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eRoughMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

    // DENOISER:ADD HERE
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eInMV, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eInNormRough, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eInViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    m_rtDescSetLayoutBind.addBinding(RtxBindings::eInRadHitD, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

    m_rtDescPool = m_rtDescSetLayoutBind.createPool(m_device);
    m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

    VkDescriptorSetAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = m_rtDescPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &m_rtDescSetLayout;
    vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);

    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;
    VkDescriptorImageInfo imageInfo{ {}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    // BUFFER: ADD HERE
    VkDescriptorImageInfo posImageInfo{ {}, m_positionTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo normImageInfo{ {}, m_normalTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo accumImageInfo{ {}, m_accumulatedTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo roughImageInfo{ {}, m_roughnessMap.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    // DENOISER: ADD HERE
    VkDescriptorImageInfo mvImageInfo{ {}, m_inMV.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo nrImageInfo{ {}, m_inNormalRoughness.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo vzImageInfo{ {}, m_inViewZ.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo rhImageInfo{ {}, m_inDiffRadianceHitDist.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorBufferInfo primitiveInfoDesc{ m_primInfo.buffer, 0, VK_WHOLE_SIZE };


    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::ePrimLookup, &primitiveInfoDesc));
    // BUFFER: ADD HERE
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::ePosMap, &posImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNormMap, &normImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eAccumMap, &accumImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eRoughMap, &roughImageInfo));
    // DENOISER: ADD HERE
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInMV, &mvImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInNormRough, &nrImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInViewZ, &vzImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInRadHitD, &rhImageInfo));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HelloVulkan::updateRtDescriptorSet()
{
    // BUFFER: ADD HERE
    VkDescriptorImageInfo imageInfo{ {}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo posImageInfo{ {}, m_positionTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo normImageInfo{ {}, m_normalTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo accumImageInfo{ {}, m_accumulatedTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo roughImageInfo{ {}, m_roughnessMap.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    // DENOISER: ADD HERE
    VkDescriptorImageInfo mvImageInfo{ {}, m_inMV.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo nrImageInfo{ {}, m_inNormalRoughness.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo vzImageInfo{ {}, m_inViewZ.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo rhImageInfo{ {}, m_inDiffRadianceHitDist.texture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::ePosMap, &posImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNormMap, &normImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eAccumMap, &accumImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eRoughMap, &roughImageInfo));
    // DENOISER: ADD HERE
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInMV, &mvImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInNormRough, &nrImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInViewZ, &vzImageInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eInRadHitD, &rhImageInfo));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HelloVulkan::createRtPipeline()
{
    enum StageIndices
    {
        eRaygen,
        eMiss,
        eMiss2,
        eClosestHit,
        //eAnyHit,
        //eAnyHit2,
        eShaderGroupCount
    };

    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.pName = "main";

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss2] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;

    /*stage.module        = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace_0.rahit.spv", true, defaultSearchPaths, true));
    stage.stage         = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit]     = stage;*/

    /*stage.module        = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace_1.rahit.spv", true, defaultSearchPaths, true));
    stage.stage         = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit2]    = stage;*/

    VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    m_rtShaderGroups.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    m_rtShaderGroups.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss2;
    m_rtShaderGroups.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    //group.anyHitShader     = eAnyHit;
    m_rtShaderGroups.push_back(group);

    /*group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader     = eAnyHit2;
    m_rtShaderGroups.push_back(group);*/

    VkPushConstantRange pushConstant{ VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                     0, sizeof(PushConstantRay) };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstant;

    std::vector<VkDescriptorSetLayout> rtDescSetLayouts = { m_descSetLayout, m_rtDescSetLayout };
    pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
    pipelineLayoutCreateInfo.pSetLayouts = rtDescSetLayouts.data();

    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);

    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    rayPipelineInfo.pStages = stages.data();

    rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
    rayPipelineInfo.pGroups = m_rtShaderGroups.data();

    if (m_rtProperties.maxRayRecursionDepth <= 10) {
        throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 10)");
    }
    rayPipelineInfo.maxPipelineRayRecursionDepth = 11;
    rayPipelineInfo.layout = m_rtPipelineLayout;

    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);

    m_sbtWrapper.create(m_rtPipeline, rayPipelineInfo);

    for (auto& s : stages)
        vkDestroyShaderModule(m_device, s.module, nullptr);
}

void HelloVulkan::createHybridRtPipeline()
{
    enum StageIndices
    {
        eRaygen,
        eMiss,
        eMiss2,
        eClosestHit,
        //eAnyHit,
        //eAnyHit2,
        eShaderGroupCount
    };

    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.pName = "main";

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceHybrid.rgen.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss2] = stage;

    stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
    stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;

    /*stage.module        = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace_0.rahit.spv", true, defaultSearchPaths, true));
    stage.stage         = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit]     = stage;*/

    /*stage.module        = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace_1.rahit.spv", true, defaultSearchPaths, true));
    stage.stage         = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit2]    = stage;*/

    VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    m_rtShaderGroups2.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    m_rtShaderGroups2.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss2;
    m_rtShaderGroups2.push_back(group);

    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    //group.anyHitShader     = eAnyHit;
    m_rtShaderGroups2.push_back(group);

    /*group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader     = eAnyHit2;
    m_rtShaderGroups2.push_back(group);*/

    VkPushConstantRange pushConstant{ VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                     0, sizeof(PushConstantRay) };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstant;

    // HYBRID: SCHIMBA LAYOUTUL
    std::vector<VkDescriptorSetLayout> rtDescSetLayouts = { m_descSetLayout, m_rtDescSetLayout };
    pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
    pipelineLayoutCreateInfo.pSetLayouts = rtDescSetLayouts.data();

    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout2);

    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    rayPipelineInfo.pStages = stages.data();

    rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups2.size());
    rayPipelineInfo.pGroups = m_rtShaderGroups2.data();

    if (m_rtProperties.maxRayRecursionDepth <= 10) {
        throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 10)");
    }
    rayPipelineInfo.maxPipelineRayRecursionDepth = 11;
    rayPipelineInfo.layout = m_rtPipelineLayout2;

    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline2);

    m_sbtWrapper2.create(m_rtPipeline2, rayPipelineInfo);

    for (auto& s : stages)
        vkDestroyShaderModule(m_device, s.module, nullptr);
}

/*void HelloVulkan::createRtShaderBindingTable()
{
    uint32_t missCount{ 2 };
    uint32_t hitCount{ 2 };
    auto     handleCount = 1 + missCount + hitCount;
    uint32_t handleSize = m_rtProperties.shaderGroupHandleSize;

    uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rtProperties.shaderGroupHandleAlignment);

    m_rgenRegion.stride = nvh::align_up(handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
    m_rgenRegion.size = m_rgenRegion.stride;
    m_missRegion.stride = handleSizeAligned;
    m_missRegion.size = nvh::align_up(missCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
    m_hitRegion.stride = handleSizeAligned;
    m_hitRegion.size = nvh::align_up(hitCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);

    uint32_t dataSize = handleCount * handleSize;
    std::vector<uint8_t> handles(dataSize);
    auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, handleCount, dataSize, handles.data());
    assert(result == VK_SUCCESS);

    VkDeviceSize sbtSize = m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size;
    m_rtSBTBuffer = m_alloc.createBuffer(sbtSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT"));

    VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_rtSBTBuffer.buffer };
    VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(m_device, &info);
    m_rgenRegion.deviceAddress = sbtAddress;
    m_missRegion.deviceAddress = sbtAddress + m_rgenRegion.size;
    m_hitRegion.deviceAddress  = sbtAddress + m_rgenRegion.size + m_missRegion.size;

    auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

    auto* pSBTBuffer = reinterpret_cast<uint8_t*>(m_alloc.map(m_rtSBTBuffer));
    uint8_t* pData{ nullptr };
    uint32_t handleIdx{ 0 };

    pData = pSBTBuffer;
    memcpy(pData, getHandle(handleIdx++), handleSize);

    pData = pSBTBuffer + m_rgenRegion.size;
    for (uint32_t c = 0; c < missCount; c++)
    {
        memcpy(pData, getHandle(handleIdx++), handleSize);
        pData += m_missRegion.stride;
    }

    pData = pSBTBuffer + m_rgenRegion.size + m_missRegion.size;
    for (uint32_t c = 0; c < hitCount; c++)
    {
        memcpy(pData, getHandle(handleIdx++), handleSize);
        pData += m_hitRegion.stride;
    }

    m_alloc.unmap(m_rtSBTBuffer);
    m_alloc.finalizeAndReleaseStaging();
}
*/

void HelloVulkan::pathtrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
    //updateFrame();
    if (m_stopAtMaxFrames && m_pcRay.frame >= m_maxFrames)
    { 
        //m_pcRay.frame = 0;
        return;
    }

    m_debug.beginLabel(cmdBuf, "Path trace");

    m_pcRay.clearColor = clearColor;

    std::vector<VkDescriptorSet> descSets{m_descSet, m_rtDescSet};
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
        (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
    vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
        0, sizeof(PushConstantRay), &m_pcRay);

    auto& regions = m_sbtWrapper.getRegions();
    //vkCmdTraceRaysKHR(cmdBuf, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion, m_size.width, m_size.height, 1);
    vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);
    m_debug.endLabel(cmdBuf);
}

void HelloVulkan::raytraceRasterizedScene(const VkCommandBuffer& cmdBuf)
{
    //updateFrame();
    if (m_stopAtMaxFrames && m_pcRay.frame >= m_maxFrames)
    { 
        //m_pcRay.frame = 0;
        return;
    }

    m_debug.beginLabel(cmdBuf, "Ray trace (hybrid)");

    // HYBRID: set other descriptors
    std::vector<VkDescriptorSet> descSets{m_descSet, m_rtDescSet};
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline2);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout2, 0,
        (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
    vkCmdPushConstants(cmdBuf, m_rtPipelineLayout2,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
        0, sizeof(PushConstantRay), &m_pcRay);

    auto& regions = m_sbtWrapper2.getRegions();
    vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);
    m_debug.endLabel(cmdBuf);
}

void HelloVulkan::populateCommonSettings(nrd::CommonSettings& commonSettings)
{
    size_t matSize = 16 * sizeof(float);
    const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
    const auto& view = CameraManip.getMatrix();
    const auto& proj = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
    memcpy(commonSettings.viewToClipMatrixPrev, commonSettings.viewToClipMatrix, matSize);
    memcpy(commonSettings.viewToClipMatrix, proj.get_value(), matSize);
    
    memcpy(commonSettings.worldToViewMatrixPrev, commonSettings.worldToViewMatrix, matSize);
    memcpy(commonSettings.worldToViewMatrix, view.get_value(), matSize);

    commonSettings.motionVectorScale[0] = 0.0f;
    commonSettings.motionVectorScale[1] = 0.0f;
    commonSettings.motionVectorScale[2] = 0.0f;
    commonSettings.isMotionVectorInWorldSpace = true;

    //commonSettings.frameIndex = m_pcRay.frame;
}

void HelloVulkan::populateReblurSettings(nrd::ReblurSettings& reblurSettings)
{
    reblurSettings.enableReferenceAccumulation = true;  // TODO: remove after it works, disables spatial filtering
    // reblurSettings.enablePerformanceMode = true;
}

void HelloVulkan::resetFrame()
{
    m_pcRay.frame = -1;
}

void HelloVulkan::updateFrame()
{
    static nvmath::mat4f refCamMatrix;
    static float         refFov{ CameraManip.getFov() };

    const auto& m = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

    if (memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || refFov != fov)
    {
        resetFrame();
        refCamMatrix = m;
        refFov = fov;
    }
    m_pcRay.frame++;
}
