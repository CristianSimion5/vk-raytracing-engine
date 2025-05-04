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

#pragma once

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"

#include <NRI.h>
#include <NRIDescs.h>
#include <Extensions/NRIWrapperVK.h>
#include <Extensions/NRIHelper.h>

#include <NRD.h>
#include <NRDIntegration.h>

struct VkDenoiseResource
{
    nvvk::Texture texture;
    VkImageViewCreateInfo ivInfo;
    nrd::ResourceType resourceType;
    nri::Format format;
};

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvkhl::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadGltfMaterials(const VkCommandBuffer& cmdBuf, VkBufferUsageFlags flags);
  void loadGltfLights(const VkCommandBuffer& cmdBuf, VkBufferUsageFlags flags);
  void loadGltfScene(const std::string& filename);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterizeGltf(const VkCommandBuffer& cmdBuf);


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1},                // Identity matrix
      {1},                // Identity matrix
      0,                  // instance Id
      0                   // material id  
  };

  // Scene info and buffers
  nvh::GltfScene m_gltfScene;
  nvvk::Buffer   m_vertexBuffer;
  nvvk::Buffer   m_normalBuffer;
  nvvk::Buffer   m_tangentBuffer;
  nvvk::Buffer   m_uvBuffer;
  nvvk::Buffer   m_indexBuffer;
  nvvk::Buffer   m_materialBuffer;
  nvvk::Buffer   m_primInfo;
  nvvk::Buffer   m_sceneDesc;
  nvvk::Buffer   m_lightBuffer;

  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene

  // Ray tracing
  void initRayTracing();
  // auto objectToVkGeometryKHR(const ObjModel& model);
  auto primitiveToGeometry(const nvh::GltfPrimMesh& prim);
  // void createBottomLevelAS();
  void createBottomLevelASGltf();
  // void createTopLevelAs();
  void createTopLevelAsGltf();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void createHybridRtPipeline();
  // void createRtShaderBindingTable();
  void pathtrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

  void raytraceRasterizedScene(const VkCommandBuffer& cmdBuf);

  void populateCommonSettings(nrd::CommonSettings& commonSettings);
  void populateReblurSettings(nrd::ReblurSettings& reblurSettings);

  void resetFrame();
  void updateFrame();

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR  m_rtBuilder;

  nvvk::DescriptorSetBindings m_rtDescSetLayoutBind;
  VkDescriptorPool            m_rtDescPool;
  VkDescriptorSetLayout       m_rtDescSetLayout;
  VkDescriptorSet             m_rtDescSet;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;
  PushConstantRay                                   m_pcRay{};

  // Hybrid pipeline
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups2;
  VkPipelineLayout                                  m_rtPipelineLayout2;
  VkPipeline                                        m_rtPipeline2;

  //nvvk::Buffer m_rtSBTBuffer;
  //VkStridedDeviceAddressRegionKHR m_rgenRegion{};
  //VkStridedDeviceAddressRegionKHR m_missRegion{};
  //VkStridedDeviceAddressRegionKHR m_hitRegion{};
  //VkStridedDeviceAddressRegionKHR m_callRegion{};
  nvvk::SBTWrapper                m_sbtWrapper;
  nvvk::SBTWrapper                m_sbtWrapper2;

  bool m_stopAtMaxFrames{false};
  int  m_maxFrames{1};

  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  struct PushConstantPost
  {
      float aspectRatio;
      int rtMode;
      int viewAccumulated;
      int useGI;
  };

  PushConstantPost m_pcPost;

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};
  // GBuffer data
  nvvk::Texture               m_positionTexture;
  nvvk::Texture               m_normalTexture;
  // Ray-traced accumulated effects 
  nvvk::Texture               m_accumulatedTexture;
  nvvk::Texture               m_roughnessMap;

  //Denoising data
  // INPUTS - IN_MV, IN_NORMAL_ROUGHNESS, IN_VIEWZ, IN_DIFF_RADIANCE_HITDIST,
  // OPTIONAL INPUTS - IN_DIFF_CONFIDENCE
  // OUTPUTS - OUT_DIFF_RADIANCE_HITDIST
  VkDenoiseResource           m_inMV;
  VkDenoiseResource           m_inNormalRoughness;
  VkDenoiseResource           m_inViewZ;
  VkDenoiseResource           m_inDiffRadianceHitDist;
  VkDenoiseResource           m_outDiffRadianceHitDist;
};
