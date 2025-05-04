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


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"

#include <vulkan/vulkan.hpp>
#include <json.hpp>

#include <NRI.h>
#include <NRIDescs.h>
#include <Extensions/NRIWrapperVK.h>
//#include <Extensions/NRIWrapperD3D12.h>
#include <Extensions/NRIHelper.h>

#include <NRD.h>
#include <NRDIntegration.hpp>
#include "main.h"

//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  bool changed = false;
  changed |= ImGui::Checkbox("Limit Max Frames", &helloVk.m_stopAtMaxFrames);
  if (helloVk.m_stopAtMaxFrames)
      changed |= ImGui::SliderInt("Max Frames", &helloVk.m_maxFrames, 1, 100);
  changed |= ImGui::SliderInt("Bounces", &helloVk.m_pcRay.depth, 1, 30, "%d", ImGuiSliderFlags_Logarithmic);

  ImGui::Separator();

  if (helloVk.m_pcPost.rtMode)
  {
      changed |= ImGui::SliderInt("Samples per pixel", &helloVk.m_pcRay.samples, 1, 100, "%d", ImGuiSliderFlags_Logarithmic);
  }
  else
  {
      changed |= ImGui::Checkbox("Shadow Rays", reinterpret_cast<bool*>(&helloVk.m_pcRay.useShadows));
      changed |= ImGui::Checkbox("Ambient Occlusion", reinterpret_cast<bool*>(&helloVk.m_pcRay.useAO));
      changed |= ImGui::Checkbox("Global Illumination", reinterpret_cast<bool*>(&helloVk.m_pcRay.useGI));
      changed |= ImGui::Checkbox("View Ray Traced effects", reinterpret_cast<bool*>(&helloVk.m_pcPost.viewAccumulated));
  }

  // TODO: change to work correctly with light buffers
  //if(ImGui::CollapsingHeader("Light"))
  //{
    //changed |= ImGui::RadioButton("Point", &helloVk.m_pcRaster.lightType, 0);
    //ImGui::SameLine();
    //changed |= ImGui::RadioButton("Infinite", &helloVk.m_pcRaster.lightType, 1);

    //changed |= ImGui::SliderFloat3("Position", &helloVk.m_pcRaster.lightPosition.x, -20.f, 20.f);
    //changed |= ImGui::SliderFloat("Intensity", &helloVk.m_pcRaster.lightIntensity, 0.f, 150.f);
  //}
  ImGui::Separator();

  changed |= ImGuiH::CameraWidget();

  if(changed)
    helloVk.resetFrame();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//static int const SAMPLE_WIDTH  = 1280;
//static int const SAMPLE_HEIGHT = 720;


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Read configuration file
  std::string path;
  bool vsync;
  int SAMPLE_WIDTH;
  int SAMPLE_HEIGHT;
  {
      using json = nlohmann::json;
      std::ifstream f(nvh::findFile("config.json", defaultSearchPaths, true));
      json data = json::parse(f);
      int id = data["scene"];
      path = data["scenes"][id];
      vsync = data["vsync"];
      SAMPLE_WIDTH = data["width"];
      SAMPLE_HEIGHT = data["height"];
  }

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);


  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  //CameraManip.setLookat(nvmath::vec3f(4.0f, 4.0f, 4.0f), nvmath::vec3f(0, 1, 0), nvmath::vec3f(0, 1, 0));
  CameraManip.setLookat(nvmath::vec3f(0, 0, 15), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 3);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // Request the ray tracing extensions
  vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);
  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  HelloVulkan helloVk;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_UNDEFINED, vsync/* vsync true*/);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0

  // Creation of the example
  //helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths, true));
  //helloVk.loadModel(nvh::findFile("media/scenes/Medieval_building.obj", defaultSearchPaths, true));
  //helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths, true));
  //helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths, true));
  //helloVk.loadModel(nvh::findFile("media/scenes/sphere.obj", defaultSearchPaths, true),
                    //nvmath::scale_mat4(nvmath::vec3f(1.5f)) * nvmath::translation_mat4(nvmath::vec3f(0.0f, 1.0f, 0.0f)));
  //helloVk.loadScene(nvh::findFile("media/scenes/Sponza.gltf", defaultSearchPaths, true));
  helloVk.loadGltfScene(nvh::findFile(path, defaultSearchPaths, true));

  helloVk.createOffscreenRender();
  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline();
  helloVk.createUniformBuffer();
  // helloVk.createObjDescriptionBuffer();
  helloVk.updateDescriptorSet();

  helloVk.initRayTracing();
  helloVk.createBottomLevelASGltf();
  helloVk.createTopLevelAsGltf();
  helloVk.createRtDescriptorSet();
  helloVk.createRtPipeline();
  helloVk.createHybridRtPipeline();
  //helloVk.createRtShaderBindingTable();

  helloVk.createPostDescriptor();
  helloVk.createPostPipeline();
  helloVk.updatePostDescriptorSet();

  nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);


  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  // NRD INTEGRATION
  //////////////////////////////////////////////////////////
  // Wrap Vulkan Device
  NrdIntegration NRD = NrdIntegration(helloVk.getSwapChain().getImageCount());
  struct NriInterface
      : public nri::CoreInterface
      , public nri::HelperInterface
      , public nri::WrapperVKInterface
  {};
  NriInterface NRI;
  
  //std::vector<uint32_t> queueFamilyIndices;
  uint32_t a = vkctx.m_queueGCT.familyIndex;
  auto physDevices = vkctx.getPhysicalDevices();
  //std::is_trivially_move_constructible <
  nri::DeviceCreationVulkanDesc deviceDesc = {};
  deviceDesc.vkDevice = helloVk.getDevice();
  deviceDesc.vkInstance = helloVk.getInstance();
  deviceDesc.vkPhysicalDevices = reinterpret_cast<nri::NRIVkPhysicalDevice*>(physDevices.data());
  deviceDesc.deviceGroupSize = physDevices.size();
  deviceDesc.queueFamilyIndices = &a;
  deviceDesc.queueFamilyIndexNum = 1;
  deviceDesc.enableNRIValidation = true;

  nri::Device* nriDevice = nullptr;
  nri::Result nriResult = nri::CreateDeviceFromVkDevice(deviceDesc, nriDevice);

  nriResult = nri::GetInterface(*nriDevice, 
      NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
  nriResult = nri::GetInterface(*nriDevice,
      NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI);
  nriResult = nri::GetInterface(*nriDevice,
      NRI_INTERFACE(nri::WrapperVKInterface), (nri::WrapperVKInterface*)&NRI);
  
  //////////////////////////////////////////////////////////
  // Initialize NRD

  const nrd::Identifier identifier1 = 100;
  const nrd::DenoiserDesc denoiserDesc[] = {
    {identifier1, nrd::Denoiser::REBLUR_DIFFUSE, SAMPLE_WIDTH, SAMPLE_HEIGHT}
  };
  nrd::InstanceCreationDesc instanceCreationDesc = {};
  instanceCreationDesc.denoisers = denoiserDesc;
  instanceCreationDesc.denoisersNum = 1;

  bool result = NRD.Initialize(instanceCreationDesc, *nriDevice, NRI, NRI);
  
  //////////////////////////////////////////////////////////
  // Wrap native pointers

  // Wrap command buffers
  std::vector<nri::CommandBuffer*> nriCommandBuffers;
  for (auto& cmdBuf : helloVk.getCommandBuffers())
  {
      nri::CommandBufferVulkanDesc commandBufferDesc = {};
      commandBufferDesc.vkCommandBuffer = cmdBuf;
      
      nri::CommandBuffer* nriCommandBuffer = nullptr;
      NRI.CreateCommandBufferVK(*nriDevice, commandBufferDesc, nriCommandBuffer);
      nriCommandBuffers.push_back(nriCommandBuffer);
  }

  // Wrap required textures
  const int N = 5;
  nri::TextureTransitionBarrierDesc entryDescs[N] = {};
  nri::Format entryFormat[N] = {};
  
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[0];
  
      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)helloVk.m_inMV.texture.image;
      textureDesc.vkFormat = helloVk.m_inMV.ivInfo.format;
      textureDesc.vkImageAspectFlags = helloVk.m_inMV.ivInfo.subresourceRange.aspectMask;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
      textureDesc.sampleNum = 1;
      textureDesc.arraySize = 1;
      textureDesc.mipNum = 1;

      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);
        
      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[1];

      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)helloVk.m_inNormalRoughness.texture.image;
      textureDesc.vkFormat = helloVk.m_inNormalRoughness.ivInfo.format;
      textureDesc.vkImageAspectFlags = helloVk.m_inNormalRoughness.ivInfo.subresourceRange.aspectMask;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
      textureDesc.sampleNum = 1;
      textureDesc.arraySize = 1;
      textureDesc.mipNum = 1;

      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);
      
      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[2];

      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)helloVk.m_inViewZ.texture.image;
      textureDesc.vkFormat = helloVk.m_inViewZ.ivInfo.format;
      textureDesc.vkImageAspectFlags = helloVk.m_inViewZ.ivInfo.subresourceRange.aspectMask;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
      textureDesc.sampleNum = 1;
      textureDesc.arraySize = 1;
      textureDesc.mipNum = 1;

      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);

      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[3];

      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)helloVk.m_inDiffRadianceHitDist.texture.image;
      textureDesc.vkFormat = helloVk.m_inDiffRadianceHitDist.ivInfo.format;
      textureDesc.vkImageAspectFlags = helloVk.m_inDiffRadianceHitDist.ivInfo.subresourceRange.aspectMask;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
      textureDesc.sampleNum = 1;
      textureDesc.arraySize = 1;
      textureDesc.mipNum = 1;

      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);

      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[4];

      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)helloVk.m_outDiffRadianceHitDist.texture.image;
      textureDesc.vkFormat = helloVk.m_outDiffRadianceHitDist.ivInfo.format;
      textureDesc.vkImageAspectFlags = helloVk.m_outDiffRadianceHitDist.ivInfo.subresourceRange.aspectMask;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
      textureDesc.sampleNum = 1;
      textureDesc.arraySize = 1;
      textureDesc.mipNum = 1;

      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);

      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }

  /*for (uint32_t i = 0; i < N; i++)
  {
      nri::TextureTransitionBarrierDesc& entryDesc = entryDescs[i];
      const nvvk::Texture& myResource;
      nri::NRIVkImage(myResource.image);
      nri::TextureVulkanDesc textureDesc = {};
      textureDesc.vkImage = (nri::NRIVkImage)myResource.image;
      textureDesc.vkFormat = ;
      textureDesc.vkImageAspectFlags = ;
      textureDesc.vkImageType = VK_IMAGE_TYPE_2D;
 
      NRI.CreateTextureVK(*nriDevice, textureDesc, (nri::Texture*&)entryDesc.texture);

      // DX12 SRV = Vulkan sampled image  nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE
      // DX12 UAV = Vulkan storage image, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL
      entryDesc.nextAccess = nri::AccessBits::SHADER_RESOURCE_STORAGE;
      entryDesc.nextLayout = nri::TextureLayout::GENERAL;
  }*/

  //////////////////////////////////////////////////////////
  nrd::CommonSettings commonSettings = {};
  nrd::ReblurSettings reblurSettings = {};
  helloVk.populateReblurSettings(reblurSettings);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(helloVk.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show UI window.
    if(helloVk.showGui())
    {
      bool changed = false;
      ImGuiH::Panel::Begin();
      changed |= ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
      changed |= ImGui::Checkbox("Path Tracer mode", reinterpret_cast<bool*>(&helloVk.m_pcPost.rtMode));
      renderUI(helloVk);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGuiH::Control::Info("", "", "(F11) Toggle Pane", ImGuiH::Control::Flags::Disabled);
      ImGuiH::Panel::End();

      if(changed)
        helloVk.resetFrame();
    }

    // Start rendering the scene
    helloVk.prepareFrame();

    // Start command buffer of this frame
    auto                   curFrame = helloVk.getCurFrame();
    const VkCommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Updating camera buffer
    helloVk.updateUniformBuffer(cmdBuf);

    // Clearing screen
    std::array<VkClearValue, 9> clearValues{};
    clearValues[0].color        = { {clearColor[0], clearColor[1], clearColor[2], clearColor[3]} };
    // BUFFER: ADD HERE
    clearValues[1].color        = { 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues[2].color        = { 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues[3].color        = { 0.0f, 0.0f };
    // DENOISER: ADD HERE
    clearValues[4].color        = { 0.0f, 0.0f, 0.0f, 0.0f };
    clearValues[5].color        = { 0.0f, 0.0f, 0.0f, 0.0f };
    clearValues[6].color        = { 0.0f, 0.0f, 0.0f, 0.0f };
    clearValues[7].color        = { 0.0f, 0.0f, 0.0f, 0.0f };
    clearValues[8].depthStencil = { 1.0f, 0};

    // Offscreen render pass
    {
      VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      offscreenRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
      offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
      offscreenRenderPassBeginInfo.renderPass      = helloVk.m_offscreenRenderPass;
      offscreenRenderPassBeginInfo.framebuffer     = helloVk.m_offscreenFramebuffer;
      offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      helloVk.updateFrame();

      if(helloVk.m_pcPost.rtMode == 1)
      {
        helloVk.pathtrace(cmdBuf, clearColor);
      }
      else
      {
        // Rendering Scene
        vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        helloVk.rasterizeGltf(cmdBuf);
        vkCmdEndRenderPass(cmdBuf);

        std::vector<VkImageMemoryBarrier> imageMemoryBarriers;
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // Change if using depth/stencil buffer
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // Previous access mask
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // New access mask

        barrier.image = helloVk.m_positionTexture.image;
        imageMemoryBarriers.emplace_back(barrier);
        barrier.image = helloVk.m_offscreenColor.image;
        imageMemoryBarriers.emplace_back(barrier);

        vkCmdPipelineBarrier(
            cmdBuf,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Source pipeline stage
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // Destination pipeline stage
            VK_DEPENDENCY_DEVICE_GROUP_BIT,
            0, nullptr,
            0, nullptr,
            static_cast<uint32_t>(imageMemoryBarriers.size()), imageMemoryBarriers.data()
        );

        helloVk.raytraceRasterizedScene(cmdBuf);

        barrier.image = helloVk.m_accumulatedTexture.image;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // Previous access mask
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // New access mask
        vkCmdPipelineBarrier(
            cmdBuf,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // Source pipeline stage
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // Destination pipeline stage
            VK_DEPENDENCY_DEVICE_GROUP_BIT,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
      }
    }

    /////////////////////////////////////////////////////
    // Render - Denoise
    /*helloVk.populateCommonSettings(commonSettings);
    NRD.SetCommonSettings(commonSettings);

    NRD.SetDenoiserSettings(identifier1, &reblurSettings);

    // INPUTS - IN_MV, IN_NORMAL_ROUGHNESS, IN_VIEWZ, IN_DIFF_RADIANCE_HITDIST,
    // OPTIONAL INPUTS - IN_DIFF_CONFIDENCE
    // OUTPUTS - OUT_DIFF_RADIANCE_HITDIST
    NrdUserPool userPool = {};
    {
        NrdIntegrationTexture tex{};
        tex.format = helloVk.m_inMV.format;
        tex.subresourceStates = &entryDescs[0];
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_MV, tex);

        tex.format = helloVk.m_inNormalRoughness.format;
        tex.subresourceStates = &entryDescs[1];
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_NORMAL_ROUGHNESS, tex);

        tex.format = helloVk.m_inViewZ.format;
        tex.subresourceStates = &entryDescs[2];
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_VIEWZ, tex);
        
        tex.format = helloVk.m_inDiffRadianceHitDist.format;
        tex.subresourceStates = &entryDescs[3];
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, tex);
        
        tex.format = helloVk.m_outDiffRadianceHitDist.format;
        tex.subresourceStates = &entryDescs[4];
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, tex);
    }

    bool enableDescriptorCaching = true;
    const nrd::Identifier denoisers[] = { identifier1 };
    // Let's goo RIP
    NRD.Denoise(denoisers, 1, *(nriCommandBuffers[curFrame]), userPool, enableDescriptorCaching);
    */
    /////////////////////////////////////////////////////

    std::array<VkClearValue, 2> clearValues2{};
    clearValues2[0].color = { {clearColor[0], clearColor[1], clearColor[2], clearColor[3]} };
    clearValues2[1].depthStencil = { 1.0f, 0 };
    
    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues2.size());
      postRenderPassBeginInfo.pClearValues    = clearValues2.data();
      postRenderPassBeginInfo.renderPass      = helloVk.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = helloVk.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering tonemapper
      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      helloVk.drawPost(cmdBuf);
      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    helloVk.submitFrame();
  }

  // Cleanup
  vkDeviceWaitIdle(helloVk.getDevice());

  for (auto& nriCmdBuf : nriCommandBuffers)
  {
      NRI.DestroyCommandBuffer(*nriCmdBuf);
  }
  // Release wrapped device
  //NRI.DestroyDevice(*nriDevice);
  // Also NRD needs to be recreated on "resize"
  NRD.Destroy();

  helloVk.destroyResources();
  helloVk.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
