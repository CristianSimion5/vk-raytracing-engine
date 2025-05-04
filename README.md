# Hardware-Accelerated Ray Tracing
A renderer that simulates natural lighting through both path tracing and hybrid ray tracing, combining rasterization with ray traced effects. It is capable of rendering and denoising ray traced scenes in real time.

Implemented in C++ using the cross-platform graphics API Vulkan, which provides access to modern hardware acceleration solutions such as Ray Tracing cores, which are necessary for achieving real-time performance when ray tracing. Simulated effects include global illumination, soft shadows, ambient occlusion and reflections.

## Features
- Cross Platform Renderer
- Switch between Path Tracing (ground truth image) and Rasterization + Ray Tracing (Hybrid Ray Tracing) Modes 
- Change RT functionality: features can be turned on and off, modify number of light bounces, color accumulation across frames
- Load GLTF Scenes/Models (only one provided in this repository)
- PBR lighting model

Video demo available [here](https://www.youtube.com/watch?v=GOE2hB0tYWQ)

## Configuration
Some settings can be configured using the `config.json` file:
```json
{
    "scenes" : [
        "media/scenes/Sponza.gltf",
        "media/scenes/fireplace/fireplace.gltf",
        "media/scenes/cornell.gltf",
        "media/scenes/suntemple/suntemple.gltf"
    ],
    "scene": 2,
    "vsync": false,
    "width": 1280,
    "height": 720
}
```

## Dependencies
- [nvpro-core](https://github.com/nvpro-samples/nvpro_core): Shared source code used for various [NVIDIA Samples](https://github.com/nvpro-samples). Used in this project as a thin framework which provides wrappers and helpers for various APIs (including Vulkan and other graphics APIs) to reduce verbosity. Also contains window management and UI functionality. nvpro-core uses the following projects:
    - GLFW: cross-platform windowing
    - GLM: mathematics library
    - Dear ImGUI: User Interface
- [Nvidia Real-Time Denoisers (NRD)](https://github.com/NVIDIA-RTX/NRD): Optimized denoising solution for ray tracing.

## Project Structure
    base_folder 
    |   nvpro_core
    |   vk-raytracing-engine


## Cloning and Building Steps
    git clone --recursive --shallow-submodules https://github.com/nvpro-samples/nvpro_core.git
    git clone https://github.com/CristianSimion5/vk-raytracing-engine.git

Follow the steps in [NRD: How to build?](https://github.com/NVIDIA-RTX/NRD?tab=readme-ov-file#how-to-build) to generate `_NRD_SDK` and `_NRI_SDK` and add them to `vk-raytracing-engine` (will be simplified in a future update)
### Build using CMake
    cd vk-raytracing-engine
    mkdir build
    cd build
    cmake ..

## Requirements
A GPU that supports at least Vulkan 1.2 and ray tracing extensions (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, `VK_KHR_deferred_host_operations`). This includes:
- Nvidia: RTX 20 series and above (GTX 10 series are also supported but ray tracing calculations are not hardware accelerated, they are instead performed using a CUDA variant)
- AMD: Radeon RX 6000 and above
- Intel: Arc series