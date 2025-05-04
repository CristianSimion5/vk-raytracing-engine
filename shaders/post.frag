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
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 1) uniform sampler2D rtTxt;

layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
  int rtMode;
  int viewAccumulated;
  int useGI;
}
pushc;

void main()
{
  vec2  uv    = outUV;
  float gamma = 1. / 2.2;
  vec4 mainImg = texture(noisyTxt, uv);
  if (pushc.rtMode == 0)
  {
    vec4 rtImg = texture(rtTxt, uv);
    if (pushc.viewAccumulated == 0)
    {
        mainImg = vec4(mainImg.rgb * rtImg.a + rtImg.rgb, 1.0f);
    }
    else
    {
        if (pushc.useGI == 1)
            mainImg.rgb = rtImg.rgb * rtImg.a;
        else
            mainImg.rgb = vec3(rtImg.a);
        //mainImg.rgb = rtImg.rgb;
    }
  }
  // Gamma correct
  fragColor   = pow(mainImg, vec4(gamma));
  //fragColor   = texture(noisyTxt, uv).rgba;
}
