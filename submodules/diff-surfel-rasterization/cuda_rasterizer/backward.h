/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int deg, int M,
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const float* transMats,
		const float* shs,
		const float3* texture_buffer,
		const int3* texture_index,
		const float* orig_points,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const glm::vec3* cam_pos,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		float3* final_color,
		float3* final_normal,
		float* final_depth,
		const float* dL_dpixels,
		const float* dL_depths,
		float * dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		float3* dL_dtex,
		float* dL_dshs);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
