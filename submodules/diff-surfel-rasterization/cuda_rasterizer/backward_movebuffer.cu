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

#include "backward.h"
#include "auxiliary.h"
#include "tex.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int deg, int M,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ shs,
	const float3* __restrict__ texture_buffer,
	const int3* __restrict__ texture_index,
	const float* __restrict__ orig_points,
	const glm::vec2* __restrict__ scales,
	const glm::vec4* __restrict__ rotations,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float3* __restrict__ final_colors,
	const float3* __restrict__ final_normals,
	const float* __restrict__ final_depths,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_dtex,
	float* dL_dshs)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	// float T = T_final;
	float T = 1.0f;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	// uint32_t contributor = toDo;
	uint32_t contributor = 0;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

	float acc_colors[C] = { 0 };
	float acc_depth = 0.0f;
	float acc_alpha = 0.0f;
	float acc_normal[3] = { 0 };

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	// const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;

	float final_color[C] = {final_colors[pix_id].x, final_colors[pix_id].y, final_colors[pix_id].z};
	float final_normal[3] = {final_normals[pix_id].x, final_normals[pix_id].y, final_normals[pix_id].z};
	float final_depth = final_depths[pix_id];

	int XX = 367;
	int YY = 230;

	// if (pix.x == XX && pix.y == YY) {
	// 	printf("back\n\n");
	// // 	printf("final_color: %f, %f, %f\n", final_color[0], final_color[1], final_color[2]);
	// // 	printf("final_normal: %f, %f, %f\n", final_normal[0], final_normal[1], final_normal[2]);
	// // 	printf("final_depth: %f\n", final_depth);
	// // 	printf("range: %d, %d\n", range.x, range.y);
	// // 	printf("pointid: %d\n", point_list[621209]);
	// }

	
	__shared__ float sorted_depth[SORT_WINDOW*BLOCK_SIZE];
	__shared__ int sorted_id[SORT_WINDOW*BLOCK_SIZE];
	int sorted_num = 0;
	
	
	if (inside)
	{
		for (int kid = 0; kid < SORT_WINDOW; kid++)
		{
			sorted_depth[kid+ block.thread_rank()*SORT_WINDOW] = FLT_MAX;
			sorted_id[kid+ block.thread_rank()*SORT_WINDOW] = -1;
		}
	}
		


#endif

	auto blend_one = [&]() {

		if (sorted_num == 0)
			return;
		--sorted_num;

		
		const float c_d = sorted_depth[block.thread_rank()*SORT_WINDOW];
		const int global_id = sorted_id[block.thread_rank()*SORT_WINDOW];

		const float2 xy = points_xy_image[global_id];
		const float3 Tu = {transMats[9 * global_id+0], transMats[9 * global_id+1], transMats[9 * global_id+2]};
		const float3 Tv = {transMats[9 * global_id+3], transMats[9 * global_id+4], transMats[9 * global_id+5]};
		const float3 Tw = {transMats[9 * global_id+6], transMats[9 * global_id+7], transMats[9 * global_id+8]};
		const float4 nor_o = normal_opacity[global_id];
		const int3 index = texture_index[global_id];

		float3 k = pix.x * Tw - Tu;
		float3 l = pix.y * Tw - Tv;
		float3 p = cross(k, l);
		
		float2 s = {p.x / p.z, p.y / p.z};
		float rho3d = (s.x * s.x + s.y * s.y); 
		float2 d = {xy.x - pixf.x, xy.y - pixf.y};
		float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 


		glm::mat3 R = quat_to_rotmat(rotations[global_id]);
		glm::mat3 S = scale_to_mat(scales[global_id], 1.0f);
		glm::mat3 L = R * S;

		float3 p_orig = ((float3*)orig_points)[global_id];


		glm::mat3x3 splat2world = glm::mat3x3(
			L[0], 
			L[1],
			glm::vec3(p_orig.x, p_orig.y, p_orig.z)
		);

		glm::vec3 UV1 = {s.x, s.y, 1.0f};

		// glm matrices are column-major
		// pos_world = splat2world * UV1
		glm::vec3 pos = splat2world * UV1;

		float3 k00 = (pixf.x - 0.5f) * Tw - Tu;
		float3 l00 = (pixf.y - 0.5f) * Tw - Tv;
		float3 p00 = cross(k00, l00);
		float2 s00 = {p00.x / p00.z, p00.y / p00.z};
		float rho00 = s00.x * s00.x + s00.y * s00.y;


		float3 k10 = (pixf.x + 0.5f) * Tw - Tu;
		float3 l10 = (pixf.y - 0.5f) * Tw - Tv;
		float3 p10 = cross(k10, l10);
		float2 s10 = {p10.x / p10.z, p10.y / p10.z};
		float rho10 = s10.x * s10.x + s10.y * s10.y;


		float3 k01 = (pixf.x - 0.5f) * Tw - Tu;
		float3 l01 = (pixf.y + 0.5f) * Tw - Tv;
		float3 p01 = cross(k01, l01);
		float2 s01 = {p01.x / p01.z, p01.y / p01.z};
		float rho01 = s01.x * s01.x + s01.y * s01.y;



		float3 k11 = (pixf.x + 0.5f) * Tw - Tu;
		float3 l11 = (pixf.y + 0.5f) * Tw - Tv;
		float3 p11 = cross(k11, l11);
		float2 s11 = {p11.x / p11.z, p11.y / p11.z};
		float rho11 = s11.x * s11.x + s11.y * s11.y;
		


		float G00 = exp(-0.5f * rho00);
		float G10 = exp(-0.5f * rho10);
		float G01 = exp(-0.5f * rho01);
		float G11 = exp(-0.5f * rho11);
		float G_c = exp(-0.5f * rho3d);
		// Eq. (2) from 3D Gaussian splatting paper.
		// Obtain alpha by multiplying with Gaussian opacity
		// and its exponential falloff from mean.
		// Avoid numerical instabilities (see paper appendix). 

		const float G = (G00 + G10 + G01 + G11 + G_c) / 5.0f;

		// const float G = exp(power);
		const float alpha = min(0.99f, nor_o.w * G);

		// if (pix.x == XX && pix.y == YY)
		// {
		// 	printf("id: %d, alpha: %f, depth: %f, sorted: %d\n", global_id, alpha, c_d, sorted_num);
		// }


		float test_T = T * (1 - alpha);
		if (test_T < 0.0001f)
		{
			done = true;
			return;
		}
		
		const float dchannel_dcolor = alpha * T;
		const float w = alpha * T;
		// Propagate gradients to per-Gaussian colors and keep
		// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
		// pair).
		float dL_dalpha = 0.0f;
		const float3 color_DC = tex2D(texture_buffer + index.x, index.y, index.z, s.x/TexRange, s.y/TexRange);
		float3 dL_drgb = {0, 0, 0};
		for (int ch = 0; ch < C; ch++)
		{
			const float c = ((float*)&color_DC)[ch];
			acc_colors[ch] += c * alpha * T;
			float accum_rec_ch = (final_color[ch] - acc_colors[ch]) / test_T;
			// Update last color (to be used in the next iteration)
			// accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
			// last_color[ch] = c;

			const float dL_dchannel = dL_dpixel[ch];
			// dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
			dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
			// Update the gradients w.r.t. color of the Gaussian. 
			// Atomic, since this pixel is just one of potentially
			// many that were affected by this Gaussian.
			// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			((float*)&dL_drgb)[ch] = dchannel_dcolor * dL_dchannel;
		}

		const float3 dL_dcolorDC = computeTexFromSH_backward(global_id, deg, M, pos, *cam_pos, shs, color_DC, dL_drgb, (glm::vec3*)dL_dshs);

		tex2D_backward(dL_dtex+index.x, index.y, index.z, s.x/TexRange, s.y/TexRange, dL_dcolorDC);
		

		float dL_dz = 0.0f;
		float dL_dweight = 0;
#if RENDER_AXUTILITY
		const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
		const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
		if (contributor == median_contributor) {
			dL_dz += dL_dmedian_depth;
			// dL_dweight += dL_dmax_dweight;
		}
#if DETACH_WEIGHT 
		// if not detached weight, sometimes 
		// it will bia toward creating extragated 2D Gaussians near front
		dL_dweight += 0;
#else
		dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
		dL_dalpha += dL_dweight - last_dL_dT;
		// propagate the current weight W_{i} to next weight W_{i-1}
		last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
		const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
		dL_dz += dL_dmd * dmd_dd;

		acc_depth += c_d * w;

		// Propagate gradients w.r.t ray-splat depths
		// accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
		accum_depth_rec = (final_depth - acc_depth) / test_T;
		
		dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;

		// Propagate gradients w.r.t. color ray-splat alphas
		acc_alpha += w;
		// accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
		accum_alpha_rec = (final_A - acc_alpha) / test_T;
		dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

		
		float normal[3] = { nor_o.x, nor_o.y, nor_o.z };
		// Propagate gradients to per-Gaussian normals
		for (int ch = 0; ch < 3; ch++) {
			acc_normal[ch] += normal[ch] * w;
			// accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
			accum_normal_rec[ch] = (final_normal[ch] - acc_normal[ch]) / test_T;
			last_normal[ch] = normal[ch];
			dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
			atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), w * dL_dnormal2D[ch]);
		}
		
#endif

		dL_dalpha *= T;
		// Update last alpha (to be used in the next iteration)
		

		// Account for fact that alpha also influences how much of
		// the background color is added if nothing left to blend
		float bg_dot_dpixel = 0;
		for (int i = 0; i < C; i++)
			bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
		dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


		// Helpful reusable temporary variables
		const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
		dL_dz += alpha * T * dL_ddepth; 
#endif
		
		float3 dL_dTu = {0, 0, 0};
		float3 dL_dTv = {0, 0, 0};
		float3 dL_dTw = {0, 0, 0};
		float2 dL_ds = {0, 0};
		float dsx_pz = 0;
		float dsy_pz = 0;
		float3 dL_dp = {0, 0, 0};
		float3 dL_dk = {0, 0, 0};
		float3 dL_dl = {0, 0, 0};



		if (rho3d <= rho2d)
		{
			dL_ds = {
				0.2f * dL_dG * -G_c * s.x + dL_dz * Tw.x,
				0.2f * dL_dG * -G_c * s.y + dL_dz * Tw.y
			};
			const float3 dz_dTw = {s.x, s.y, 1.0};
			dsx_pz = dL_ds.x / p.z;
			dsy_pz = dL_ds.y / p.z;
			dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
			dL_dk = cross(l, dL_dp);
			dL_dl = cross(dL_dp, k);

			dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
			dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
			dL_dTw = {
				pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
				pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
				pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};
		}
		else
		{
			dL_ds = {
				0.2f * dL_dG * -G_c * s.x,
				0.2f * dL_dG * -G_c * s.y,
			};
			dsx_pz = dL_ds.x / p.z;
			dsy_pz = dL_ds.y / p.z;
			dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
			dL_dk = cross(l, dL_dp);
			dL_dl = cross(dL_dp, k);

			dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
			dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
			dL_dTw = {
				pixf.x * dL_dk.x + pixf.y * dL_dl.x, 
				pixf.x * dL_dk.y + pixf.y * dL_dl.y, 
				pixf.x * dL_dk.z + pixf.y * dL_dl.z};
		
			dL_dTw.z += dL_dz;
		}
		
		

		// G00
		dL_ds = {
			0.2f * dL_dG * -G00 * s00.x,
			0.2f * dL_dG * -G00 * s00.y
		};
		dsx_pz = dL_ds.x / p00.z;
		dsy_pz = dL_ds.y / p00.z;
		dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s00.x + dsy_pz * s00.y)};
		dL_dk = cross(l00, dL_dp);
		dL_dl = cross(dL_dp, k00);

		const float3 dL_dTu_00 = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
		const float3 dL_dTv_00 = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
		const float3 dL_dTw_00 = {
			(pixf.x - 0.5f) * dL_dk.x + (pixf.y - 0.5f) * dL_dl.x, 
			(pixf.x - 0.5f) * dL_dk.y + (pixf.y - 0.5f) * dL_dl.y, 
			(pixf.x - 0.5f) * dL_dk.z + (pixf.y - 0.5f) * dL_dl.z};
		
		// G10
		dL_ds = {
			0.2f * dL_dG * -G10 * s10.x,
			0.2f * dL_dG * -G10 * s10.y
		};
		dsx_pz = dL_ds.x / p10.z;
		dsy_pz = dL_ds.y / p10.z;
		dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s10.x + dsy_pz * s10.y)};
		dL_dk = cross(l10, dL_dp);
		dL_dl = cross(dL_dp, k10);

		const float3 dL_dTu_10 = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
		const float3 dL_dTv_10 = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
		const float3 dL_dTw_10 = {
			(pixf.x + 0.5f) * dL_dk.x + (pixf.y - 0.5f) * dL_dl.x, 
			(pixf.x + 0.5f) * dL_dk.y + (pixf.y - 0.5f) * dL_dl.y, 
			(pixf.x + 0.5f) * dL_dk.z + (pixf.y - 0.5f) * dL_dl.z};

		// G01
		dL_ds = {
			0.2f * dL_dG * -G01 * s01.x,
			0.2f * dL_dG * -G01 * s01.y
		};
		dsx_pz = dL_ds.x / p01.z;
		dsy_pz = dL_ds.y / p01.z;
		dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s01.x + dsy_pz * s01.y)};
		dL_dk = cross(l01, dL_dp);
		dL_dl = cross(dL_dp, k01);
		
		const float3 dL_dTu_01 = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
		const float3 dL_dTv_01 = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
		const float3 dL_dTw_01 = {
			(pixf.x - 0.5f) * dL_dk.x + (pixf.y + 0.5f) * dL_dl.x, 
			(pixf.x - 0.5f) * dL_dk.y + (pixf.y + 0.5f) * dL_dl.y, 
			(pixf.x - 0.5f) * dL_dk.z + (pixf.y + 0.5f) * dL_dl.z};
		
		// G11
		dL_ds = {
			0.2f * dL_dG * -G11 * s11.x,
			0.2f * dL_dG * -G11 * s11.y
		};
		dsx_pz = dL_ds.x / p11.z;
		dsy_pz = dL_ds.y / p11.z;
		dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s11.x + dsy_pz * s11.y)};
		dL_dk = cross(l11, dL_dp);
		dL_dl = cross(dL_dp, k11);
		
		const float3 dL_dTu_11 = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
		const float3 dL_dTv_11 = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
		const float3 dL_dTw_11 = {
			(pixf.x + 0.5f) * dL_dk.x + (pixf.y + 0.5f) * dL_dl.x, 
			(pixf.x + 0.5f) * dL_dk.y + (pixf.y + 0.5f) * dL_dl.y, 
			(pixf.x + 0.5f) * dL_dk.z + (pixf.y + 0.5f) * dL_dl.z};
		

		dL_dTu.x += (dL_dTu_00.x + dL_dTu_10.x + dL_dTu_01.x + dL_dTu_11.x);
		dL_dTu.y += (dL_dTu_00.y + dL_dTu_10.y + dL_dTu_01.y + dL_dTu_11.y);
		dL_dTu.z += (dL_dTu_00.z + dL_dTu_10.z + dL_dTu_01.z + dL_dTu_11.z);
		dL_dTv.x += (dL_dTv_00.x + dL_dTv_10.x + dL_dTv_01.x + dL_dTv_11.x);
		dL_dTv.y += (dL_dTv_00.y + dL_dTv_10.y + dL_dTv_01.y + dL_dTv_11.y);
		dL_dTv.z += (dL_dTv_00.z + dL_dTv_10.z + dL_dTv_01.z + dL_dTv_11.z);
		dL_dTw.x += (dL_dTw_00.x + dL_dTw_10.x + dL_dTw_01.x + dL_dTw_11.x);
		dL_dTw.y += (dL_dTw_00.y + dL_dTw_10.y + dL_dTw_01.y + dL_dTw_11.y);
		dL_dTw.z += (dL_dTw_00.z + dL_dTw_10.z + dL_dTw_01.z + dL_dTw_11.z);


		atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
		atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
		atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
		atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
		atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
		atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
		atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
		atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
		atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);

		// Update gradients w.r.t. opacity of the Gaussian
		atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

		T = test_T;

		for(int kid=1; kid<SORT_WINDOW; kid++)
		{
			sorted_depth[kid+ block.thread_rank()*SORT_WINDOW-1] = sorted_depth[kid+ block.thread_rank()*SORT_WINDOW];
			sorted_id[kid+ block.thread_rank()*SORT_WINDOW-1] = sorted_id[kid+ block.thread_rank()*SORT_WINDOW];
		}
		sorted_depth[SORT_WINDOW-1+ block.thread_rank()*SORT_WINDOW] = FLT_MAX;

	};

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// const float ddelx_dx = 0.5f * W;
	// const float ddely_dy = 0.5f * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the FRONT
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			// if ((pix.x == 360&& pix.y == 224))
			// {
			// 	printf("rounds: %d\n", rounds);
			// 	printf("blocki: %d\n", i);
			// 	printf("coll_id: %d\n", coll_id);
			// 	printf("pix: %d, %d\n", pix.x, pix.y);
			// 	printf("XXYY: %d, %d\n", XX, YY);
			// }
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			// for (int i = 0; i < C; i++)
			// 	collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// if (pix.x == XX && pix.y == YY && (i == 2 || i == 3))
		// {	
		// 	printf("block i: %d\n", i);
		// 	for(int st = 0; st < BLOCK_SIZE; st++)
		// 	{
		// 		printf("id: %d\n", collected_id[st]);
		// 	}
		// }

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (sorted_num == SORT_WINDOW)
				blend_one();
				
			if (done)
				break;
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor ++;
			// if (contributor >= last_contributor)
			// 	continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (c_d < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			float3 k00 = (pixf.x - 0.5f) * Tw - Tu;
			float3 l00 = (pixf.y - 0.5f) * Tw - Tv;
			float3 p00 = cross(k00, l00);
			float2 s00 = {p00.x / p00.z, p00.y / p00.z};
			float rho00 = s00.x * s00.x + s00.y * s00.y;


			float3 k10 = (pixf.x + 0.5f) * Tw - Tu;
			float3 l10 = (pixf.y - 0.5f) * Tw - Tv;
			float3 p10 = cross(k10, l10);
			float2 s10 = {p10.x / p10.z, p10.y / p10.z};
			float rho10 = s10.x * s10.x + s10.y * s10.y;


			float3 k01 = (pixf.x - 0.5f) * Tw - Tu;
			float3 l01 = (pixf.y + 0.5f) * Tw - Tv;
			float3 p01 = cross(k01, l01);
			float2 s01 = {p01.x / p01.z, p01.y / p01.z};
			float rho01 = s01.x * s01.x + s01.y * s01.y;



			float3 k11 = (pixf.x + 0.5f) * Tw - Tu;
			float3 l11 = (pixf.y + 0.5f) * Tw - Tv;
			float3 p11 = cross(k11, l11);
			float2 s11 = {p11.x / p11.z, p11.y / p11.z};
			float rho11 = s11.x * s11.x + s11.y * s11.y;
			


			float G00 = exp(-0.5f * rho00);
			float G10 = exp(-0.5f * rho10);
			float G01 = exp(-0.5f * rho01);
			float G11 = exp(-0.5f * rho11);
			float G_c = exp(-0.5f * rho3d);
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 

			const float G = (G00 + G10 + G01 + G11 + G_c) / 5.0f;

			// const float G = exp(power);
			float alpha = min(0.99f, opa * G);

			// if (pix.x == XX && pix.y == YY) {
			// 	if (collected_id[j] == 51493 || collected_id[j]== 32086)
			// 		printf("id: %d, alpha: %f, G: %f, opa: %f, G_c: %f, j: %d, i:%d\n", collected_id[j], alpha, G, opa, G_c, j, i);
			// }

			if (alpha < 1.0f / 255.0f)
				continue;

			int id = collected_id[j];
			bool rhochoice = rho3d <= rho2d;
			for (int kid = 0; kid < SORT_WINDOW; kid++)
			{
				if (c_d < sorted_depth[kid+ block.thread_rank()*SORT_WINDOW])
				{
					swap(sorted_depth[kid+ block.thread_rank()*SORT_WINDOW], c_d);
					swap(sorted_id[kid+ block.thread_rank()*SORT_WINDOW], id);
				}
			}
			++sorted_num;
			

			// T = T / (1.f - alpha);
			
		}
	}
	if(!done)
	{
		while(sorted_num > 0)
		{
			blend_one();
		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	// if (shs)
	// 	computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5f * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5f * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);
}

void BACKWARD::render(
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
	float* dL_dshs)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H, deg, M,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		shs,
		texture_buffer,
		texture_index,
		orig_points,
		scales,
		rotations,
		cam_pos,
		depths,
		final_Ts,
		n_contrib,
		final_color,
		final_normal,
		final_depth,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dtex,
		dL_dshs
		);
}
