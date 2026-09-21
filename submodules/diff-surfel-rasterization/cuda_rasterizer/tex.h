#ifndef _TEX_H_
#define _TEX_H_

#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>



inline __device__ glm::vec3 computeTexFromSH(int idx, int deg, int max_coeffs, const glm::vec3 pos, glm::vec3 campos, const float* shs, const float3 tex_DC)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = {SH_C0 * tex_DC.x, SH_C0 * tex_DC.y, SH_C0 * tex_DC.z};

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	return glm::max(result, 0.0f);
}


inline __device__ float3 computeTexFromSH_backward(int idx, int deg, int max_coeffs, const glm::vec3 pos, glm::vec3 campos, const float* shs, const float3 tex_DC, const float3 dL_dcolor, glm::vec3* dL_dshs)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = {SH_C0 * tex_DC.x, SH_C0 * tex_DC.y, SH_C0 * tex_DC.z};

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;
	

	if (deg > 0)
	{
		
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	bool clamped[3] = {result.x < 0, result.y < 0, result.z < 0};


	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	glm::vec3 dL_dRGB = {dL_dcolor.x, dL_dcolor.y, dL_dcolor.z};
	dL_dRGB.x *= clamped[0] ? 0 : 1;
	dL_dRGB.y *= clamped[1] ? 0 : 1;
	dL_dRGB.z *= clamped[2] ? 0 : 1;
	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);

	float dRGBdsh0 = SH_C0;
	float3 dL_drgbDC = {dRGBdsh0 * dL_dRGB.x, dRGBdsh0 * dL_dRGB.y, dRGBdsh0 * dL_dRGB.z};
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		// dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		// dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		// dL_dsh[3] = dRGBdsh3 * dL_dRGB;
		atomicAdd(&(dL_dshs[idx * max_coeffs + 1].x), dRGBdsh1 * dL_dRGB.x);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 1].y), dRGBdsh1 * dL_dRGB.y);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 1].z), dRGBdsh1 * dL_dRGB.z);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 2].x), dRGBdsh2 * dL_dRGB.x);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 2].y), dRGBdsh2 * dL_dRGB.y);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 2].z), dRGBdsh2 * dL_dRGB.z);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 3].x), dRGBdsh3 * dL_dRGB.x);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 3].y), dRGBdsh3 * dL_dRGB.y);
		atomicAdd(&(dL_dshs[idx * max_coeffs + 3].z), dRGBdsh3 * dL_dRGB.z);

		// dRGBdx = -SH_C1 * sh[3];
		// dRGBdy = -SH_C1 * sh[1];
		// dRGBdz = SH_C1 * sh[2];
		
		if (deg > 1)
		{
			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			// dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			// dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			// dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			// dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			// dL_dsh[8] = dRGBdsh8 * dL_dRGB;
			atomicAdd(&(dL_dshs[idx * max_coeffs + 4].x), dRGBdsh4 * dL_dRGB.x);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 4].y), dRGBdsh4 * dL_dRGB.y);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 4].z), dRGBdsh4 * dL_dRGB.z);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 5].x), dRGBdsh5 * dL_dRGB.x);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 5].y), dRGBdsh5 * dL_dRGB.y);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 5].z), dRGBdsh5 * dL_dRGB.z);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 6].x), dRGBdsh6 * dL_dRGB.x);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 6].y), dRGBdsh6 * dL_dRGB.y);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 6].z), dRGBdsh6 * dL_dRGB.z);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 7].x), dRGBdsh7 * dL_dRGB.x);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 7].y), dRGBdsh7 * dL_dRGB.y);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 7].z), dRGBdsh7 * dL_dRGB.z);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 8].x), dRGBdsh8 * dL_dRGB.x);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 8].y), dRGBdsh8 * dL_dRGB.y);
			atomicAdd(&(dL_dshs[idx * max_coeffs + 8].z), dRGBdsh8 * dL_dRGB.z);

			// dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			// dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			// dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				// dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				// dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				// dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				// dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				// dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				// dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				// dL_dsh[15] = dRGBdsh15 * dL_dRGB;
				atomicAdd(&(dL_dshs[idx * max_coeffs + 9].x), dRGBdsh9 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 9].y), dRGBdsh9 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 9].z), dRGBdsh9 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 10].x), dRGBdsh10 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 10].y), dRGBdsh10 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 10].z), dRGBdsh10 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 11].x), dRGBdsh11 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 11].y), dRGBdsh11 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 11].z), dRGBdsh11 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 12].x), dRGBdsh12 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 12].y), dRGBdsh12 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 12].z), dRGBdsh12 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 13].x), dRGBdsh13 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 13].y), dRGBdsh13 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 13].z), dRGBdsh13 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 14].x), dRGBdsh14 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 14].y), dRGBdsh14 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 14].z), dRGBdsh14 * dL_dRGB.z);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 15].x), dRGBdsh15 * dL_dRGB.x);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 15].y), dRGBdsh15 * dL_dRGB.y);
				atomicAdd(&(dL_dshs[idx * max_coeffs + 15].z), dRGBdsh15 * dL_dRGB.z);

				// dRGBdx += (
				// 	SH_C3[0] * sh[9] * 3.f * 2.f * xy +
				// 	SH_C3[1] * sh[10] * yz +
				// 	SH_C3[2] * sh[11] * -2.f * xy +
				// 	SH_C3[3] * sh[12] * -3.f * 2.f * xz +
				// 	SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
				// 	SH_C3[5] * sh[14] * 2.f * xz +
				// 	SH_C3[6] * sh[15] * 3.f * (xx - yy));
				
				// dRGBdy += (
				// 	SH_C3[0] * sh[9] * 3.f * (xx - yy) +
				// 	SH_C3[1] * sh[10] * xz +
				// 	SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
				// 	SH_C3[3] * sh[12] * -3.f * 2.f * yz +
				// 	SH_C3[4] * sh[13] * -2.f * xy +
				// 	SH_C3[5] * sh[14] * -2.f * yz +
				// 	SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				// dRGBdz += (
				// 	SH_C3[1] * sh[10] * xy +
				// 	SH_C3[2] * sh[11] * 4.f * 2.f * yz +
				// 	SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
				// 	SH_C3[4] * sh[13] * 4.f * 2.f * xz +
				// 	SH_C3[5] * sh[14] * (xx - yy));

			}


		}
	}
	// glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
	// float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	
	
	return dL_drgbDC;
}




#if BILINEAR

inline __device__ float3 tex2D(const float3* tex, int width, int height, float u, float v) {
	// u, v are in [-1, 1]
	u = (u + 1.0f) * 0.5f;
	v = (v + 1.0f) * 0.5f;
	// if (u < 0 || u > 1 || v < 0 || v > 1) printf("u, v: %f, %f, W, H: %d, %d\n", u, v, width, height); // TODO: may out bound
	const float x = u * (width - 1);
	const float y = v * (height - 1);
	const int x0 = min(max((int)x, 0), width - 1);
	const int y0 = min(max((int)y, 0), height - 1);
	const int x1 = min(max((int)x + 1, 0), width - 1);
	const int y1 = min(max((int)y + 1, 0), height - 1);

	if (x0 == x1 && y0 == y1) return tex[y0 * width + x0];
	else if (x0 == x1)
	{
		const float t = y - y0;
		const float3 c0 = tex[y0 * width + x0];
		const float3 c1 = tex[y1 * width + x0];
		return (1 - t) * c0 + t * c1;
	}
	else if (y0 == y1)
	{
		const float s = x - x0;
		const float3 c0 = tex[y0 * width + x0];
		const float3 c1 = tex[y0 * width + x1];
		return (1 - s) * c0 + s * c1;
	}
	else
	{
		const float s0 = x - x0;
		const float s1 = x1 - x;
		const float t0 = y - y0;
		const float t1 = y1 - y;
		const float3 c00 = tex[y0 * width + x0];
		const float3 c01 = tex[y0 * width + x1];
		const float3 c10 = tex[y1 * width + x0];
		const float3 c11 = tex[y1 * width + x1];

		return s1*t1*c00 + s0*t1*c01 + s1*t0*c10 + s0*t0*c11;
	}
	
	
	
}

#else

__device__ float3 tex2D(const float3* tex, int width, int height, float u, float v) {
	// u, v are in [-1, 1]
	u = (u + 1.0f) * 0.5f;
	v = (v + 1.0f) * 0.5f;
	// if (u < 0 || u > 1 || v < 0 || v > 1) printf("u, v: %f, %f, W, H: %d, %d\n", u, v, width, height); // TODO: may out bound
	int x = min(max(int(u * (width-1)), 0), width - 1);
	int y = min(max(int(v * (height-1)), 0), height - 1);
	return tex[y * width + x];
}

#endif



#ifdef BILINEAR

inline __device__ void tex2D_backward(float3* dL_dtex, int width, int height, float u, float v, float3 dL_drgb)
{
	u = (u + 1.0f) * 0.5f;
	v = (v + 1.0f) * 0.5f;
	// if (u < 0 || u > 1 || v < 0 || v > 1) printf("u, v: %f, %f, W, H: %d, %d\n", u, v, width, height); // TODO: may out bound
	const float x = u * (width - 1);
	const float y = v * (height - 1);
	const int x0 = min(max((int)x, 0), width - 1);
	const int y0 = min(max((int)y, 0), height - 1);
	const int x1 = min(max((int)x + 1, 0), width - 1);
	const int y1 = min(max((int)y + 1, 0), height - 1);
	
	if (x0 == x1 && y0 == y1)
	{
	 	atomicAdd(&(dL_dtex[y0 * width + x0].x), dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x0].y), dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x0].z), dL_drgb.z);
	}
	else if (x0 == x1)
	{
		const float t = y - y0;
		// const float3 c0 = tex[y0 * width + x0];
		// const float3 c1 = tex[y1 * width + x0];
		// return (1 - t) * c0 + t * c1;
		atomicAdd(&(dL_dtex[y0 * width + x0].x), (1 - t) * dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x0].y), (1 - t) * dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x0].z), (1 - t) * dL_drgb.z);
		atomicAdd(&(dL_dtex[y1 * width + x0].x), t * dL_drgb.x);
		atomicAdd(&(dL_dtex[y1 * width + x0].y), t * dL_drgb.y);
		atomicAdd(&(dL_dtex[y1 * width + x0].z), t * dL_drgb.z);

	}
	else if (y0 == y1)
	{
		const float s = x - x0;
		// const float3 c0 = tex[y0 * width + x0];
		// const float3 c1 = tex[y0 * width + x1];
		// return (1 - s) * c0 + s * c1;
		atomicAdd(&(dL_dtex[y0 * width + x0].x), (1 - s) * dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x0].y), (1 - s) * dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x0].z), (1 - s) * dL_drgb.z);
		atomicAdd(&(dL_dtex[y0 * width + x1].x), s * dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x1].y), s * dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x1].z), s * dL_drgb.z);

	}
	else
	{
		const float s0 = x - x0;
		const float s1 = x1 - x;
		const float t0 = y - y0;
		const float t1 = y1 - y;
		// const float3 c00 = tex[y0 * width + x0];
		// const float3 c01 = tex[y0 * width + x1];
		// const float3 c10 = tex[y1 * width + x0];
		// const float3 c11 = tex[y1 * width + x1];

		// return s1*t1*c00 + s0*t1*c01 + s1*t0*c10 + s0*t0*c11;
		atomicAdd(&(dL_dtex[y0 * width + x0].x), s1 * t1 * dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x0].y), s1 * t1 * dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x0].z), s1 * t1 * dL_drgb.z);
		atomicAdd(&(dL_dtex[y0 * width + x1].x), s0 * t1 * dL_drgb.x);
		atomicAdd(&(dL_dtex[y0 * width + x1].y), s0 * t1 * dL_drgb.y);
		atomicAdd(&(dL_dtex[y0 * width + x1].z), s0 * t1 * dL_drgb.z);
		atomicAdd(&(dL_dtex[y1 * width + x0].x), s1 * t0 * dL_drgb.x);
		atomicAdd(&(dL_dtex[y1 * width + x0].y), s1 * t0 * dL_drgb.y);
		atomicAdd(&(dL_dtex[y1 * width + x0].z), s1 * t0 * dL_drgb.z);
		atomicAdd(&(dL_dtex[y1 * width + x1].x), s0 * t0 * dL_drgb.x);
		atomicAdd(&(dL_dtex[y1 * width + x1].y), s0 * t0 * dL_drgb.y);
		atomicAdd(&(dL_dtex[y1 * width + x1].z), s0 * t0 * dL_drgb.z);
		
	}
	
	
}

#else
__device__ void tex2D_backward(float3* dL_dtex, int width, int height, float u, float v, float3 dL_drgb)
{
	// u, v are in [-1, 1]
	u = (u + 1.0f) * 0.5f;
	v = (v + 1.0f) * 0.5f;
	int x = min(max(int(u * (width-1)), 0), width - 1);
	int y = min(max(int(v * (height-1)), 0), height - 1);
	int idx = y * width + x;
	atomicAdd(&(dL_dtex[idx].x), dL_drgb.x);
	atomicAdd(&(dL_dtex[idx].y), dL_drgb.y);
	atomicAdd(&(dL_dtex[idx].z), dL_drgb.z);
}
#endif



#endif // _TEX_H_