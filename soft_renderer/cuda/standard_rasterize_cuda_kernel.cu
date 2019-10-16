#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}

// static __inline__ __device__ double atomicMin(double* address, double val) {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                 __double_as_longlong(fminf(val,  __longlong_as_double(assumed))));
//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }

#endif


namespace{

__device__ float atomicMin(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_i, assumed,
                __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

template <typename scalar_t>
    __device__ __forceinline__ bool check_face_frontside(const scalar_t *face) {
        return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
    }

template <typename scalar_t>
__device__ __forceinline__ bool check_pixel_inside(const scalar_t *w) {
    return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
}

template <typename scalar_t>
__device__ __forceinline__ void barycentric_clip(scalar_t *w) {
    for (int k = 0; k < 3; k++) w[k] = max(min(w[k], 1.), 0.);
    const scalar_t w_sum = max(w[0] + w[1] + w[2], 1e-5);
    for (int k = 0; k < 3; k++) w[k] /= w_sum;
}

template <typename scalar_t>
__global__ void forward_rasterize_cuda_kernel(
        // const scalar_t* __restrict__ vertices, //[bz, nv, 3]
        const scalar_t* __restrict__ face_vertices, //[bz, nf, 3, 3]
        float*  depth_buffer,
        int*  triangle_buffer,
        float*  baryw_buffer,        
        int batch_size, int h, int w, 
        int ntri) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * ntri) {
        return;
    }
    // const int is = image_size;
    const scalar_t* face = &face_vertices[i * 9];
    scalar_t bw[3];
    // scalar_t depth_min = 10000000;
    /* return if backside */
    // if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        // return;
    /* p[num][xy]: x, y is (-1, 1). */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = face[3 * num + dim]; // no normalize
        }
    }

    /* compute face_inv */
    scalar_t face_inv_star[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_determinant = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
    /* set to global memory */
    scalar_t face_inv[9];
    for (int k = 0; k < 9; k++) {
        face_inv[k] = face_inv_star[k] / face_inv_determinant;
    }

    int x_min = max((int)ceil(min(p[0][0], min(p[1][0], p[2][0]))), 0);
    int x_max = min((int)floor(max(p[0][0], max(p[1][0], p[2][0]))), w - 1);
    
    int y_min = max((int)ceil(min(p[0][1], min(p[1][1], p[2][1]))), 0);
    int y_max = min((int)floor(max(p[0][1], max(p[1][1], p[2][1]))), h - 1);

    int bn = i/ntri;
    for(int y = y_min; y <= y_max; y++) //h
    {
        for(int x = x_min; x <= x_max; x++) //w
        {
            bw[0] = face_inv[3 * 0 + 0] * x + face_inv[3 * 0 + 1] * y + face_inv[3 * 0 + 2];
            bw[1] = face_inv[3 * 1 + 0] * x + face_inv[3 * 1 + 1] * y + face_inv[3 * 1 + 2];
            bw[2] = face_inv[3 * 2 + 0] * x + face_inv[3 * 2 + 1] * y + face_inv[3 * 2 + 2];
            
            barycentric_clip(bw);

            if(check_pixel_inside(bw))// && check_face_frontside(face))
            {
                // const 
                // barycentric_clip(bw);
                scalar_t zp = 1. / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);

                atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                if(depth_buffer[bn*h*w + y*w + x] == zp)
                {
                    // depth_min = zp;
                    // atomic long long for two int
                    // scalar_t tri_ind = i%ntri;
                    // atomicAdd( (int*)&depth_buffer[bn*h*w + y*w + x],  (int)zp);
                    // atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                    triangle_buffer[bn*h*w + y*w + x] = (int)(i%ntri);
                    for(int k=0; k<3; k++){
                        baryw_buffer[bn*h*w*3 + y*w*3 + x*3 + k] = bw[k];
                    }
                    // buffers[bn*h*w*2 + y*w*2 + x*2 + 1] = p_depth;
                }
            }
        }
    }

}
    

template <typename scalar_t>
__global__ void forward_rasterize_colors_cuda_kernel(
        // const scalar_t* __restrict__ vertices, //[bz, nv, 3]
        const scalar_t* __restrict__ face_vertices, //[bz, nf, 3, 3]
        const scalar_t* __restrict__ face_colors, //[bz, nf, 3, 3]
        float*  depth_buffer,
        int*  triangle_buffer,
        float*  images,        
        int batch_size, int h, int w, 
        int ntri) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * ntri) {
        return;
    }
    // const int is = image_size;
    const scalar_t* face = &face_vertices[i * 9];
    const scalar_t* color = &face_colors[i * 9];
    int bn = i/ntri;

    scalar_t bw[3];
    
    
    // scalar_t depth_min = 10000000;
    /* return if backside */
    // if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        // return;
    /* p[num][xy]: x, y is (-1, 1). */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = face[3 * num + dim]; // no normalize
        }
    }
    scalar_t cl[3][3];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 3; dim++) {
            cl[num][dim] = color[3 * num + dim]; //[3p,3rgb]
        }
    }
    /* compute face_inv */
    scalar_t face_inv_star[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_determinant = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
    /* set to global memory */
    scalar_t face_inv[9];
    for (int k = 0; k < 9; k++) {
        face_inv[k] = face_inv_star[k] / face_inv_determinant;
    }

    int x_min = max((int)ceil(min(p[0][0], min(p[1][0], p[2][0]))), 0);
    int x_max = min((int)floor(max(p[0][0], max(p[1][0], p[2][0]))), w - 1);
    
    int y_min = max((int)ceil(min(p[0][1], min(p[1][1], p[2][1]))), 0);
    int y_max = min((int)floor(max(p[0][1], max(p[1][1], p[2][1]))), h - 1);

    for(int y = y_min; y <= y_max; y++) //h
    {
        for(int x = x_min; x <= x_max; x++) //w
        {
            bw[0] = face_inv[3 * 0 + 0] * x + face_inv[3 * 0 + 1] * y + face_inv[3 * 0 + 2];
            bw[1] = face_inv[3 * 1 + 0] * x + face_inv[3 * 1 + 1] * y + face_inv[3 * 1 + 2];
            bw[2] = face_inv[3 * 2 + 0] * x + face_inv[3 * 2 + 1] * y + face_inv[3 * 2 + 2];
            

            if(check_pixel_inside(bw))// && check_face_frontside(face))
            {
                // const 
                barycentric_clip(bw);
                scalar_t zp = 1. / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);

                atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                if(depth_buffer[bn*h*w + y*w + x] == zp)
                {
                    // depth_min = zp;
                    // atomic long long for two int
                    // scalar_t tri_ind = i%ntri;
                    // atomicAdd( (int*)&depth_buffer[bn*h*w + y*w + x],  (int)zp);
                    // atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                    triangle_buffer[bn*h*w + y*w + x] = (int)(i%ntri);
                    for(int k=0; k<3; k++){
                        // baryw_buffer[bn*h*w*3 + y*w*3 + x*3 + k] = bw[k];
                        images[bn*h*w*3 + y*w*3 + x*3 + k] = bw[0]*cl[0][k] + bw[1]*cl[1][k] + bw[2]*cl[2][k];
                    }
                    // buffers[bn*h*w*2 + y*w*2 + x*2 + 1] = p_depth;
                }
            }
        }
    }

}
    
}

std::vector<at::Tensor> forward_rasterize_cuda(
    at::Tensor face_vertices,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor baryw_buffer,
    int h,
    int w){

    const auto batch_size = face_vertices.size(0);
    const auto ntri = face_vertices.size(1);

    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * ntri - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_cuda1", ([&] {
      forward_rasterize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        depth_buffer.data<float>(),
        triangle_buffer.data<int>(),
        baryw_buffer.data<float>(),
        batch_size, h, w,
        ntri);
      }));
    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_cuda2", ([&] {
        forward_rasterize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        depth_buffer.data<float>(),
        triangle_buffer.data<int>(),
        baryw_buffer.data<float>(),
        batch_size, h, w,
        ntri);
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_rasterize_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {depth_buffer, triangle_buffer, baryw_buffer};
}


std::vector<at::Tensor> forward_rasterize_colors_cuda(
    at::Tensor face_vertices,
    at::Tensor face_colors,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor images,
    int h,
    int w){

    const auto batch_size = face_vertices.size(0);
    const auto ntri = face_vertices.size(1);

    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * ntri - 1) / threads +1);
    //initial 

    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_colors_cuda", ([&] {
      forward_rasterize_colors_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        face_colors.data<scalar_t>(),
        depth_buffer.data<float>(),
        triangle_buffer.data<int>(),
        images.data<float>(),
        batch_size, h, w,
        ntri);
      }));
    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_colors_cuda", ([&] {
        forward_rasterize_colors_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
          face_vertices.data<scalar_t>(),
          face_colors.data<scalar_t>(),
          depth_buffer.data<float>(),
          triangle_buffer.data<int>(),
          images.data<float>(),
          batch_size, h, w,
          ntri);
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_rasterize_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {depth_buffer, triangle_buffer, images};
}



