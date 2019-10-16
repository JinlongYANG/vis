
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer.functional as srf


class SoftRasterizer(nn.Module):
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=1, far=100, 
                 anti_aliasing=False, fill_back=False, eps=1e-3, 
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface'):
        super(SoftRasterizer, self).__init__()

        if dist_func not in ['hard', 'euclidean', 'barycentric']:
            raise ValueError('Distance function only support hard, euclidean and barycentric')
        if aggr_func_rgb not in ['hard', 'softmax']:
            raise ValueError('Aggregate function(rgb) only support hard and softmax')
        if aggr_func_alpha not in ['hard', 'prod', 'sum']:
            raise ValueError('Aggregate function(a) only support hard, prod and sum')
        if texture_type not in ['surface', 'vertex']:
            raise ValueError('Texture type only support surface and vertex')

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.anti_aliasing = anti_aliasing
        self.eps = eps
        self.fill_back = fill_back
        self.sigma_val = sigma_val
        self.dist_func = dist_func
        self.dist_eps = dist_eps
        self.gamma_val = gamma_val
        self.aggr_func_rgb = aggr_func_rgb
        self.aggr_func_alpha = aggr_func_alpha
        self.texture_type = texture_type

    def forward(self, mesh, mode=None):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)
        images = srf.soft_rasterize(mesh.face_vertices, mesh.face_textures, image_size,
                                    self.background_color, self.near, self.far, 
                                    self.fill_back, self.eps,
                                    self.sigma_val, self.dist_func, self.dist_eps,
                                    self.gamma_val, self.aggr_func_rgb, self.aggr_func_alpha,
                                    self.texture_type)
        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images



class StandardRasterizer(nn.Module):
    def __init__(self,  batch_size = 1, image_size=256):
        super(StandardRasterizer, self).__init__()
        self.image_size = image_size
        h = w = image_size
        self.depth_buffer = torch.zeros([batch_size, h, w]).float().cuda() + 1e6
        self.triangle_buffer = torch.zeros([batch_size, h, w]).int().cuda() - 1
        self.images = torch.zeros([batch_size, h, w, 3]).float().cuda()
        self.h = h
        self.w = w

    def forward(self, face_vertices, face_colors, return_buffers):
        # depth_buffer, triangle_buffer, baryw_buffer = srf.standard_rasterize(face_vertices, image_size=self.image_size)
        # # render colors
        # bz, nf, device = face_vertices.shape[0], face_vertices.shape[1], face_vertices.device
        # triangle_buffer = triangle_buffer + (torch.arange(bz, dtype=torch.int32).to(device) * nf)[:, None, None]
        # face_colors = face_colors.reshape((bz * nf, 3, 3))
        # color_buffer = face_colors[triangle_buffer.long()] #[bz, h, w, 3, 3]

        # # 
        # weighted_color_buffer = (color_buffer*baryw_buffer[...,None]).sum(3)
        # images = weighted_color_buffer.permute(0, 3, 1, 2)
        # import ipdb; ipdb.set_trace()d
        self.depth_buffer[:] = 1e6
        self.triangle_buffer[:] = -1
        self.images[:] = 0.

        # import ipdb; ipdb.set_trace()
        srf.standard_rasterize_colors(face_vertices, face_colors, self.depth_buffer, self.triangle_buffer, self.images, self.h, self.w)

        if return_buffers:
            return self.images.permute(0,3,1,2).clone(), self.triangle_buffer.clone(), self.depth_buffer.clone()
        else:
            return self.images.permute(0,3,1,2).clone()