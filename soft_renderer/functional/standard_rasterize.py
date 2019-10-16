import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import soft_renderer.cuda.standard_rasterize as standard_rasterize_cuda


def standard_rasterize(face_vertices, image_size=256):
    if face_vertices.device == "cpu":
        raise TypeError('Rasterize module supports only cuda Tensors')
    
    # import ipdb; ipdb.set_trace()
    h = w = image_size
    depth_buffer = torch.zeros([face_vertices.shape[0], h, w]).float().cuda() + 1e6
    triangle_buffer = torch.zeros([face_vertices.shape[0], h, w]).int().cuda() - 1
    baryw_buffer = torch.zeros([face_vertices.shape[0], h, w, 3]).float().cuda()
    depth_buffer, triangle_buffer, baryw_buffer = standard_rasterize_cuda.standard_rasterize(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    return depth_buffer, triangle_buffer, baryw_buffer

def standard_rasterize_colors(face_vertices, face_colors, depth_buffer, triangle_buffer, images, h, w):
    if face_vertices.device == "cpu":
        raise TypeError('Rasterize module supports only cuda Tensors')
    
    # # import ipdb; ipdb.set_trace()
    # from time import time; st = time()

    # h = w = image_size
    # depth_buffer = torch.zeros([face_vertices.shape[0], h, w]).float().cuda() + 1e6
    # triangle_buffer = torch.zeros([face_vertices.shape[0], h, w]).int().cuda() - 1
    # images = torch.zeros([face_vertices.shape[0], h, w, 3]).float().cuda()
    # print(time() - st)

    # st = time()
    standard_rasterize_cuda.standard_rasterize_colors(face_vertices, face_colors, depth_buffer, triangle_buffer, images, h, w)
    # print(time() - st)
    # if return_buffers:
    #     return images, triangle_buffer, depth_buffer
    # else:
    #     return images
