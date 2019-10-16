"""
Demo visibility.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2

import soft_renderer as sr
import soft_renderer.functional as srf

from time import time

import soft_renderer.cuda.standard_rasterize as standard_rasterize_cuda


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 2], triangles[i, 1])
            f.write(s)

def get_visibility(vertices, triangles, image_size=256):
    '''
        vertices: [batch_size, nv, 3]. range:[-1, 1]
        triangles: [batch_size, nf, 3]
    '''
    bz = vertices.shape[0]
    h = w = image_size
    device = vertices.device
    # vertices[...,-1] = -vertices[...,-1]
    vertices = vertices*image_size/2. + image_size/2.

    depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
    triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
    baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
    vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
    
    st = time()
    face_vertices = srf.face_vertices(vertices, triangles)
    standard_rasterize_cuda.standard_rasterize(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    triangle_buffer = triangle_buffer.reshape(bz, -1)
    for i in range(bz):
        tri_visind = torch.unique(triangle_buffer[i])[1:].long()
        vert_visind = triangles[i,tri_visind,:].flatten()
        vert_vis[i, torch.unique(vert_visind.long())] = 1.0
    print(time() - st)
    return vert_vis

def get_visibility_z(vertices, triangles, image_size=256):
    '''
        vertices: [batch_size, nv, 3]. range:[-1, 1]
        triangles: [batch_size, nf, 3]
    '''
    bz = vertices.shape[0]
    h = w = image_size
    device = vertices.device
    # vertices[...,-1] = -vertices[...,-1]
    vertices = vertices*image_size/2. + image_size/2.

    depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
    triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
    baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
    vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
    
    st = time()
    face_vertices = srf.face_vertices(vertices, triangles)
    standard_rasterize_cuda.standard_rasterize(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    zrange = vertices[...,-1].max() - vertices[...,-1].min()
    for i in range(bz):
        for j in range(vertices.shape[1]):
            [x,y,z] = vertices[i, j]
            ul = depth_buffer[i, int(torch.floor(y)), int(torch.floor(x))]
            ur = depth_buffer[i, int(torch.floor(y)), int(torch.ceil(x))]
            dl = depth_buffer[i, int(torch.ceil(y)), int(torch.floor(x))]
            dr = depth_buffer[i, int(torch.ceil(y)), int(torch.ceil(x))]

            yd = y - torch.floor(y)
            xd = x - torch.floor(x)
            depth = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd
            if z < depth + zrange*0.02:
                vert_vis[i, j] = 1.0
    print(time() - st)
    return vert_vis

def main():
    # load obj
    mesh = sr.Mesh.from_obj('data/obj/body.obj', normalization=False, load_texture=False, texture_type='surface')
    vertices = mesh.vertices*0.8
    triangles = mesh.faces
    # vertices = vertices.expand(100,-1,-1)
    # triangles = triangles.expand(100,-1,-1)
    vert_vis = get_visibility(vertices, triangles, image_size=512)

    ## save obj
    vertices = vertices.detach().cpu().numpy()[0]
    triangles = triangles.detach().cpu().numpy()[0]
    vert_vis = vert_vis.detach().cpu().numpy()[0]
    colors = np.tile(vert_vis[:,None], (1,3))

    obj_name = 'data/obj/body_vis.obj'
    print('saving obj...')
    write_obj_with_colors(obj_name, vertices, triangles, colors)
    print('Done! Check obj in {}'.format(obj_name))
    



if __name__ == '__main__':
    main()