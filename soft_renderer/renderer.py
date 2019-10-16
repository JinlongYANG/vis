
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import soft_renderer as sr


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.Rasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode=None):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def forward(self, vertices, faces, textures=None, mode=None, texture_type='surface'):
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode)


class SHSoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='vertex', #flat, Gouraud, Phong
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(SoftRenderer, self).__init__()

        # light
        self.light_mode = light_mode
        if light_mode == 'sh':
            self.lighting = sr.SHLighting()
        else:
            self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)
        # camera
        self.transform = sr.Transform(camera_mode, 
                                      K, R, t, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        # self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode=None):
        self.set_texture_mode(mesh.texture_type)
        # mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def forward(self, vertices, faces, textures=None, sh_coeff = None, mode=None, texture_type='surface', maskout = True, return_mask = False, is_lighting=False):
        tex_bary_weights = textures[...,3:].detach()
        textures = textures[...,:3]#.detach()

        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        
        if is_lighting:
            mesh = self.lighting(mesh, sh_coeff, tex_bary_weights)
        self.set_texture_mode(mesh.texture_type)
        mesh = self.transform(mesh)
        images = self.rasterizer(mesh, mode)

        # fix mouth issue: mask out inner mouth part
        # Observation: the rendered vertices in mouth part actually belong to back head, which have negative normal directions
        # Method: 1. calculate normal direction of each vertex. 2. set color of negtive normal to 0, others 1
        # transformed_vertices = torch.stack([mesh.vertices[b,mesh.faces.detach().long()[b,...], :] for b in range(faces.shape[0])])
        # v10 = transformed_vertices[:, :, 0] - transformed_vertices[:, :, 1]
        # v12 = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 1]

        # fnorm = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)        # 
        # fmaskcolor = fnorm[:,:,2:].gt(0).float()
        # mask_textures = fmaskcolor[:,:,None,:].expand(textures.shape)
        
        # mesh_mask = sr.Mesh(mesh.vertices.detach(), faces, textures=mask_textures.detach(), texture_type=texture_type)
        # masks = self.rasterizer(mesh_mask, mode).detach()

        
        # if maskout:
        #     images = images*masks
        
        if return_mask:
            return images, masks
        else:
            return images



class TexSoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, is_tranform=True,
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='vertex',
                 camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='direcntional', # directional, spherical 
                 shading_mode='Gouraud', #flat, Gouraud, Phong
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(TexSoftRenderer, self).__init__()

        # light
        self.light_mode = light_mode
        if light_mode == 'directional':
            self.lighting = sr.Lighting(light_mode,
                                        light_intensity_ambient, light_color_ambient,
                                        light_intensity_directionals, light_color_directionals,
                                        light_directions)
        elif light_mode == 'spherical':
            self.lighting = sr.SHLighting()
        else:
            print('no lighting')
        # camera
        if is_tranform:
            self.transform = sr.Transform(camera_mode, 
                                        K, R, t, dist_coeffs, orig_size,
                                        perspective, viewing_angle, viewing_scale, 
                                        eye, camera_direction)

        # rasterization
        self.texture_type = texture_type
        anti_aliasing = False
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)
        self.is_transform = is_tranform

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def forward(self, vertices, faces, textures=None, sh_coeff = None, maskout = False, return_mask = False, is_lighting=False, is_normal=False):
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=self.texture_type)
        # import ipdb; ipdb.set_trace()
        if is_lighting:
            if self.light_mode=='directional':
                lights = self.lighting(mesh)
            elif self.light_mode=='spherical':
                lights = self.lighting(mesh, sh_coeff)
            mesh.textures = torch.cat([mesh.textures, lights], -1)
        if self.is_transform:
            mesh = self.transform(mesh)
        if is_normal:
            transformed_vertices = torch.stack([mesh.vertices[b,mesh.faces.detach().long()[b,...], :] for b in range(faces.shape[0])])
            v10 = transformed_vertices[:, :, 0] - transformed_vertices[:, :, 1]
            v12 = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 1]
            fnorm = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)
            mesh.textures = torch.cat([mesh.textures, fnorm[:,:,None,:].expand(-1,-1,3,-1)], -1)

        images = self.rasterizer(mesh)
        # fnorm = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)        # 
        return images
        # else:
        #     images = images[:,[0,1,2,-1],:,:]
        
        # if is_lighting and (self.light_mode is not None):
        #     lights_image = images[:,3:6,:,:]
        #     alpha = images[:,-1,:,:]
        #     images = images[:,:3,:,:]*lights_image
        #     images = torch.cat([images, alpha[:,None,:,:]], 1)

        # fix mouth issue: mask out inner mouth part
        # Observation: the rendered vertices in mouth part actually belong to back head, which have negative normal directions
        # Method: 1. calculate normal direction of each vertex. 2. set color of negtive normal to 0, others 1
        # transformed_vertices = torch.stack([mesh.vertices[b,mesh.faces.detach().long()[b,...], :] for b in range(faces.shape[0])])
        # v10 = transformed_vertices[:, :, 0] - transformed_vertices[:, :, 1]
        # v12 = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 1]

        # fnorm = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)        # 
        # fmaskcolor = fnorm[:,:,2:].gt(0).float()
        # mask_textures = fmaskcolor[:,:,None,:].expand(textures.shape)
        
        # mesh_mask = sr.Mesh(mesh.vertices.detach(), faces, textures=mask_textures.detach(), texture_type=texture_type)
        # masks = self.rasterizer(mesh_mask, mode).detach()

        
        # if maskout:
        #     images = images*masks
        
        # if return_mask:
        #     return images, masks
        # else:
        #     return images


class ColorRenderer(nn.Module):
    def __init__(self, batch_size=1, image_size=256, is_tranform=True, camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1]):
        super(ColorRenderer, self).__init__()

        # camera
        if is_tranform:
            self.transform = sr.Transform(camera_mode, 
                                        K, R, t, dist_coeffs, orig_size,
                                        perspective, viewing_angle, viewing_scale, 
                                        eye, camera_direction)

        # rasterization
        self.rasterizer = sr.StandardRasterizer(batch_size, image_size)
        self.is_transform = is_tranform
        self.image_size = image_size
    

    def forward(self, vertices, faces, textures=None, return_buffers=False):
        mesh = sr.Mesh(vertices, faces, textures=textures)
        if self.is_transform:
            mesh = self.transform(mesh)
        face_vertices = mesh.face_vertices
        face_vertices[...,1] = -face_vertices[...,1]
        face_vertices = mesh.face_vertices*self.image_size/2 + self.image_size/2
        # face_colors = srf.face_vertices(faces, textures)
        # images = 
        # fnorm = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)        # 
        return self.rasterizer(face_vertices, textures, return_buffers)
