import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import soft_renderer.functional as srf


class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def forward(self, light):
        return srf.ambient_lighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0)):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def forward(self, light, normals):
        return srf.directional_lighting(light, normals,
                                        self.light_intensity, self.light_color, 
                                        self.light_direction)


class Lighting(nn.Module):
    def __init__(self, light_mode='surface',
                 intensity_ambient=0.5, color_ambient=[1,1,1],
                 intensity_directionals=0.5, color_directionals=[1,1,1],
                 directions=[0,1,0]):
        super(Lighting, self).__init__()

        # if light_mode not in ['surface', 'vertex']:
        #     raise ValueError('Lighting mode only support surface and vertex')

        self.light_mode = light_mode
        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions)])
        
    def forward(self, mesh):
        # if self.light_mode == 'surface':
        #     light = torch.zeros_like(mesh.faces, dtype=torch.float32).to(mesh.device)
        #     light = light.contiguous()
        #     light = self.ambient(light)
        #     for directional in self.directionals:
        #         light = directional(light, mesh.surface_normals)
        #     mesh.textures = mesh.textures * light[:, :, None, :]

        # elif self.light_mode == 'vertex':
        light = torch.zeros_like(mesh.vertices, dtype=torch.float32).to(mesh.device)#to('cuda')#cuda()
        light = light.contiguous()
        light = self.ambient(light)
        for directional in self.directionals:
            light = directional(light, mesh.vertex_normals)
        # print(mesh.textures.shape, light.shape)

        # 
        face_ver_lights = torch.stack([light[b,mesh.faces.detach().long()[b,...], :] for b in range(mesh.faces.shape[0])])
        # texture_lights = torch.sum(face_ver_lights[:,:,None,:,:]*tex_bary_weights[:,:,:,:,None], axis=3)
        # mesh.textures = mesh.textures * texture_lights

        return face_ver_lights 


class SHLighting(nn.Module):
    def __init__(self, nsh = 9):
        super(SHLighting, self).__init__()
        # pi = np.pi
        # att = pi*[1,2/3,.25]
        # if nsh==3:
        #     self.constant_factor = [att[0]*(1/sqrt(4*pi)), att[1]*(sqrt(3/(4*pi))), att[1]*(sqrt(3/(4*pi)))]
        # elif ns==9:
        #     self.constant_factor = [att[0]*(1/sqrt(4*pi)), att[1]*(sqrt(3/(4*pi))), att[1]*(sqrt(3/(4*pi))),\
        #                    att[1]*(sqrt(3/(4*pi))), att[2]*(1/2)*(sqrt(5/(4*pi))), att[2]*(3*sqrt(5/(12*pi))),\
        #                    att[2]*(3*sqrt(5/(12*pi))), att[2]*(3*sqrt(5/(48*pi))), att[2]*(3*sqrt(5/(12*pi)))]
        # self.constant_factor = []
        # self.ones = torch.ones([bz, nv])

    def forward(self, mesh, sh_coeff, tex_bary_weights=None):
        # SH_coeff [bz, 9(shcoeff), 3(rgb)]
        # SH lights:
        N = mesh.vertex_normals.clone() #[bz, nv, 3]
        sh = torch.stack([torch.ones([N.shape[0], N.shape[1]]).cuda(), N[...,0], N[...,1], \
        # sh = torch.stack([torch.sum(N**2, 2), N[...,0], N[...,1], \
                N[...,2], N[...,0]*N[...,1], N[...,0]*N[...,2], 
                N[...,1]*N[...,2], N[...,0]**2 - N[...,1]**2, 3*(N[...,2]**2) - 1], 2) # [bz, nv, 9]
        light = torch.sum(sh_coeff[:,None,:,:]*sh[:,:,:,None], 2) #[bz, nv, 3] 
        # 
        face_ver_lights = torch.stack([light[b,mesh.faces.long()[b,...], :] for b in range(mesh.faces.shape[0])])
        # texture_lights = torch.sum(face_ver_lights[:,:,None,:,:]*tex_bary_weights[:,:,:,:,None], axis=3)
        # import ipdb; ipdb.set_trace()
        # mesh.textures = mesh.textures * texture_lights

        return face_ver_lights 