import numpy as np

def grid(xc,yc,Lx,Ly,Nx,Ny):
    x = xc+np.linspace(-Lx/2,Lx/2,num=Nx)
    y = yc-np.linspace(-Ly/2,Ly/2,num=Ny)
    x,y = np.meshgrid(x,y)
    
    return x,y

def gridRows(xc,yc,Lx,Ly,Nx,Ny):
    x,y = grid(xc,yc,Lx,Ly,Nx,Ny)
    x = np.squeeze(x[0,:])
    y = np.squeeze(y[:,0])
    
    return x,y