import sisl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import issparse
import scipy.sparse as sp
import scipy.sparse.linalg as scilin
from Block_matrices.Block_matrices import block_td, Build_BTD_vectorised, test_partition_2d_sparse_matrix, slices_to_npslices, Blocksparse2Numpy
from nanobtl.utils import *
from nanobtl.cap import *


def transCAP(E, K, H, WCAP_l, WCAP_r, bc_axis, eavg=False, kavg=False, eta=1e-3, indices_left=None,indices_right=None, refl=False):
    """WCAP_r and l have to be the size of the electrodes, H is sisl.Hamiltonian object with nsc transverse only"""
    nk = len(K)
    ne = len(E)
    nl = WCAP_l.shape[1]
    nr = WCAP_r.shape[1]
    n  = H.no
    if type(indices_left) != slice:
        indices_left = slice(0,nl)
    if type(indices_right) != slice:
        indices_right = slice(n-nr,n)
    Id_ez          = (np.tile(np.eye(n), [ne,nk,1,1]).transpose(1,2,3,0)*(E+1j*eta)).transpose(3,0,1,2)
    WCAP_left             = np.zeros((n,n), dtype=np.complex128)
    WCAP_right            = np.zeros((n,n), dtype=np.complex128)
    WCAP_left[indices_left,indices_left]    = WCAP_l
    WCAP_right[indices_right,indices_right] = WCAP_r
    Gl = np.tile( 1j*(WCAP_left  - WCAP_left.conj().T) , [ne,nk,1,1])
    Gr = np.tile( 1j*(WCAP_right - WCAP_right.conj().T) , [ne,nk,1,1])
    Hprime = np.tile( WCAP_left + WCAP_right , [nk,1,1] )
    nsc = [1,1,1]
    nsc[bc_axis] = 3
    H.set_nsc(nsc)
    for i,k in enumerate(K):
        Hprime[i,:,:] += H.Hk([k,k,k], format='array')
    G = np.linalg.inv( Id_ez - np.tile( Hprime , [ne,1,1,1] ) )
    Te = np.trace(np.matmul(np.matmul(G,Gl),np.matmul(G.conj().transpose(0,1,3,2),Gr)), axis1=2, axis2=3)
    res = Te
    if refl:
        Re = np.trace(np.matmul(np.matmul(G,Gl),np.matmul(G.conj().transpose(0,1,3,2),Gl)), axis1=2, axis2=3) - 1j*np.trace(np.matmul(Gl,G-G.conj().transpose(0,1,3,2)), axis1=2, axis2=3)
        res = np.array([Te,Re])
    if kavg and eavg:
        return np.mean(res, axis=(-1,-2))
    if kavg:
        return np.mean(res, axis=-1)
    elif eavg:
        return np.mean(res, axis=-2)
    else:
        return res

def CAP(f, geometry, side, mode="r", P=[], axis=0, epsilon_r=0.66):
    """f takes 2 arguments: r, dr_CAP
    Return the Self-energy associated to this Complex absorbing potential and geometry (numpy array)
    mode can be 'r'(real space) or 'n'(partition space). If mode is 'n', a partition must be precised"""
    dH_CAP = sisl.Hamiltonian(geometry, dtype='complex128')
    
    if type(axis) == int:
        r = geometry.xyz[:, axis]
    else:
        r = axis[0]*geometry.xyz[:,0] + axis[1]*geometry.xyz[:,1] + axis[2]*geometry.xyz[:,2]

    if side == 'right':
        r = r - np.min(r)
        if len(P) > 0: #if partition is input, the potential's r is the average over the partition
            dr = np.max(r)
            for i in range(len(P)-1):
                if P[i+1] > len(r):
                    break
                if mode == "r":
                    r[P[i]:P[i+1]] = np.mean(r[P[i]:P[i+1]])
                elif mode == "n":
                    r[P[i]:P[i+1]] = i/(len(P)-2)*dr
        Wr = f(r, np.max(r)+epsilon_r)
        
    if side == 'left':
        r = np.max(r) - r
        if len(P) > 0:
            dr = np.max(r)
            for i in range(len(P)-1):
                if P[i+1] > len(r):
                    break
                if mode == "r":
                    r[P[i]:P[i+1]] = np.mean(r[P[i]:P[i+1]])
                elif mode == "n":
                    r[P[i]:P[i+1]] = i/(len(P)-2)*dr
        Wr = f(r, np.max(r)+epsilon_r)

    orbs = dH_CAP.geom.a2o([i for i in range(len(r))]) # if you have just 1 orb per atom, then orb = ia
    for orb,wr in zip(orbs, Wr):
        dH_CAP[orb, orb] = complex(0, -wr)

    return dH_CAP.Hk(format="array", dtype=np.complex128)

def wang(r, dr, c, l=None, dl=None):
    """Most popular CAP function"""
    Ry_a02 = 27.2/2*0.53**2
    return Ry_a02 * (2*np.pi/dr)**2 * 4/c**2 * (dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2)

def manolopoulos(r, dr, params, l=None, dl=None, Emin=6):
    Ry_a02 = 27.2/2*0.53**2
    kmin = (Emin/Ry_a02)**0.5
    #a = (1 - 16/c**3)
    #b = (1 - 17/c**3)/c**2
    a, b, c = params[0], params[1], params[2]
    delta = c/(2*dr*kmin)
    x = 2*delta*kmin*r
    return Ry_a02 * kmin**2 * ( a*x - b*x**3 + 4/(c-x)**2 - 4/(c+x)**2 )

def exp(r , dr, params, l=None, dl=None):
    return params[0]*( np.exp(params[1]*(r/(dr-r))**params[2]) - 1 )

def mononomial(r, dr, params, l=None, dl=None):
    return params[0]*(r/dr)**params[1]

def osc_wang(r, dr, params, periods=None, l=None, dl=None):
    Ry_a02 = 27.2/2*0.53**2
    return Ry_a02 * (2*np.pi/dr)**2 * 4/params[0]**2 * (dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2 - params[1]*np.sin(2*np.pi/params[2] *(r-params[3]))*np.exp(-params[4]*r/dr) )

def osc(r, dr, params, periods=None):
    """add one sinus for each period, params length must be 3*len(periods) + 1"""
    c = params[0]
    Ry_a02 = 27.2/2*0.53**2
    res = dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2
    for i,T in enumerate(periods):
        res -= params[3*i+1] * np.sin(2*np.pi/T*(r-params[3*i+2])) * np.exp(-params[3*i+3]*r/dr)
    return res * Ry_a02 * (2*np.pi/dr)**2 * 4/c**2

def plot_CAP(WCAP, geom, axis, vmin=None, vmax=None, s=None, cmap=None, R=[0.1,1.6], h=[0,-2.7]):
    """Scatter plot of a CAP matrix on a geometry"""
    Wi = WCAP.imag
    Wr = WCAP.real
    real = np.any(Wr)
    hop  = np.any(WCAP - np.diag(WCAP.diagonal()))
    if real and hop:
        figsize = [2,2]
    elif real or hop:
        figsize = [1,2]
    else:
        figsize = [1,1]
    plotn = 1
    plt.subplot(figsize[0],figsize[1],plotn)
    plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c="k", s=5)
    plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c=np.diag(Wi), vmin=vmin, vmax=vmax, s=s, cmap=cmap)
    plt.title("On-site imaginary part of CAP")
    plt.xlabel('x [Ang]')
    plt.ylabel('y [Ang]')
    plt.axis("equal")
    plt.colorbar(label = "WCAP [eV]")
    plotn += 1
    if real:
        plt.subplot(figsize[0],figsize[1],plotn)
        plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c="k", s=5)
        plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c=np.diag(Wr), vmin=vmin, vmax=vmax, s=s, cmap=cmap)
        plt.title("On-site real part of CAP")
        plt.xlabel('x [Ang]')
        plt.axis("equal")
        plt.colorbar(label = "WCAP [eV]")
        plotn += 1
    if hop:     
        H  = sisl.Hamiltonian(geom)
        H.construct([R,h])
        H_ = matrix_elements(H,pdir = 1-axis, tdir = axis,which = 0)
        X  = []
        Y  = []
        Wiplus = []
        Wrplus = []
        for i in range(Wi.shape[0]):
            for j in range(Wi.shape[1]):
                if abs(H_[1,i,j]) > 0 and abs(WCAP[i,j]) > 0: # in cell currents
                    X.append((geom.xyz[i,0] + geom.xyz[j,0])/2)
                    Y.append((geom.xyz[i,1] + geom.xyz[j,1])/2)
                    Wiplus.append(Wi[i,j])
                    Wrplus.append(Wr[i,j])
                elif abs(WCAP[i,j]) > 0 and i != j: # out of cell currents
                    if axis==0:
                        X.append((geom.xyz[i,0] + geom.xyz[j,0]  )/2)
                        Y.append((geom.xyz[i,1] + geom.xyz[j,1] + geom.cell[1-axis,1-axis] )/2)
                        X.append((geom.xyz[i,0] + geom.xyz[j,0]  )/2)
                        Y.append((geom.xyz[i,1] + geom.xyz[j,1] - geom.cell[1-axis,1-axis] )/2)
                    elif axis==1:
                        X.append((geom.xyz[i,0] + geom.xyz[j,0] + geom.cell[1-axis,1-axis] )/2)
                        Y.append((geom.xyz[i,1] + geom.xyz[j,1]  )/2)
                        X.append((geom.xyz[i,0] + geom.xyz[j,0] - geom.cell[1-axis,1-axis] )/2)
                        Y.append((geom.xyz[i,1] + geom.xyz[j,1]  )/2)
                    Wiplus.append(Wi[i,j])
                    Wiplus.append(Wi[i,j])
                    Wrplus.append(Wr[i,j])
                    Wrplus.append(Wr[i,j])
        plt.subplot(figsize[0],figsize[1],plotn)
        mark = ['H','h']
        plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c="k", s=5)
        plt.scatter(np.array(X), np.array(Y), c=np.array(Wiplus), marker=mark[axis], vmin=vmin, vmax=vmax, s=s, cmap=cmap)
        plt.title('Imaginary part of hopping')
        plt.ylabel('y [Ang]')
        plt.xlabel('x [Ang]')
        plt.axis('equal')
        plt.colorbar(label = "WCAP [eV]")
        plotn += 1
        if real:
            plt.subplot(figsize[0],figsize[1],plotn)
            plt.scatter(geom.xyz[:,0],geom.xyz[:,1],c="k", s=5)
            plt.scatter(np.array(X), np.array(Y), c=np.array(Wrplus), marker=mark[axis], vmin=vmin, vmax=vmax, s=s, cmap=cmap)
            plt.title('Real part of hopping')
            plt.xlabel('x [Ang]')
            plt.axis('equal')
            plt.colorbar(label = "WCAP [eV]")
    plt.show()