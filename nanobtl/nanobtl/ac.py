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

def transAC(E, Omega, K, H, SE_l, SE_r, bc_axis, eta=1e-3, dtype=np.complex128):
    """H is sisl.Hamiltonian object with nsc transverse only, energy independent self-energies
    Returns the AC coefficients T and Y (k-averaged)"""
    ne = len(E)
    no = len(Omega)
    nk = len(K)
    hw = 6.582119569e-16*Omega
    nl = SE_l.shape[1]
    nr = SE_r.shape[1]
    n  = H.no
    Id             = np.tile(np.eye(n, dtype=dtype), [ne,no,1,1])
    Id_ez          = (Id.transpose(1,2,3,0)*(E+1j*eta)).transpose(3,0,1,2)
    Id_omega       = (Id.transpose(0,2,3,1)*(hw)).transpose(0,3,1,2)
    SE_left             = np.zeros((n,n), dtype=dtype)
    SE_right            = np.zeros((n,n), dtype=dtype)
    SE_left[:nl,:nl]    = SE_l
    SE_right[-nr:,-nr:] = SE_r
    Gl = np.tile( 1j*(SE_left  - SE_left.conj().T) , [ne,no,1,1])
    Gr = np.tile( 1j*(SE_right - SE_right.conj().T) , [ne,no,1,1])

    Te = np.zeros((ne,no), dtype=dtype)
    Re = np.zeros((ne,no), dtype=dtype)
    Yl = np.zeros((ne,no), dtype=dtype)
    Yr = np.zeros((ne,no), dtype=dtype)
    
    nsc = [1,1,1]
    nsc[bc_axis] = 3
    H.set_nsc(nsc)
    for i,k in enumerate(K):
        I_H   = Id_ez - np.tile( H.Hk([k,k,k], format='array', dtype=dtype) + SE_left + SE_right , [ne,no,1,1] )
        Gdag  = np.linalg.inv( I_H ).conj().transpose(0,1,3,2)
        Gplus = np.linalg.inv( I_H + Id_omega ) #CAN BE IMPROVED BY ONLY 1 INV IF E GRID IS COMPATIBLE WITH OMEGA GRID
        
        Te += np.trace(np.matmul(np.matmul(Gplus,Gl),np.matmul(Gdag,Gr)), axis1=2, axis2=3)/nk
        Re += (np.trace(np.matmul(np.matmul(Gplus,Gl),np.matmul(Gdag,Gl)), axis1=2, axis2=3) - 1j*np.trace(np.matmul(Gl,Gplus-Gdag), axis1=2, axis2=3))/nk
        Yl += -1j*np.trace(np.matmul(np.matmul(Gplus,Gl),Gdag), axis1=2, axis2=3)/nk
        Yr += -1j*np.trace(np.matmul(np.matmul(Gplus,Gr),Gdag), axis1=2, axis2=3)/nk
    
    return Te, Re, Yl, Yr

def transAC_block(E, Omega, K, H, SE_l, SE_r, bc_axis, P, eta=1e-3, dtype=np.complex128):
    """H is sisl.Hamiltonian object with nsc transverse only, energy independent self-energies
    P is the partition of the hamiltonian in order to build the BTD matrix
    Returns the AC coefficients T and Y (k-averaged)"""
    ne = len(E)
    no = len(Omega)
    nk = len(K)
    hw = 6.582119569e-16*Omega
    nl = SE_l.shape[1]
    nr = SE_r.shape[1]
    n  = H.no
    Id             = np.tile(np.eye(n, dtype=dtype), [ne,no,1,1])
    Id_ez          = (Id.transpose(1,2,3,0)*(E+1j*eta)).transpose(3,0,1,2)
    Id_omega       = (Id.transpose(0,2,3,1)*(hw)).transpose(0,3,1,2)
    
    SE_left             = np.zeros((n,n), dtype=dtype)
    SE_right            = np.zeros((n,n), dtype=dtype)
    SE_left[:nl,:nl]    = SE_l
    SE_right[-nr:,-nr:] = SE_r
    Gl = np.tile( -2*SE_left.imag , [ne*no,1,1])
    Gr = np.tile( -2*SE_right.imag , [ne*no,1,1])

    Te = np.zeros((ne*no), dtype=dtype)
    Re = np.zeros((ne*no), dtype=dtype)
    Yl = np.zeros((ne*no), dtype=dtype)
    Yr = np.zeros((ne*no), dtype=dtype)
    
    mask = np.eye(len(P)-1) #only need to compute columns corresponding to electrods of inverse
    mask[:,0:int(np.argwhere(P==nl))] = 1
    mask[:,int(np.argwhere(P==n-nr)):] = 1
    
    nsc = [1,1,1]
    nsc[bc_axis] = 3
    H.set_nsc(nsc)
    for i,k in enumerate(K):
        I_H      = Id_ez - np.tile( H.Hk([k,k,k], format='array', dtype=dtype) + SE_left + SE_right , [ne,no,1,1] )
        I_H_plus = I_H + Id_omega
        
        f, S = test_partition_2d_sparse_matrix(sp.identity(P[-1]).tocsr(),P)
        nS   = slices_to_npslices(S)
        i1,j1,d1 = [], [], []
        i1_plus,j1_plus,d1_plus = [], [], []

        for j in range(ne):
            for p in range(no):
                i0,j0,d0 = sp.find(I_H[j,p])
                i1.append(i0) ; j1.append(j0) ; d1.append(d0)
                i0,j0,d0 = sp.find(I_H_plus[j,p])
                i1_plus.append(i0) ; j1_plus.append(j0) ; d1_plus.append(d0)
        
        Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)
        
        ia = [j for j in range(len(Av))]
        ib = [j for j in range(len(Bv))]
        ic = [j for j in range(len(Cv))]
        IH_BTD = block_td(Av,Bv, Cv,ia, ib, ic)
        
        Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1_plus), np.vstack(j1_plus), np.vstack(d1_plus), nS)
        
        ia = [j for j in range(len(Av))]
        ib = [j for j in range(len(Bv))]
        ic = [j for j in range(len(Cv))]
        IH_plus_BTD = block_td(Av,Bv, Cv,ia, ib, ic)
        
        G_BTD = IH_BTD.Invert(mask)
        Gdag = Blocksparse2Numpy(G_BTD, IH_BTD.all_slices).conj().transpose(0,2,1)
        
        G_plus_BTD = IH_plus_BTD.Invert(mask)
        Gplus = Blocksparse2Numpy(G_plus_BTD, IH_plus_BTD.all_slices)
        
        Te += np.trace(np.matmul(np.matmul(Gplus,Gl),np.matmul(Gdag,Gr)), axis1=1, axis2=2)/nk
        Re += (np.trace(np.matmul(np.matmul(Gplus,Gl),np.matmul(Gdag,Gl)), axis1=1, axis2=2) - 1j*np.trace(np.matmul(Gl,Gplus-Gdag), axis1=1, axis2=2))/nk
        Yl += -1j*np.trace(np.matmul(np.matmul(Gplus,Gl),Gdag), axis1=1, axis2=2)/nk
        Yr += -1j*np.trace(np.matmul(np.matmul(Gplus,Gr),Gdag), axis1=1, axis2=2)/nk
    
    return Te.reshape((ne,no)), Re.reshape((ne,no)), Yl.reshape((ne,no)), Yr.reshape((ne,no))

def fermidirac(E, V, T=298):
    k = 8.617333262e-5 #eV/K
    return 1/(1+np.exp((E-V)/(k*T)))

def d_fermidirac(E, V, T=298):
    k = 8.617333262e-5 #eV/K
    return -1/(k*T) * np.exp((E-V)/(k*T)) / (1+np.exp((E-V)/(k*T)))**2

def inteT(E, V, Omega, T, Temp=298):
    """Integrand of the T and R AC coefficients"""
    hbar = 6.582119569e-16 #eV/Hz
    hw = Omega*hbar
    inteT = np.zeros_like(T)
    for i, w in enumerate(hw):
        if hw[i] > 0:
            inteT[:,i] = (fermidirac(E,V,Temp) - fermidirac(E+hw[i],V,Temp))/hw[i] * T.T[i]
        else:
            inteT[:,i] = -d_fermidirac(E,V,Temp) * T.T[i]
    return inteT

def inteY(E, V, Omega, Y,Temp=298):
    """Integrand of the Y AC coefficients"""
    hbar = 6.582119569e-16 #eV/Hz
    hw = Omega*hbar
    inteY = np.zeros_like(Y)
    for i, w in enumerate(hw):
        inteY[:,i] = (fermidirac(E,V,Temp) - fermidirac(E+hw[i],V,Temp)) * Y.T[i]
    return inteY

def admittance(E, Omega, K, H, SE_l, SE_r, bc_axis, V=0, block=False, P=[], eta=1e-3, Temp=298, dtype=np.complex128):
    """Return the admittance of the system described buy the sisl Hamiltonian H and the energy independent
    numpy matrices SE_l and SE_r, along transport axis, for several frequencies Omega
    V is the gate voltage, close to zero"""
    if block:
        Tlr, Tll, Yl, Yr = transAC_block(E, Omega, K, H, SE_l, SE_r, bc_axis, P, eta=eta, dtype=dtype)
    else:
        Tlr, Tll, Yl, Yr = transAC(E, Omega, K, H, SE_l, SE_r, bc_axis, eta=eta, dtype=dtype)
    iTlr = inteT(E, V, Omega, Tlr, Temp=Temp)
    iTll = inteT(E, V, Omega, Tll, Temp=Temp)
    iYl = inteY(E, V, Omega, Yl, Temp=Temp)
    iYr = inteY(E, V, Omega, Yr, Temp=Temp)
    dE = E[1] - E[0]
    Yclr = np.sum(iTlr,axis=0)*dE
    Ycll = np.sum(iTll,axis=0)*dE
    Ydl = np.sum(iYl,axis=0)*dE
    Ydr = np.sum(iYr,axis=0)*dE
    Pl = np.zeros_like(Omega, dtype=dtype)
    for i,w in enumerate(Omega):
        if (Ydl+Ydr)[i] != 0:
            Pl[i] = -((Yclr+Ycll)[i]/(Ydl+Ydr)[i])
    return Yclr + Pl*Ydr