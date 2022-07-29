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

def trans(E, K, H_left, V_left, H_center_plus, H_right, V_right, eavg=False, kavg=False, eta=1e-3, indices_left=None, indices_right=None, indices_device=None):
    """Transmission"""
    nk = len(K)
    Te = np.zeros((len(E),nk), dtype=np.complex128)
    nl = H_left.shape[1]
    nr = H_right.shape[1]
    nd = H_center_plus.shape[1]
    if type(indices_left) != list: #SOLUTION DE SECOURS
        indices_left   = [i for i in range(nl)]
    if type(indices_right) != list: #SOLUTION DE SECOURS
        indices_right  = [i for i in range(nd-nr,nd)]
    if type(indices_device) != list: #SOLUTION DE SECOURS
        indices_device = [i for i in range(nl,nd-nr)]
    
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        hk_dplus = phase_mult(H_center_plus, k)
        vk_dl    = hk_dplus[indices_device,:][:,indices_left ]
        vk_dr    = hk_dplus[indices_device,:][:,indices_right]
        hk_d     = hk_dplus[indices_device,:][:,indices_device]
        hk_r     = phase_mult(H_right,k)
        vk_r     = phase_mult(V_right,k)

        def trans_e(e, hk_l, vk_l, vk_dl, vk_dr, hk_d, hk_r, vk_r, eta=1e-3):
            """Return transmission coeff for a certain energy e
            CAPk_eig_l can be the tuple (eigenvectors, eigenvalues) for H+WCAP"""
            #if CAPk_eig_l != None and CAPk_eig_r != None:
            SE_l = SE(e+1j *eta, hk_l, vk_l)
            Il              = np.eye(hk_l.shape[1])
            g_surface_left  = np.linalg.inv(Il*(e+1j*eta) - hk_l - SE_l)
            se_left         = vk_dl.dot(g_surface_left).dot(vk_dl.T.conj())
            SE_r = SE(e+1j *eta, hk_r, vk_r)
            Ir              = np.eye(hk_r.shape[1])
            g_surface_right = np.linalg.inv(Ir*(e+1j*eta) - hk_r - SE_r)
            se_right        = vk_dr.dot(g_surface_right).dot(vk_dr.T.conj())
            Id = np.eye(hk_d.shape[1])
            I_H = Id*(e+1j*eta) - hk_d- se_left - se_right
            G = np.linalg.inv(I_H) 
            Gl = 1j * (se_left  - se_left.conj().T)
            Gr = 1j * (se_right - se_right.conj().T)
            return np.trace(np.linalg.multi_dot([G,Gl,G.conj().T,Gr]))

        trans_vect_e = np.vectorize(lambda e : trans_e(e, hk_l, vk_l, vk_dl, vk_dr, hk_d, hk_r, vk_r, eta)) #vectorize to enable arrays of energy
        Te[:,i] = trans_vect_e(E)
    
    if kavg and eavg:
        return np.mean(Te)
    elif kavg:
        return np.mean(Te, axis=1)
    elif eavg:
        return np.mean(Te, axis=0)
    else:
        return Te

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
   
def PDOS(E, K, H_left, V_left, H_center_plus, H_right, V_right, asum=True, eavg=False, kavg=True, eta_el=1e-3, eta=0):
    """Return projected density of state as a vector"""
    nk = len(K)
    nl = H_left.shape[1]
    nr = H_right.shape[1]
    nd = H_center_plus.shape[1]
    indices_left   = [i for i in range(nl)]
    indices_device = [i for i in range(nl,nd-nr)]
    indices_right  = [i for i in range(nd-nr,nd)]
    P = np.zeros((nd-nl-nr,len(E),nk))
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        hk_dplus = phase_mult(H_center_plus, k)
        vk_dl    = hk_dplus[indices_device,:][:,indices_left ]
        vk_dr    = hk_dplus[indices_device,:][:,indices_right]
        hk_d     = hk_dplus[indices_device,:][:,indices_device]
        hk_r     = phase_mult(H_right,k)
        vk_r     = phase_mult(V_right,k)
        for j,e in enumerate(E):
            Il              = np.eye(hk_l.shape[1])
            g_surface_left  = np.linalg.inv(Il*(e+1j*eta_el) - hk_l - SE(e+1j *eta_el, hk_l, vk_l))
            se_left         = vk_dl.dot(g_surface_left).dot(vk_dl.T.conj())
            Ir              = np.eye(hk_r.shape[1])
            g_surface_right = np.linalg.inv(Ir*(e+1j*eta_el) - hk_r - SE(e+1j *eta_el, hk_r, vk_r))
            se_right        = vk_dr.dot(g_surface_right).dot(vk_dr.T.conj())
            Id = np.eye(hk_d.shape[1])
            I_H = Id*(e+1j*eta) - hk_d- se_left - se_right
            G = np.linalg.inv(I_H)
            P[:,j,i] = -1/np.pi * np.diag(G.imag)
    if asum:
        P = np.mean(P, axis=0)
        if kavg and eavg:
            return np.mean(P)
        elif kavg:
            return np.mean(P, axis=1)
        elif eavg:
            return np.mean(P, axis=0)
        else:
            return P
    else:
        if kavg and eavg:
            return np.mean(P)
        elif kavg:
            return np.mean(P, axis=2)
        elif eavg:
            return np.mean(P, axis=1)
        else:
            return P

def refl(E, K, H_left, V_left, H_center_plus, H_right, V_right, eavg=False, kavg=False, eta=1e-3, indices_left=None, indices_right=None, indices_device=None):
    nk = len(K)
    Te = np.zeros((len(E),nk), dtype=np.complex128)
    nl = H_left.shape[1]
    nr = H_right.shape[1]
    nd = H_center_plus.shape[1]
    if type(indices_left) != list: #SOLUTION DE SECOURS
        indices_left   = [i for i in range(nl)]
    if type(indices_right) != list: #SOLUTION DE SECOURS
        indices_right  = [i for i in range(nd-nr,nd)]
    if type(indices_device) != list: #SOLUTION DE SECOURS
        indices_device = [i for i in range(nl,nd-nr)]
    
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        hk_dplus = phase_mult(H_center_plus, k)
        vk_dl    = hk_dplus[indices_device,:][:,indices_left ]
        vk_dr    = hk_dplus[indices_device,:][:,indices_right]
        hk_d     = hk_dplus[indices_device,:][:,indices_device]
        hk_r     = phase_mult(H_right,k)
        vk_r     = phase_mult(V_right,k)

        def trans_e(e, hk_l, vk_l, vk_dl, vk_dr, hk_d, hk_r, vk_r, eta=1e-3):
            """Return reflection coeff for a certain energy e"""
            SE_l = SE(e+1j *eta, hk_l, vk_l)
            Il              = np.eye(hk_l.shape[1])
            g_surface_left  = np.linalg.inv(Il*(e+1j*eta) - hk_l - SE_l)
            se_left         = vk_dl.dot(g_surface_left).dot(vk_dl.T.conj())
            SE_r = SE(e+1j *eta, hk_r, vk_r)
            Ir              = np.eye(hk_r.shape[1])
            g_surface_right = np.linalg.inv(Ir*(e+1j*eta) - hk_r - SE_r)
            se_right        = vk_dr.dot(g_surface_right).dot(vk_dr.T.conj())
            Id = np.eye(hk_d.shape[1])
            I_H = Id*(e+1j*eta) - hk_d- se_left - se_right
            G = np.linalg.inv(I_H) 
            Gl = 1j * (se_left  - se_left.conj().T)
            Gr = 1j * (se_right - se_right.conj().T)
            return np.trace(np.linalg.multi_dot([G,Gl,G.conj().T,Gl])) - 1j*np.trace(np.dot(Gl,G-G.conj().T))

        trans_vect_e = np.vectorize(lambda e : trans_e(e, hk_l, vk_l, vk_dl, vk_dr, hk_d, hk_r, vk_r, eta)) #vectorize to enable arrays of energy
        Te[:,i] = trans_vect_e(E)
    
    if kavg and eavg:
        return np.mean(Te)
    elif kavg:
        return np.mean(Te, axis=1)
    elif eavg:
        return np.mean(Te, axis=0)
    else:
        return Te

def refl_Block(E, nk, H_left, V_left, V_dev_left, H_device, P, Tbulk, WCAP=None, gs_left_ke=None, eavg=False, kavg=False, bc_axis=1, eta=1e-3):
    """As refl but with blocks. Only H_device is a sisl hamiltonian"""
    K              = np.linspace(0,1-1/nk, nk)
    ne = len(E)
    Re             = np.zeros((ne,nk), dtype=np.complex128)
    nl             = H_left.shape[1]
    nd             = H_device.no
    Il             = np.eye(nl)
    Id_ez          = (np.tile(np.eye(nd), [ne,1,1]).transpose(1,2,0)*(E+1j*eta)).transpose(2,0,1)
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        k_ = [0,0,0]
        k_[bc_axis] = k
        hk_d = H_device.Hk(k_)
        vk_dl    = phase_mult(V_dev_left, k)
        
        if isinstance(WCAP, np.ndarray):
            hk_d += WCAP
            
        if isinstance(gs_left_ke, np.ndarray): #option to reuse already stored left self-energy
                g_surface_left = gs_left_ke[:,i]
        else:
                g_surface_left  = np.array([np.linalg.inv(Il*(e+1j*eta) - hk_l - SE(e+1j *eta, hk_l, vk_l)) for e in E]) 
        
        se_left     = vk_dl.dot(g_surface_left.transpose(1,2,0)).transpose(2,0,1).dot(vk_dl.T.conj())
        I_H = Id_ez - np.tile(np.asarray(hk_d),[ne,1,1]) - se_left
        
        f, S = test_partition_2d_sparse_matrix(sp.identity(P[-1]).tocsr(),P)
        nS   = slices_to_npslices(S)
        i1,j1,d1 = [], [], []
        for j in range(ne):
            i0,j0,d0 = sp.find(I_H[j])
            i1.append(i0) ; j1.append(j0) ; d1.append(d0)
        Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)
        ia = [j for j in range(len(Av))]
        ib = [j for j in range(len(Bv))]
        ic = [j for j in range(len(Cv))]
        BTD = block_td(Av,Bv, Cv,ia, ib, ic)
        mask = np.eye(BTD.N) #only need to compute first block of inverse
        G = BTD.Invert(mask).Block(0,0)
        n0 = G.shape[1]
        Gl = 1j * (se_left[:,:n0,:n0]  - se_left[:,:n0,:n0].conj().transpose(0,2,1))
        Re[:,i] = Tbulk[:,i] - 1j*np.trace(np.matmul(Gl,G-G.conj().transpose(0,2,1)), axis1=1, axis2=2) + np.trace(np.matmul(np.matmul(G,Gl),np.matmul(G.conj().transpose(0,2,1),Gl)), axis1=1, axis2=2)

    if kavg and eavg:
        return np.mean(Re)
    elif kavg:
        return np.mean(Re, axis=1)
    elif eavg:
        return np.mean(Re, axis=0)
    else:
        return Re

def bond_currents(E, K, H_left, V_left, H_center_plus, H_right, V_right, eavg=False, kavg=False, eta=1e-3, indices_left=None, indices_right=None, indices_device=None):
    """Bond current matrix inside the device"""
    nk = len(K)
    nl = H_left.shape[1]
    nr = H_right.shape[1]
    nd = H_center_plus.shape[1]
    J  = np.zeros((len(E),nk,nd-nl-nr,nd-nl-nr))
    if type(indices_left) != list: #SOLUTION DE SECOURS
        indices_left   = [i for i in range(nl)]
    if type(indices_right) != list: #SOLUTION DE SECOURS
        indices_right  = [i for i in range(nd-nr,nd)]
    if type(indices_device) != list: #SOLUTION DE SECOURS
        indices_device = [i for i in range(nl,nd-nr)]
    
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        hk_dplus = phase_mult(H_center_plus, k)
        vk_dl    = hk_dplus[indices_device,:][:,indices_left ]
        vk_dr    = hk_dplus[indices_device,:][:,indices_right]
        hk_d     = hk_dplus[indices_device,:][:,indices_device]
        hk_r     = phase_mult(H_right,k)
        vk_r     = phase_mult(V_right,k)
        
        for j,e in enumerate(E):
            SE_l = SE(e+1j *eta, hk_l, vk_l)
            Il              = np.eye(hk_l.shape[1])
            g_surface_left  = np.linalg.inv(Il*(e+1j*eta) - hk_l - SE_l)
            se_left         = vk_dl.dot(g_surface_left).dot(vk_dl.T.conj())
            
            SE_r = SE(e+1j *eta, hk_r, vk_r)
            Ir              = np.eye(hk_r.shape[1])
            g_surface_right = np.linalg.inv(Ir*(e+1j*eta) - hk_r - SE_r)
            se_right        = vk_dr.dot(g_surface_right).dot(vk_dr.T.conj())
            Id = np.eye(hk_d.shape[1])
            I_H = Id*(e+1j*eta) - hk_d- se_left - se_right
            if issparse(I_H):
                G = scilin.inv(I_H)
            else:
                G = np.linalg.inv(I_H) 
            Gl = 1j * (se_left  - se_left.conj().T)
            Al = np.linalg.multi_dot([G,Gl,G.conj().T])
            J[j,i] = 2.43e-4 * np.multiply(Al, hk_d).imag
    
    if kavg and eavg:
        return np.mean(np.mean(J, axis=1), axis=0)
    elif kavg:
        return np.mean(J, axis=1)
    elif eavg:
        return np.mean(J, axis=0)
    else:
        return J

def plot_J(J, H, axis, vmin=None, vmax=None, s=None, R=[0.1,1.6], hop=[0,-2.7], mode=0):
    """Plot bond current on geometry designed by H"""
    H_= matrix_elements(H,pdir = 1-axis, tdir = axis,which = 0, mode=mode)
    X = []
    Y = []
    Jplus = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(H_[H_.shape[0]//2,i,j]) > 0 and J[i,j] > 0: # in cell currents
                X.append((H.xyz[i,0] + H.xyz[j,0])/2)
                Y.append((H.xyz[i,1] + H.xyz[j,1])/2)
                Jplus.append(J[i,j])
            elif J[i,j] > 0 and i != j: # out of cell currents
                if axis==0:
                    X.append((H.xyz[i,0] + H.xyz[j,0]  )/2)
                    Y.append((H.xyz[i,1] + H.xyz[j,1] + H.cell[1-axis,1-axis] )/2)
                    X.append((H.xyz[i,0] + H.xyz[j,0]  )/2)
                    Y.append((H.xyz[i,1] + H.xyz[j,1] - H.cell[1-axis,1-axis] )/2)
                elif axis==1:
                    X.append((H.xyz[i,0] + H.xyz[j,0] + H.cell[1-axis,1-axis] )/2)
                    Y.append((H.xyz[i,1] + H.xyz[j,1]  )/2)
                    X.append((H.xyz[i,0] + H.xyz[j,0] - H.cell[1-axis,1-axis] )/2)
                    Y.append((H.xyz[i,1] + H.xyz[j,1]  )/2)
                Jplus.append(J[i,j])
                Jplus.append(J[i,j])

    mark = ['H','h']
    plt.scatter(np.array(X), np.array(Y), c=np.array(Jplus), marker=mark[axis], vmin=vmin, vmax=vmax, s=s)
    plt.colorbar(label = "Positive bond current")
    plt.scatter(H.xyz[:,0],H.xyz[:,1],c="k", s=5)
    plt.xlabel('x [Ang]')
    plt.ylabel('y [Ang]')
    plt.axis('equal')
    plt.show()

def quasi_particle(E_guess,k,H_left, V_left, H_center_plus, H_right, V_right, eta, tol, maxiter):
    """Solves the quasi-particle equation (H+SEH).Psi = E.Psi, based on a first guess
    Returns the double dot of this state with the antihermitian self-energy (close to zero means a bound state)
    Returns the energy of the eigenstate and the eigenstate"""
    E0 = E_guess
    E1 = E_guess+2*tol #just so that it 
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    hkl = phase_mult(H_left,k)
    vkl = phase_mult(V_left,k)
    hkr = phase_mult(H_right,k)
    vkr = phase_mult(V_right,k)
    hkd = phase_mult(H_center_plus,k)

    nl = hkl.shape[0]
    nr = hkr.shape[0]
    nd = hkd.shape[0]

    SE_ = np.zeros((nd,nd), dtype=np.complex128)
    cpt = 0
    
    while abs(E1-E0) > tol and cpt < maxiter:
        cpt += 1
        E0 = E1
        SE_l = SE(E0+1j*eta,hkl,vkl)
        SE_r = SE(E0+1j*eta,hkr,vkr)
        SE_[:nl,:nl]   = SE_l
        SE_[-nr:,-nr:] = SE_r
        SE_S = (SE_ + SE_.conj().T)/2

        Ev, V = np.linalg.eigh(hkd+SE_S)
        idx1, E1 = find_nearest(Ev, E0)

    SE_A = (SE_ - SE_.conj().T)/2
    state = V[:,idx1]
    bound = abs(state.conj().T.dot(SE_A).dot(state))
    return bound, E1, state