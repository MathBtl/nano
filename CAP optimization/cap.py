import sisl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import issparse
import scipy.sparse as sp
import scipy.sparse.linalg as scilin
from Block_matrices.Block_matrices import block_td, Build_BTD_vectorised, test_partition_2d_sparse_matrix, slices_to_npslices, Blocksparse2Numpy

"""------------------------------------------------------ Utils ---------------------------------------------------------------------------"""

def G_CAP(WCAP, H, V, nb_val, hermitian = False):
    """Diagonalization of H + WCAP"""
    if hermitian:
        Ec, Uc = np.linalg.eigh(H)
        UU = np.einsum('ik,kj->kij', Uc, Uc.T.conj())
    elif nb_val == "all":
        Ec, Uc = np.linalg.eig(H + WCAP)
        UU = np.einsum('ik,kj->kij', Uc, np.linalg.inv(Uc))
    else:
        Ec, Uc = scilin.eigs(H + WCAP, nb_val, sigma=0, which='LM')
        UU = np.einsum('ik,kj->kij', Uc, np.linalg.inv(Uc))
    B  = np.matmul(np.matmul(V,UU),V.conj().T)
    return B, Ec
    
def SE(z,H,V, eps=1e-15):
    """Self-energy with recursion algorithm"""
    n          =  len(H)
    DT=np.complex128
    S00 = np.eye(n).astype(DT)
    S01 = np.zeros((n,n)).astype(DT)
    alpha      =  np.zeros((n,n),dtype = DT)
    beta       =  np.zeros((n,n),dtype = DT)
    igb        =  np.zeros((n,n),dtype = DT)
    igb  [:,:] =  S00*z - H
    alpha[:,:] =  V - S01 * z
    beta [:,:] =  V.conj().T - S01.conj().T * z
    sse        =  np.zeros((n,n),dtype = DT)
    while True:
        gb       = np.linalg.inv(igb)
        gb_beta  = np.dot(gb,beta)
        gb_alpha = np.dot(gb,alpha)
        sse     += np.dot(alpha,gb_beta)
        igb     -= np.dot(alpha,gb_beta) + np.dot(beta,gb_alpha)
        alpha    = np.dot(alpha,gb_alpha)
        beta     = np.dot(beta,gb_beta)
        if (np.sum(np.abs(alpha))+np.sum(np.abs(beta)))<eps:
            return sse

def matrix_elements(H,pdir,tdir, which, mode = 0, dtype=np.float64):
    """Draw matrix used in the next phase_mult function from sisl Hamiltonian object"""
    nsc    = H.nsc
    no     = H.no
    num_p  = nsc[pdir]
    assert np.mod(num_p,2)==1
    out = np.zeros((num_p, no,no),dtype=dtype)
    R_count = 0
    for nR in range(-(num_p//2), num_p//2+1):
        for i in range(no):
            for j in range(no):
                if pdir==0 and tdir == 1:
                    tup = (nR, which,0)
                elif pdir==1 and tdir==0:
                    tup = (which, nR,0)
                element = H[i,j,tup]
                if mode==1:
                    element = element[0]
                out[R_count,i,j] = element
        R_count+=1
    return out

def phase_mult(A, k):
    """A is a (n,m,m) hamiltonian, first dimension is the periodic boundary condition
    Return bloch sum"""
    assert np.mod(A.shape[0],2)==1
    n = A.shape[0]//2
    N = np.arange(-n, n+1)
    K = k*N
    return  (A.transpose(1,2,0)*np.exp(1j *2 * np.pi* K)).transpose(2,0,1).sum(axis=0)

def rmse(A,B):
    """root mean squared error between 2 vectors"""
    return (np.square(A - B)).mean()**0.5

def max_square(A,B):
    return (np.square(A - B)).max()

def geom_from_xyz(filename, atom, R, nsc):
    """build sisl geometry object from xyz file"""
    geom = sisl.io.xyz.xyzSile(filename).read_geometry()
    C  = sisl.Atom(atom, R = R)
    for i in range(geom.na):
        geom.atoms[i] = C
    geom.set_nsc(nsc)
    return geom

def btd_partition(H,P):
    """Test if the input partition to build the BTD is sufficient"""
    f, S = test_partition_2d_sparse_matrix(H.Hk(),P)
    if f > 0.999:
        print("Partition OK")
        return True
    else:
        print("Warning : partition NOT OK")
        return False
    
def g_surface(E, nk, H, V, eta=1e-3):
    """Electrode isolated green function shape (len(E),nk,H.shape[1],H.shape[1])"""
    K          = np.linspace(0,1-1/nk, nk)
    n          = H.shape[1]
    I          = np.eye(n)
    gs         = np.zeros((len(E),nk,n,n), dtype=np.complex128)
    for i,k in enumerate(K):
        hk     = phase_mult(H, k)
        vk     = phase_mult(V, k)
        for j,e in enumerate(E):
            gs[j,i]  = np.linalg.inv(I*(e+1j*eta) - hk - SE(e+1j *eta, hk, vk))
    return gs

def block_diag(arr, num):
    """return a block diagonal matrix with num times arr on its diagonal"""
    rows, cols = arr.shape
    result = np.zeros((num * rows, num * cols), dtype=arr.dtype)
    for k in range(num):
        result[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols] = arr
    return result

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def is_pos_def(x):
    """True if x is positive definite, False otherwise"""
    return np.all(np.linalg.eigvals(x) > 0)

def interpolateCAP(WCAP0,geom0,axis0,geom1,axis1):
    """Interpolate a diagonal CAP from one geometry into another, along an axis ([x,y,z])"""
    r0 = np.dot(geom0.xyz,axis0)/np.linalg.norm(axis0)
    r0 -= r0.min()
    r1 = np.dot(geom1.xyz,axis1)/np.linalg.norm(axis1)
    r1 -= r1.min()
    fimag = interp1d(r0,WCAP0.imag.diagonal())
    freal = interp1d(r0,WCAP0.real.diagonal())
    return np.diag( 1j*fimag(r1) + freal(r1) )

"""------------------------------------------------------ Refl, trans & PDOS ---------------------------------------------------------------------------"""

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

def refl(E, nk, H_left, V_left, H_center_plus, Tbulk, WCAP=None, gs_left_ke=None, eproc="max", kavg=False, eta=1e-3):
    """Left electrode and device only, the CAP is applied on the device, Tbulk is the transmission of the electrode
	WCAP can be k-dependent (first dimension)"""
    K              = np.linspace(0,1-1/nk, nk)
    if kavg:
        Re	       = np.zeros((len(E)), dtype=np.complex128)  
    else:
        Re         = np.zeros((len(E),nk), dtype=np.complex128)
    nl             = H_left.shape[1]
    nd             = H_center_plus.shape[1]
    indices_left   = [i for i in range(nl)]
    indices_device = [i for i in range(nl,nd)]
    Il             = np.eye(nl)
    Id             = np.eye(nd-nl)
    for i,k in enumerate(K):
        hk_l     = phase_mult(H_left, k)
        vk_l     = phase_mult(V_left, k)
        hk_dplus = phase_mult(H_center_plus, k)
        vk_dl    = hk_dplus[indices_device,:][:,indices_left ]
        hk_d     = hk_dplus[indices_device,:][:,indices_device]
        
        if isinstance(WCAP, np.ndarray) and len(WCAP.shape)==2: #k-independent CAP
            hk_d += WCAP
        elif isinstance(WCAP, np.ndarray) and len(WCAP.shape)==3: #k-dependent CAP
            hk_d += WCAP[i]
            
        for j,e in enumerate(E):
            if isinstance(gs_left_ke, np.ndarray): #option to reuse already stored left self-energy
                g_surface_left  = gs_left_ke[j,i]
            else:
                g_surface_left  = np.linalg.inv(Il*(e+1j*eta) - hk_l - SE(e+1j *eta, hk_l, vk_l))
            se_left   = vk_dl.dot(g_surface_left).dot(vk_dl.T.conj())
            I_H       = Id*(e+1j*eta) - hk_d- se_left
            if issparse(I_H):
                G = scilin.inv(I_H)
            else:
                G = np.linalg.inv(I_H) 
            Gl = 1j * (se_left  - se_left.conj().T)
            
            if kavg:
                Re[j]  += (Tbulk[j] - 1j*np.trace(Gl.dot(G-G.conj().T)) + np.trace(np.linalg.multi_dot([G,Gl,G.conj().T,Gl])))/nk
            else:
                Re[j,i] = Tbulk[j,i] - 1j*np.trace(Gl.dot(G-G.conj().T)) + np.trace(np.linalg.multi_dot([G,Gl,G.conj().T,Gl]))
    		
    if eproc == "avg":
        return np.mean(np.abs(Re), axis=0)
    elif eproc == "max":
        return np.max(np.abs(Re.real), axis=0)
    else:
        return Re

def refl1(E, K, H_left, V_left, H_center_plus, H_right, V_right, eavg=False, kavg=False, eta=1e-3, indices_left=None, indices_right=None, indices_device=None):
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
            """Return transmission coeff for a certain energy e"""
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

"""------------------------------------------------------ CAP ---------------------------------------------------------------------------"""

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

def CAP_eig(Weig, Vc, NC, f=None, side="right"):
    """Weig is a real vector of length NC*Hmin.no if f is None, or Hmin.no if f is a function
	Apply CAP on the eigenstates (Vc) of each unit cell (NC cells), with potentially a ramp up f(n)"""
    Vc_block = block_diag(Vc, NC)
    if f == None:
        W = np.linalg.multi_dot([Vc_block,np.diag(Weig),Vc_block.conj().T])
    else:
        Wdiag = np.concatenate([Weig*f(n) for n in range(NC)])
        if side == "left":
            Wdiag = np.flip(Wdiag)
        W = np.linalg.multi_dot([Vc_block,np.diag(Wdiag),Vc_block.conj().T])
    return -1j*W

def wang(r, dr, c, l=None, dl=None):
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

def nothing(r, dr, params, l=None, dl=None):
    return 0*r

def cos_wang(r, dr, params, l=None, dl=None):
    Ry_a02 = 27.2/2*0.53**2
    return Ry_a02 * (2*np.pi/dr)**2 * 4/params[0]**2 * (dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2 + params[1]*np.cos(2*np.pi/2.1 * r))

def sin_wang(r, dr, params, l=None, dl=None):
    Ry_a02 = 27.2/2*0.53**2
    return Ry_a02 * (2*np.pi/dr)**2 * 4/params[0]**2 * (dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2 + params[1]*np.sin(2*np.pi/2.1 * r))

def osc_wang(r, dr, params, periods=None, l=None, dl=None):
    Ry_a02 = 27.2/2*0.53**2
    return Ry_a02 * (2*np.pi/dr)**2 * 4/params[0]**2 * (dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2 - params[1]*np.sin(2*np.pi/params[2] *(r-params[3]))*np.exp(-params[4]*r/dr) )

def opt(r, dr, params, l=None, dl=None):
    return params

def perp_square(l, dl, params):
    return params[0]*(1-np.abs(l/dl)**params[1])**params[2]

def osc(r, dr, params, periods=None):
    """add one sinus for each period, params length must be 3*len(periods) + 1"""
    c = params[0]
    Ry_a02 = 27.2/2*0.53**2
    res = dr**2/(dr-r)**2 + dr**2/(dr+r)**2 - 2
    for i,T in enumerate(periods):
        res -= params[3*i+1] * np.sin(2*np.pi/T*(r-params[3*i+2])) * np.exp(-params[3*i+3]*r/dr)
    return res * Ry_a02 * (2*np.pi/dr)**2 * 4/c**2

wang_perp = lambda r, dr, params, l, dl : wang(r,dr,3.84582436)*perp_square(l,dl, params)

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

"""------------------------------------------------------ AC ---------------------------------------------------------------------------"""

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
