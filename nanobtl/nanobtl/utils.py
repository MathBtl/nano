import sisl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import issparse
import scipy.sparse as sp
import scipy.sparse.linalg as scilin
from Block_matrices.Block_matrices import block_td, Build_BTD_vectorised, test_partition_2d_sparse_matrix, slices_to_npslices, Blocksparse2Numpy

"""------------------------------------------------------ Utils ---------------------------------------------------------------------------"""
    
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