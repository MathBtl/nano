import sisl
import numpy as np
import scipy.sparse as sp
import scipy.optimize as sciopt
import cap
import pandas as pd
import os
from time import time

"""------------------------------------------------------ Input ---------------------------------------------------------------------------"""

df = pd.read_csv("input.csv")

output_path="registre.csv"

for index, row in df.iterrows():

	start = time()

	Emin       = float(row['Emin'])
	Emax       = float(row['Emax'])
	ne         = int(row['ne'])
	kmin       = float(row['kmin'])
	kmax       = float(row['kmax'])
	nk         = int(row['nk'])
	eta        = float(row['eta'])
	kproc      = row['kproc'] # avg, none or inde
	if kproc == "avg":
		kavg = True
	elif kproc == "none" or kproc == "inde":
		kavg = False
	axis       = int(row['axis']) #transport axis
	dev_file  = "./geom/"+row['dev_file'] #the device geometry must contain the electrodes and must be ordered like left-dev-right
	elec_left_file  = "./geom/"+row['elec_left_file'] #same order as device
	elec_right_file = "./geom/"+row['elec_right_file'] #same order as device
	atom        = "C"
	r_atom      = 1.43420
	R           = [0.1, 1.5] #used only if a xyz file is input instead of a TSHS file
	hop         = [0., -2.7]	#to build the hamiltonian
	f_CAP       = cap.wang
	params_CAP0 = np.array([2.62])
	nsc         = eval(row['nsc']) #3 in the periodic boundary axis only
	if row['Tobj_file'][-3:] != "npy": #(ne,nk) shape
		Tobj_file = None
	else:
		Tobj_file = "./Tobj/"+row['Tobj_file']
	if row['Robj_file'][-3:] != "npy": #(ne,nk) shape
		Robj_file = None
	else:
		Robj_file = "./Tobj/"+row['Robj_file']
	if type(row['tol']) != float:
		tol = None
	else:
		tol        = row['tol']
	
	save_file  = "./res_opt/"+row['save_file']

	if "[" in row['mode']: #if input is a list, words possible: diag, real, hop, X0_zero, refl, sym, transym
		mode   = eval(row['mode']) 
	else: #can also be a string
		mode   = row['mode']
	
	sym = 'sym' in mode
	transym = 'transym' in mode
	real = 'real' in mode
	refl = 'refl' in mode

	"""------------------------------------------------------ Optim ---------------------------------------------------------------------------"""

	E = np.linspace(Emin,Emax,ne)
	K = np.linspace(kmin,kmax,nk)

	### SETUP HAMILTONIAN

	if dev_file[-4:] == "TSHS":
		Hd = sisl.io.siesta.tshsSileSiesta(dev_file).read_hamiltonian()
		Hl = sisl.io.siesta.tshsSileSiesta(elec_left_file).read_hamiltonian()
		Hr = sisl.io.siesta.tshsSileSiesta(elec_right_file).read_hamiltonian()
		mode_obj = 0
	else:
		geom = cap.geom_from_xyz(dev_file, atom, r_atom, nsc)
		Hd    = sisl.Hamiltonian(geom)
		Hd.construct([R, hop])
		
		geom_l = cap.geom_from_xyz(elec_left_file, atom, r_atom, nsc)
		Hl = sisl.Hamiltonian(geom_l)
		Hl.construct([R, hop])	
		geom_r = cap.geom_from_xyz(elec_right_file, atom, r_atom, nsc)
		Hr = sisl.Hamiltonian(geom_r)
		Hr.construct([R, hop])
		
		mode_obj = 0

	Hd.set_nsc(nsc)
	bc_axis = int(np.where(np.array(nsc) > 1)[0])

	elecs  = {"left":Hl, "right":Hr}
	
	### SETUP OBJECTIVE TRANSMISSION AND REFLECTION

	if Tobj_file == None:
		nsc_obj = nsc.copy()
		nsc_obj[axis] = 3
		Hd.set_nsc(nsc_obj)
		Hl.set_nsc(nsc_obj)
		Hr.set_nsc(nsc_obj)
		H_center_plus_obj = cap.matrix_elements(Hd, pdir = bc_axis, tdir = axis, which=0, mode=mode_obj)
		H_left_obj    = cap.matrix_elements(Hl,pdir = bc_axis, tdir = axis,which = 0, mode=mode_obj)
		V_left_obj    = cap.matrix_elements(Hl,pdir = bc_axis, tdir = axis,which = -1, mode=mode_obj)
		H_right_obj   = cap.matrix_elements(Hr,pdir = bc_axis, tdir = axis,which = 0, mode=mode_obj)
		V_right_obj   = cap.matrix_elements(Hr,pdir = bc_axis, tdir = axis,which = 1, mode=mode_obj)
		Tobj = cap.trans(E, K, H_left_obj, V_left_obj, H_center_plus_obj, H_right_obj, V_right_obj, eavg=False, kavg=kavg, eta=eta).real
		Hd.set_nsc(nsc)
		Hl.set_nsc(nsc)
		Hr.set_nsc(nsc)
	else:	
		Tobj = np.load(Tobj_file)

	if "refl" in mode:
		if Robj_file == None:
			nsc_obj = nsc.copy()
			nsc_obj[axis] = 3
			Hd.set_nsc(nsc_obj)
			Hl.set_nsc(nsc_obj)
			Hr.set_nsc(nsc_obj)
			H_center_plus_obj = cap.matrix_elements(Hd, pdir = bc_axis, tdir = axis, which=0, mode=mode_obj)
			H_left_obj    = cap.matrix_elements(Hl,pdir = bc_axis, tdir = axis,which = 0, mode=mode_obj)
			V_left_obj    = cap.matrix_elements(Hl,pdir = bc_axis, tdir = axis,which = -1, mode=mode_obj)
			H_right_obj   = cap.matrix_elements(Hr,pdir = bc_axis, tdir = axis,which = 0, mode=mode_obj)
			V_right_obj   = cap.matrix_elements(Hr,pdir = bc_axis, tdir = axis,which = 1, mode=mode_obj)
			Robj = cap.refl1(E, K, H_left_obj, V_left_obj, H_center_plus_obj, H_right_obj, V_right_obj, eavg=False, kavg=kavg, eta=eta).real
			Hd.set_nsc(nsc)
			Hl.set_nsc(nsc)
			Hr.set_nsc(nsc)
		else:
			Robj = np.load(Robj_file)
		Tobj = np.array([Tobj,Robj])
	else:
		Tobj = Tobj

	### SETUP VARIABLE VECTOR AND X0

	nvar    = 0
	no      = {}
	X_slice = {}
	indices = {}
	indptr  = {}
	X0      = []
	for side in ["left", "right"]:
		H_elec      = elecs[side]
		#nsc_CAP = [1,1,1]
		H_elec.set_nsc(nsc) #avoid hopping between last and first
		no_el     = H_elec.no
		X0_arr = np.zeros((no_el,no_el), dtype=np.complex128)
		if "diag" in mode:
			X0_arr += np.diag(np.diag(cap.CAP(lambda r,dr : f_CAP(r,dr,params_CAP0), elecs[side], side, axis=elecs[side].cell[axis]/np.linalg.norm(elecs[side].cell[axis])))-1j*eta) #just so that the first element is non zero
		if "hop" in mode:
			X0_arr += eta*(H_elec.Hk().toarray() - np.tril(H_elec.Hk().toarray())) #as the result will be symmetric, we only need the upper triangle, multiplied by eta because have to be little to satisfy Gamma positive definite
		if transym: #The first half of the matrix is one part of the transverse symmetry
			X0_csr     = sp.csr_matrix(np.abs(X0_arr)[:no_el//2]) #scipy.sparse.csr matrix format to convert 2D array to 1D vector in optimize
		else:
			X0_csr     = sp.csr_matrix(np.abs(X0_arr)[:]) #scipy.sparse.csr matrix format to convert 2D array to 1D vector in optimize
		indices_el = X0_csr.indices
		indptr_el  = X0_csr.indptr
		if "X0_zero" in mode: #set X0 to 0
			X0_el = np.zeros_like(X0_csr.data)
		else:
			X0_el = X0_csr.data
		if real:
			indices_el = np.repeat(indices_el, 2)
			indptr_el  = indptr_el*2
			X0_el      = np.ravel([X0_el,( np.zeros(len(X0_el)) )],'F') #imaginary and real part are alternate, random real part btw -0.5 and 0.5			
		no[side]      = no_el
		X_slice[side] = [i for i in range(nvar, nvar+len(X0_el))]
		indices[side] = indices_el
		indptr[side]  = indptr_el
		X0.append(X0_el)
		nvar += len(X0_el)
	if sym: #Hright = Hleft have to be the same
		X0 = X0[0]
	else:
		X0 = np.concatenate(X0)
	
	### SETUP LINEAR CONSTRAINT FOR HOPPING ELEMENTS

	if "hop" in mode: #When we had hopping elements, we have to make sure that the matrix is still positive definite
					  #Thus, Linear constraint to obtain a diagonally dominant matrix
		Cons = np.zeros((no["right"],len(X0))) #SYM AND NOT TRANSYM AND NOT REAL
		for i in range(no["right"]):
			for p,j in enumerate(indices["right"]):
				if p >= indptr["right"][i] and p < indptr["right"][i+1]: #same row
					if i==j:
						Cons[i,p] = 1 #diagonal element
					else:
						Cons[i,p] = -1 #same row
				elif i==j:
					Cons[i,p] = -1 #same column (symmetrise)
		constraints = sciopt.LinearConstraint(Cons, [eta]*no["right"], [np.inf]*no["right"])
		method = "trust-constr"
		options = {"verbose":True}
		if real:
			bound = [(0,np.inf),(-np.inf,np.inf)]*(len(X0)//2)
		else:
			bound = [(0,np.inf)]*len(X0)
	else:
		if real:
			bound = ([0,-np.inf]*len(X0)//2,[np.inf]*len(X0))
		else:
			bound = ([0]*len(X0),[np.inf]*len(X0))
	
	### SETUP VARIABLE TO CAP MATRIX CONVERSION

	def WCAP_proc(X, sym, transym, real):
		"""Convert the variable vector to CAP matrix"""
		if sym:
			if transym:
				if real:
					WCAP = np.zeros_like(X, dtype=np.complex128)
					WCAP[::2] = -1j*X[::2]
					WCAP[1::2] = -X[1::2]
				else:
					WCAP = -1j*X
				WCAP_arr = np.zeros((no["right"],no["right"]), dtype=np.complex128)
				WCAP_arr[:no["right"]//2] = sp.csr_matrix((WCAP, indices["right"], indptr["right"]), shape=(no["right"]//2, no["right"])).toarray()
				WCAP_arr[no["right"]//2:,no["right"]//2:] = WCAP_arr[:no["right"]//2,:no["right"]//2]
				return cap.symmetrize(WCAP_arr)
			else:
				if real:
					WCAP = np.zeros_like(X, dtype=np.complex128)
					WCAP[::2] = -1j*X[::2]
					WCAP[1::2] = -X[1::2]
					return cap.symmetrize(sp.csr_matrix((WCAP, indices["right"], indptr["right"]), shape=(no["right"], no["right"])).toarray())
				else:
					WCAP = -1j*X
					return cap.symmetrize(sp.csr_matrix((WCAP, indices["right"], indptr["right"]), shape=(no["right"], no["right"])).toarray())
		else:
			if transym:
				WCAP_arr = {}
				for side in ["left", "right"]:
					if real:
						WCAP = np.zeros_like(X[X_slice[side]], dtype=np.complex128)
						WCAP[::2] = -1j*X[X_slice[side]][::2]
						WCAP[1::2] = -X[X_slice[side]][1::2]
					else:
						WCAP = -1j*X[X_slice[side]]
					WCAP_arr[side] = np.zeros((no[side],no[side]), dtype=np.complex128)
					WCAP_arr[side][:no[side]//2] = sp.csr_matrix((WCAP, indices[side], indptr[side]), shape=(no[side]//2, no[side])).toarray()
					WCAP_arr[side][no[side]//2:,no[side]//2:] = WCAP_arr[:no[side]//2,:no[side]//2]
					WCAP_arr[side] = cap.symmetrize(WCAP_arr[side])
				return WCAP_arr
			else:
				if real:
					WCAP_arr = {}
					for side in ["left", "right"]:
						WCAP = np.zeros_like(X[X_slice[side]], dtype=np.complex128)
						WCAP[::2] = -1j*X[X_slice[side]][::2]
						WCAP[1::2] = -X[X_slice[side]][1::2]
						WCAP_arr[side] = cap.symmetrize(sp.csr_matrix((WCAP, indices[side], indptr[side]), shape=(no[side], no[side])).toarray())
					return WCAP_arr
				else:
					WCAP_arr = {}
					for side in ["left", "right"]:
						WCAP = -1j*X[X_slice[side]]
						WCAP_arr[side] = cap.symmetrize(sp.csr_matrix((WCAP, indices[side], indptr[side]), shape=(no[side], no[side])).toarray())
					return WCAP_arr
	
	### SETUP FUNCTION TO MINIMIZE
	
	def obj_fun(X):
		"""Function to optimize"""
		WCAP_arr = WCAP_proc(X, sym, transym, real)
		if sym:
			WCAP_l, WCAP_r =  WCAP_arr, WCAP_arr
		else:
			WCAP_l, WCAP_r =  WCAP_arr["left"], WCAP_arr["right"]
		if "hop" in mode:
			return cap.rmse(cap.transCAP(E, K_, Hd, WCAP_l, WCAP_r, bc_axis, eavg=False, kavg=kavg, eta=eta, refl=refl).real , Tobj)
		else:
			return np.ravel(cap.transCAP(E, K_, Hd, WCAP_l, WCAP_r, bc_axis, eavg=False, kavg=kavg, eta=eta, refl=refl).real - Tobj)
	
	### OPTIMIZATION

	if kproc == "inde":
		kavg = True
		WCAP_arr = []
		error = 0
		Tobj_full = Tobj.copy()
		for j,k in enumerate(K):
			print('k = ',k)
			K_ = np.array([k])
			if "refl" in mode:
				Tobj = Tobj_full[:,:,j]
			else:
				Tobj = Tobj_full[:,j]
			if "hop" in mode:
				res = sciopt.minimize(obj_fun, X0, method=method, options = options, bounds = bound, constraints=constraints, tol=tol)
				error += res.fun
			else:
				res = sciopt.least_squares(obj_fun, X0, bounds = bound, ftol=tol, verbose=2)
				error += (2*res.cost/ne)
			WCAP_arr.append(WCAP_proc(res.x, sym, transym, real))
			if sym: #save at every iteration in case of early termination of the program
					np.save(save_file, WCAP_arr)
			else:
				WCAP_save = []
				for i in range(len(K)):
					WCAP_save.append( [WCAP_arr[i]["left"],WCAP_arr[i]["right"]] )
				np.save(save_file, np.array(WCAP_save))
		error = (error/nk)**0.5
	
	else:
		K_ = K
		if "hop" in mode: #cannot use least squares with constraints
			res = sciopt.minimize(obj_fun, X0, method=method, options = options, bounds = bound, constraints=constraints, tol=tol)
			error = res.fun
		else:
			res = sciopt.least_squares(obj_fun, X0, bounds = bound, ftol=tol, verbose=2)
			error = (2*res.cost/ne/nk)**0.5
		WCAP_arr = WCAP_proc(res.x, sym, transym, real)
	
	### POST-PROCESS RESULT
		
	if sym:
		np.save(save_file, WCAP_arr)
	elif kproc != "inde":
		WCAP_save = []
		for side in ["left","right"]:
			WCAP_save.append( WCAP_arr[side] )
		np.save(save_file, np.array(WCAP_save))
	else:
		WCAP_save = []
		for i in range(len(K)):
			WCAP_save.append( [WCAP_arr[i]["left"],WCAP_arr[i]["right"]] )
		np.save(save_file, np.array(WCAP_save))
	
	### WRITE IN REGISTRE

	df.at[index,'Nvar'] = len(X0)
	df.at[index,'time(min)'] = (time()-start)/60
	df.at[index,'error'] = error
	df.iloc[index:index+1].to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)