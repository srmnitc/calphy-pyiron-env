import pandas as pd
import os
import numpy as np
from calphy.integrators import kb
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

api_key = "dWQ1z2YGDjKOpcAiJkeZtIG5D1lEq1YP"

def extract_fe(df, lattice, temperature, 
               fitorder=3, 
               concentration_limits=[0, 0.5], 
               concentration_steps=10000, 
               plot=False,
               add_entropy=None):
    
    if add_entropy is None:
        if lattice in ['lqd', 'liquid']:
            add_entropy=False
        else:
            add_entropy=True

    df_f = df[(df['lattice']==lattice) & (df['temperature']==temperature)]
    fit = np.polyfit(df_f['concentration'], df_f['fe'], fitorder)
    concfine = np.linspace(concentration_limits[0], concentration_limits[1], concentration_steps)
    fitted = np.polyval(fit, concfine)
    
    if add_entropy:
        def _ent(c):
            return -kb*(c*np.log(c) + (1-c)*np.log(1-c))
        smix = np.where(concfine != 0, _ent(concfine), 0)
        fitted = fitted-temperature*smix

    if plot:
        plt.plot(df_f['concentration'], df_f['fe'], 'o', label=f'{lattice}-calc')
        plt.plot(concfine, fitted, label=f'{lattice}-fit')
        plt.xlabel('concentration')
        plt.ylabel('FE')
        plt.legend(frameon=False)
    return concfine, fitted, fit

def create_dataframe(pr):
    df = pr.job_table()
    df = df[df['status'] == 'finished']
    fes = []
    for j in df.job:
        fes.append(pr[f'{j}/output/energy_free'])
    df["fe"] = fes
    df = df.drop(['id', 'status', 'subjob', 'projectpath', 'project', 'timestart', 'timestop', 'totalcputime', 'computer',
           'hamilton', 'hamversion', 'parentid', 'masterid'], axis=1)
    temps = []
    concs = []
    structs =[]
    for name in df.job:
        raw = name.split('_')
        temps.append(int(raw[-1]))
        structs.append(raw[0])
        concs.append(float(raw[1].replace('d', '.')))
    df['temperature'] = temps
    df['concentration'] = concs
    df['lattice'] = structs
    return df
    
def replace_atom(structure, new_species, reps=5, to_replace=0):
    structure[np.random.permutation(len(structure))[:to_replace]] = new_species
    return structure

def replace_atoms(structure, species, number_of_atoms):
    #copy structure
    print(structure.get_chemical_formula())
    new_structure = structure.copy()
    #get indicies of replaceable atom
    symbols = new_structure.get_chemical_symbols()
    #atom which are not the required ones
    indices = [count for count, sym in enumerate(symbols) if sym != species]
    #now find random from these indices
    np.random.shuffle(indices)
    selected_indices = indices[:number_of_atoms]
    #now replace
    new_structure[selected_indices] = species
    print(new_structure.get_chemical_formula())
    return new_structure

def create_structures(pr, comp, repetitions=4):
    """
    Create off stoichiometric structures
    
    Parameters
    ----------
    pr: pyiron project
        
    comp: required composition, float
    
    repetitions: int
        required super cell size
        
    """
    structure_fcc = pr.create.structure.ase.bulk('Al', cubic=True, a=4.135).repeat(repetitions)
    n_li = int(comp*len(structure_fcc))
    structure_fcc[np.random.permutation(len(structure_fcc))[:n_li]] = 'Li'
    
    structure_b32 = pr.create.structure.ase.read('AlLi_poscar', format='vasp')
    n_li = int((0.5-comp)*len(structure_b32))
    rinds = len(structure_b32)//2 + np.random.choice(range(len(structure_b32)//2), n_li, replace=False)
    structure_b32[rinds] = 'Al'
    return structure_fcc, structure_b32
    
    
def fe_at(p, temp, threshold=1E-1):
    """
    Get the free energy at a given temperature
    
    Parameters
    ----------
    p: pyiron Job
        Pyiron job with calculated free energy and temperature
        
    temp: float
        Required temperature
        
    threshold: optional, default 1E-1
        Minimum difference needed between required temperature and temperature found in pyiron job
        
    Returns
    -------
    float: free energy value at required temperature
    """
    arg = np.argsort(np.abs(p.output.temperature-temp))[0]
    th = np.abs(p.output.temperature-temp)[arg] 
    if th > threshold:
        raise ValueError("not a close match, threshold %f"%th)
    return p.output.energy_free[arg]

def normalise_curves(fe_arrs, t_arr):
    ref_arr = fe_arrs[0]
    norm, m, c = normalise_fe(ref_arr, t_arr)
    norms = []
    norms.append(norm)
    for fe_arr in fe_arrs[1:]:
        norms.append(fe_arr-(m*t_arr+c))
    return norms
    
def normalise_fe(fe_arr, t_arr):
    """
    Get the enthalpy of mixing by fitting and subtracting a straight line connecting the end points.
    
    Parameters
    ----------
    fe_arr: list of floats
        array of free energy values as function of composition
        
    conc_arr: list of floats
        array of composition values
    
    Returns
    -------
    norm: list of floats
        normalised free energy
    
    m: float
        slope of the fitted line
    
    c: float
        intercept of the fitted line
    """
    m = (fe_arr[-1]-fe_arr[0])/(t_arr[-1]-t_arr[0])
    c = fe_arr[-1]-m*(t_arr[-1]-t_arr[0])
    norm = fe_arr-(m*t_arr+c)
    return norm, m, c

def find_common_tangent(conc, fe1, fe2, guess_range=None, fit_order=3, lower_bound=None, upper_bound=None, ):
    f1 = np.polyfit(conc, fe1, fit_order)
    f2 = np.polyfit(conc, fe2, fit_order)
    df1 = np.polyder(f1)
    df2 = np.polyder(f2)
    
    if guess_range is None:
        guess_range = [min(conc), max(conc)]
        
    def _eqns(x):
        _f1 = lambda x: np.polyval(f1, x)
        _f2 = lambda x: np.polyval(f2, x)
        _df1 = lambda x: np.polyval(df1, x)
        _df2 = lambda x: np.polyval(df2, x)
        x1, x2 = x[0], x[1]
        eq1 = _df1(x1) - _df2(x2)
        eq2 = _df1(x1)*(x1 - x2) - (_f1(x1) - _f2(x2))
        return [eq1, eq2]

    
    if lower_bound is None:
        lower_bound = (min(conc), min(conc))   # lower bounds on x1, x2
    if upper_bound is None:
        upper_bound = (max(conc), max(conc))    # upper bounds
    res = least_squares(_eqns, guess_range, bounds=(lower_bound, upper_bound))
    return res.x, np.array([np.polyval(f1, res.x[0]), np.polyval(f2, res.x[1])]), res.cost