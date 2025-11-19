#!/usr/bin/env python
# coding: utf-8

# ## Useful Functions

# In[2]:


# Definition of useful functions.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect

parsec = 3.086e16 
lightyear = 9.461e15
AU = 1.5e11
M_sol = 2e30
I_BAND_ZEROPOINT = 22.0
L_BAND_ZEROPOINT = 18.0

def einstein_radius(ml, dl, ds):  # Returns theta_e (I believe in meters).
    g = 6.6743e-11  # Gravitational constant.
    c = 299792458  # Speed of light in m/s.
    k = 4*g/c**2
    theta_e = (np.sqrt(ml*k*((ds-dl)/(dl*ds))))
    return theta_e

def d_beta(ds, d_perp, theta_e): # in units of Einstein radius.
    del_beta = (d_perp/ds)/theta_e
    return del_beta

def impact_parameter_spitzer(u0_earth, del_beta):
    u0_spitzer = np.array([u0_earth - del_beta, u0_earth + del_beta])
    return u0_spitzer

def u_t(tau, u0):
    u_t = np.sqrt(tau**2 + u0**2)
    return u_t

def d_tau(ds, d_par, theta_e):
    del_tau = (d_par/ds)/theta_e
    return del_tau

def magnification(u):
    A = (u*u + 2)/(u*np.sqrt(u*u + 4))
    return A


# ## Error Functions

# In[ ]:


def Spitzer_flux_error(F):
    """
    Calculates the noise error for Spitzer
    inputs: Flux
    outputs: Noise error y: y = F/sigma
    """
    y = 1.06*F + 24.8*np.sqrt(F) - 16.7
    y = np.maximum(y, 1e-10)  # Ensure y is not less than 1e-10 to avoid division by zero or negative noise
    sigma = F/y

    return sigma


# In[ ]:


def OGLE_flux_error(F):
    """Returns the SNR and its error for a given flux F using the fitted model."""
    # Using the MCMC fit parameters
    from numpy.random import normal
    import numpy as np

    sqrtF = np.sqrt(F)
    mean = [3.116138, -3.341458]
    sig = [0.166425, 3.120069]
    m, b = normal(mean, sig)
    y = m * sqrtF + b
    #print(y)

    sig2 = F/y

    sig = np.sqrt(sig2)

    return sig


# ## Generate Lightcurve

# In[ ]:


from typing import Tuple, Callable, Optional, List

def generate_lightcurve(
                        magnification: Callable[[np.ndarray], np.ndarray],
                        u0: float,
                        t0: float,
                        tE: float,
                        dbbinary: float,
                        dt: float,
                        FS1: float,
                        FS2: float,
                        FB: float,
                        sigfunction: Callable,
                        t: Optional[np.ndarray] = None,
                        trange: Optional[Tuple[float, float]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate lightcurve for given parameters and epochs or date range.

    Parameters
    ----------
    SL_magnification : Callable[[float, float, float, np.ndarray], np.ndarray]
        Single lens magnification function.
    u0 : float
        Impact parameter.
    t0 : float
        Time of closest approach.
    tE : float
        Einstein radius crossing time.
    dbbinary : float
        Binary separation.
    dt : float
        Binary separation.
    FS1 : float
        Primary source flux.
    FS2 : float
        Secondary source flux.
    FB : float
        Blend flux.
    sig : np.ndarray
        Standard deviation of Gaussian errors.
    t : Optional[np.ndarray], optional
        Array of epochs (default is None).
    trange : Optional[Tuple[float, float]], optional
        Date range (default is None).
        - tstart : float, start of date range.
        - tfinish : float, end of date range.

    Returns
    -------
    t_data : np.ndarray
        Generated time points.
    flux_data : np.ndarray
        Generated flux data with noise.
    flux_err_data : np.ndarray
        Gaussian errors.

    Raises
    ------
    ValueError
        If neither epochs nor date range is provided.
        If u0 or t0 or tE are not floats.
    """
    if np.isnan(FB):
        FB = 0.0
    if not isinstance(u0, float):
        print(f'u0 = {u0}')
        raise ValueError('u0 must be a float.')
    if not isinstance(t0, float):
        print(f't0 = {t0}')
        raise ValueError('t0 must be a float.')
    if not isinstance(tE, float):
        print(f'tE = {tE}')
        raise ValueError('tE must be a float.')
    if not isinstance(dbbinary, float):
        print(f'dbbinary = {dbbinary}')
        raise ValueError('dbbinary must be a float.')
    if not isinstance(dt, float):
        print(f'dt = {dt}')
        raise ValueError('dt must be a float.')
    if not isinstance(FS1, float):
        print(f'FS1 = {FS1}')
        raise ValueError('FS1 must be a float.')
    if not isinstance(FS2, float):
        print(f'FS2 = {FS2}')
        raise ValueError('FS2 must be a float.')
    if not isinstance(FB, float):
        print(f'FB = {FB}')
        raise ValueError('FB must be a float.')

    # Generate epochs if not provided
    if t is None:
        if trange is None:
            raise ValueError('Either epochs or date range must be provided.')
        tstart, tfinish = trange
        days = np.arange(np.floor(tstart), np.ceil(tfinish)) # Generate integer days within the range
        t = []
        for day in days:
            if tstart <= day <= tfinish:
                # Generate epochs for the night
                num_epochs = np.random.randint(0, 9)  # Random number of epochs between 0 and 8
                epochs = np.random.choice(np.arange(0.3, 0.6, 0.0104167), size=num_epochs, replace=False)
                t.extend(day + epochs)
    t = np.array(t)

    # Generate the trajectory
    tau = (t0 - t)/tE
    ut = u_t(tau, u0) # primary source trajectory

    u0_binary = u0 + dbbinary
    t0_binary = t0 + dt
    tau_binary = (t0_binary - t)/tE
    ut_binary = u_t(tau_binary, u0_binary)  # binary source trajectory

    # Generate theoretical flux

    A1 = magnification(ut)
    A2 = magnification(ut_binary)

    flux_theoretical = A1*FS1 + A2*FS2 + FB

    sig = sigfunction(flux_theoretical)

    # Generate Gaussian noise
    noise = np.random.normal(0, sig, size=t.shape)

    # Generate noisy flux data
    flux_data = flux_theoretical + noise

    # Gaussian errors
    flux_err_data = np.full(t.shape, sig)

    return t, flux_data, flux_err_data


# In[ ]:


def binary_timeshift(d_binary, ds, phi_binary, er):
  """
  Calcuate the difference in peak times between the primary and secondary sources.

  Inputs
  ------
  d_binary: distance between the two sources in AU
  ds: distance to the primary source in parsecs
  phi_binary: angle of binary separation in degrees
  er: Einstein radius in meters

  Returns
  -------
  dt_binary: difference in peak times in seconds
  """
  beta_binary = (d_binary/ds)/er
  dt_binary = beta_binary*np.cos(phi_binary)

  return dt_binary


# In[ ]:


try:
    import MulensModel as mm
except ModuleNotFoundError:
    get_ipython().system('pip install --user MulensModel')
    import MulensModel as mm


# In[ ]:


finite_source_methods = [
    # Uniform source
    'finite_source_uniform_Gould94',               # 0, 10E-3 < rho < 1 (has a bug)
    'finite_source_uniform_Gould94_direct',        # 1, 10E-3 < rho < 1
    'finite_source_uniform_WittMao94',             # 2, rho < 0.01
    'finite_source_uniform_Lee09',                 # 3, rho > 0.01

    # Limb-darkened source
    'finite_source_LD_WittMao94',                  # 4, rho < 0.01
    'finite_source_LD_Yoo04',                      # 5, 10E-3 < rho < 1
    'finite_source_LD_Yoo04_direct',               # 6, 10E-3 < rho < 1
    'finite_source_LD_Lee09'                       # 7, rho > 0.01
]

# objective function
def binary_source_chi2(theta: np.ndarray,
                       model1: mm.Model,
                       model2: mm.Model,
                       data: List,
                       verbose: Optional[bool] = False,
                       return_fluxes: Optional[bool] = False,
                       FS1 = None,
                       FS2 = None,
                       FB = None
                       ) -> float:
    """
    chi2 function for a binary-source, single-lens, microlensing model.

    Parameters
    ----------
    theta : np.ndarray
        Array of model parameters being fit.
    model1 : mm.Model
        Primary source model.
    model2 : mm.Model
        Secondary source model.
    data : list
        List of data arrays.
    verbose : bool, optional
        Default is False.
        Print the primary source flux, secondary source flux, and blend flux.
    FS1 : float, optional
        Primary source flux. If provided, will be used in the model.
    FS2 : float, optional
        Secondary source flux. If provided, will be used in the model.
    FB : float, optional
        Blend flux. If provided, will be used in the model.

    Returns
    -------
    float
        The chi2 value.

    Notes
    -----
    The model parameters are unpacked from theta and set to the model1 and model2 parameters.
    model1 and model2 are MulensModel.Model objects; see MulensModel documentation
    (https://rpoleski.github.io/MulensModel/) for more information.
    """
    # Unpack the data
    t, flux, flux_err = data

    # Model parameters being fit
    labels = ['t_0', 'u_0', 't_E']

    # Change the values of model.parameters to those in theta
    theta1 = theta[:3]
    theta2 = theta[3:]
    for (label, value1, value2) in zip(labels, theta1, theta2):
        setattr(model1.parameters, label, value1)
        setattr(model2.parameters, label, value2)

    # Calculate the model magnification for each source
    A1 = model1.get_magnification(t)
    A2 = model2.get_magnification(t)

    # Determine which fluxes need to be fitted
    fit_FS1 = FS1 is None
    fit_FS2 = FS2 is None
    fit_FB = FB is None

    # Create lists of known and unknown flux components
    known_fluxes = {}
    unknown_components = []

    if not fit_FS1:
        known_fluxes['FS1'] = FS1
    else:
        unknown_components.append('FS1')

    if not fit_FS2:
        known_fluxes['FS2'] = FS2
    else:
        unknown_components.append('FS2')

    if not fit_FB:
        known_fluxes['FB'] = FB
    else:
        unknown_components.append('FB')

    # If all fluxes are provided, no fitting needed
    if not unknown_components:
        model_flux = A1 * FS1 + A2 * FS2 + FB
        chi2 = np.sum(((flux - model_flux) / flux_err)**2)

        if return_fluxes:
            return chi2, FS1, FS2, FB
        else:
            return chi2

    # Calculate residual flux after subtracting known components
    residual_flux = flux.copy()
    if 'FS1' in known_fluxes:
        residual_flux -= known_fluxes['FS1'] * A1
    if 'FS2' in known_fluxes:
        residual_flux -= known_fluxes['FS2'] * A2
    if 'FB' in known_fluxes:
        residual_flux -= known_fluxes['FB']

    # Build design matrix and RHS vector for unknown components
    n_unknown = len(unknown_components)
    design_matrix = np.zeros((n_unknown, n_unknown))
    rhs_vector = np.zeros(n_unknown)

    # Weights for weighted least squares
    weights = flux_err**-2

    for i, comp_i in enumerate(unknown_components):
        # Get the coefficient vector for component i
        if comp_i == 'FS1':
            coeff_i = A1
        elif comp_i == 'FS2':
            coeff_i = A2
        else:  # comp_i == 'FB'
            coeff_i = np.ones_like(flux)

        # Right-hand side vector
        rhs_vector[i] = np.sum(coeff_i * residual_flux * weights)

        for j, comp_j in enumerate(unknown_components):
            # Get the coefficient vector for component j
            if comp_j == 'FS1':
                coeff_j = A1
            elif comp_j == 'FS2':
                coeff_j = A2
            else:  # comp_j == 'FB'
                coeff_j = np.ones_like(flux)

            # Design matrix element
            design_matrix[i, j] = np.sum(coeff_i * coeff_j * weights)

    # Solve the linear system
    try:
        fitted_values = np.linalg.solve(design_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        # Handle singular matrix: return a large chi2 value
        return 1e16

    # Assign the fitted values back to the appropriate variables
    result_fluxes = {'FS1': FS1, 'FS2': FS2, 'FB': FB}
    for i, component in enumerate(unknown_components):
        result_fluxes[component] = fitted_values[i]

    FS1_final = result_fluxes['FS1']
    FS2_final = result_fluxes['FS2']
    FB_final = result_fluxes['FB']

    # Print the flux parameters
    if verbose:
        print(f"Primary source flux: {FS1_final}")
        print(f"Secondary source flux: {FS2_final}")
        print(f"Blend flux: {FB_final}")

    # Calculate the model flux and chi2
    model_flux = A1 * FS1_final + A2 * FS2_final + FB_final
    chi2 = np.sum(((flux - model_flux) / flux_err)**2)

    # In case something goes wrong with the linear algebra
    if np.isnan(chi2) or np.isinf(chi2):
        print(f"NaN or inf encountered in chi2 calculation: theta={theta}, chi2={chi2}")
        return 1e16

    if return_fluxes:
        return chi2, FS1_final, FS2_final, FB_final
    else:
        return chi2


# In[ ]:


# making up event parameters and data stats
u0 = 0.4321
t0 = 7892.123
tE = 23.4
dbbinary = 0.5
dt = 0.5

# Checking the binary_source functions
FS1 = 70.0
FS2 = 50.0
FB = 10.0
theta = np.array([t0, u0, tE, t0+dt, u0+dbbinary, tE])  # initial guess for parameters

trange = (t0-82, t0+70)

# generating fake Earth data
t_data, flux_data, flux_err_data = generate_lightcurve(magnification, u0, t0, tE, dbbinary, dt, FS1, FS2, FB, OGLE_flux_error, trange=trange)


earth_data = [t_data, flux_data, flux_err_data]

# MulensModel Objects
pspl_primary = mm.Model({'t_0': t0, 'u_0': u0, 't_E': tE})
pspl_secondary = mm.Model({'t_0': t0+dt, 'u_0': u0+dbbinary, 't_E': tE})

# initial chi2 value (will print the fluxes)
chi2, FS1_fitted, FS2_fitted, FB_fitted = binary_source_chi2(theta, pspl_primary, pspl_secondary, earth_data, verbose=True, return_fluxes=True)

print(f"chi^2 = {chi2}")
print(f"n = {len(t_data)}")
print(f"ndof = {len(t_data) - len(theta)}")
print(f"reduced chi^2 = {chi2 / (len(t_data) - len(theta))}")
print(f"Primary source flux true: {FS1}")
print(f"Secondary source flux true: {FS2}")
print(f"Blend flux true: {FB}")

def percent_check(true_value, fitted_value, percentage_threshold=10.0):
    """
    Check if the fitted value is within a certain percentage of the true value.
    """
    return np.abs(fitted_value - true_value) < percentage_threshold / 100 * true_value

# check that the reduced chi2 and flux parameters are within the acceptable range
if not percent_check(chi2 / (len(t_data) - len(theta)), 1.0):
    raise ValueError("Reduced chi^2 is not within the acceptable range.")

if not percent_check(FS1, FS1_fitted):
    raise ValueError("Primary source flux is not within the acceptable range.")

if not percent_check(FS2, FS2_fitted):
    raise ValueError("Secondary source flux is not within the acceptable range.")

if not percent_check(FB, FB_fitted, percentage_threshold=15.0):
    raise ValueError("Blend flux is not within the acceptable range.")


# If the flux values are similar and reduced $\chi^2$ is near 1, then the binary_source_chi2 seems to be working.

# ## Fitting Functioms

# In[ ]:


from scipy.optimize import minimize

def flat_line_fit(data):
  """
  Fit a flat line to the data and return the chi2 value and best-fit flux. The fit is
  initialized by the mean flux in the data array

  Parameters
  ----------
  data : tuple
      A tuple containing the time, flux, and flux error arrays.

  Returns
  -------
  chi2 : float
      The chi-squared value of the best-fit model.
  best_flux : float
      The best-fit flux level of the flat line.
  """
  t, flux, flux_err = data

  def chi2_flat(F, data):
    """
    Chi2 function for a flat line model.

    Parameters
    ----------
    F : float
        The flux level of the flat line.
    data : tuple
        A tuple containing the time, flux, and flux error arrays.

    Returns
    -------
    chi2 : float
        The chi-squared value of the model.
    """
    t, flux, flux_err = data
    if F < 0:
        penalty = ((F / 50)**2)
    else:
        penalty = 0
    chi2 = np.sum(((flux - F) / flux_err)**2)

    return chi2

  chi2_initial = chi2_flat(np.mean(flux), data)

  # Minimize using Nelder-Mead
  result = minimize(chi2_flat, np.mean(flux), args=(data,), method='Nelder-Mead')

  # Calculate best chi2
  F_best = result.x[0]
  chi2 = chi2_flat(F_best, data)

  return chi2, F_best


# In[ ]:


def single_source_fit(data, theta0, t0_ref=0.0):
  """
  Fit a single-lens, single-source model to the data.

  Parameters
  ----------
  data : tuple
      A tuple containing the time, flux, and flux error arrays.
  theta0 : list
      Initial guess for the model parameters [u0, t0, tE, FS, FB].
  t0_ref : float
      Reference time for the model.

  Returns
  -------
  chi2 : float
      The chi-squared value of the best-fit model.
  best_params : list
      The best-fit model parameters [u0, t0, tE, FS, FB].
  """
  t, flux, flux_err = data

  def chi2_single(theta, data, t0_ref):
    """
    Chi2 function for a single-lens, single-source model.

    Parameters
    ----------
    theta : list
        Model parameters [u0, t0, tE, FS, FB].
    data : tuple
        A tuple containing the time, flux, and flux error arrays.
    t0_ref : float
        Reference time for the model.

    Returns
    -------
    chi2 : float
        The chi-squared value of the model.
    """
    t, flux, flux_err = data
    u0, t0, tE, FS, FB = theta
    tau = (t0-t)/tE
    ut = u_t(tau, u0)
    A = magnification(ut)
    F = FS*A + FB

    if FB < 0:
        penalty = ((FB / 50)**2)
    else:
        penalty = 0

    penalty += (((t0_ref -t0) / 5)**2)

    chi2 = np.sum(((flux - F) / flux_err)**2) + penalty

    return chi2

  chi2_initial = chi2_single(theta0, data, t0_ref)

  # Minimize using Nelder-Mead
  result = minimize(chi2_single, theta0, args=(data, t0_ref,), method='Nelder-Mead')

  # Calculate best chi2
  u0_best, t0_best, tE_best, FS_best, FB_best = result.x
  chi2 = chi2_single([u0_best, t0_best, tE_best, FS_best, FB_best], data, t0_ref)

  return chi2, [u0_best, t0_best, tE_best, FS_best, FB_best]


# ## Spitzer lc Detectable

# In[ ]:


# remake spitzer_detectable with fits
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def Spitzer_lc_detectable(
    ml, dl, ds, u0_earth, phis, phi_binary, d_binary,
    tE,
    FS1_I, FS2_I, FB_I,
    FS1_L, FS2_L, FB_L,
    event_id
    ):

    """

    Tags a cell Spitzer-only detectable if a source is only microlensed as seen
    from Spitzer.

    Inputs:
    ml: mass of the lens in solar masses
    ds: distance to the primary source in parsecs
    dl: distance to the lens in parsecs
    u0_earth: u0 of the event as seen from Earth in units of Einstein radius
    phis: angle between Earth and Spitzer in degrees
    phi_binary: angle of binary separation in degrees
    d_binary: distance between the two sources in AU
    tE: Einstein radius crossing time (days)
    FS1_I: flux of the primary source seen from Earth
    FS2_I: flux of the secondary source seen from Earth
    FB_I: flux of the blend seen from Earth
    FS1_L: flux of the primary source seen from Spitzer
    FS2_L: flux of the secondary source seen from Spitzer
    FB_L: flux of the blend seen from Spitzer
    event_id: unique identifier for the event

    Returns whether or not the secondary source appears lensed from Spitzer only.

    Raises:
    ValueError: If any of the input parameters are not of the correct type.
    """
    # type checks
    #------------
    float_params = {"ml": ml, "dl": dl, "ds": ds, "u0_earth": u0_earth,
                    "phi_binary": phi_binary, "d_binary": d_binary, "tE": tE,
                    "FS1_I": FS1_I, FS2_I: FS2_I, FB_I: FB_I,
                    "FS1_L": FS1_L, FS2_L: FS2_L, FB_L: FB_L}
    for param, value in float_params.items():
        if not (np.isscalar(value) and np.isreal(value)):
            print(f'{param} = {value}')
            raise ValueError(f'{param} must be a real scalar number (i.e., a float).')
        
    #check that event_id is a string and not empty
    if not isinstance(event_id, str) or event_id.strip() == "":
        print(f'event_id = {event_id}')
        raise ValueError('event_id must be a non-empty string.')

    # print(f'phis = {phis}')
    if not isinstance(phis, np.ndarray):
        raise ValueError('phis must be a numpy array.')

    # constants
    #------------
    t0 = 0.0
    parsec = 3.086e16
    lightyear = 9.461e15
    AU = 1.5e11
    M_sol = 2e30
    g = 6.6743e-11  # Gravitational constant
    c = 299792458  # Speed of light in m/s
    k = 4*g/(c**2)

    # unit conversions to SI units
    #-----------------------------
    ml_si = ml*M_sol #converts to kg
    ds_si = ds*parsec #converts to m
    dl_si = dl*parsec
    d_binary_si = d_binary*AU #converts to m
    phi_binary_rad = phi_binary * np.pi / 180 #converts to radians
    phis_rad = phis * np.pi / 180

    # event scaling (Einstein radius)
    #--------------
    er = np.sqrt(ml_si*k*((ds_si-dl_si)/(dl_si*ds_si))) # in meters...
    # print(ml, k, ds, dl)
    # 4.0000000000000004e+29 2.9704641076474662e-27 1.543e+20 2.6231e+20

    # initializing storage parameters
    #--------------------------------
    is_Spitzer_only_detectable = np.zeros((len(phis), 5))
    letters_array = []
    import pandas as pd
    params = pd.DataFrame(columns=["event_id",
        "unique_id",
        "F_best",
        "F_best_Spitzer",
        "best1_u0", "best1_t0", "best1_tE", "best1_fs", "best1_fb",
        "best2_u0", "best2_t0", "best2_tE", "best2_fs", "best2_fb",
        "best1Spitzer_u0", "best1Spitzer_t0", "best1Spitzer_tE", "best1Spitzer_fs", "best1Spitzer_fb",
        "best2Spitzer_u0", "best2Spitzer_t0", "best2Spitzer_tE", "best2Spitzer_fs", "best2Spitzer_fb",
        "chi2_earth_flat",
        "chi2_earth_binary",
        "chi2_earth_single1",
        "chi2_earth_single2",
        "chi2_spitzer_flat",
        "chi2_spitzer_binary1",
        "chi2_spitzer_single1",
        "chi2_spitzer_single2",
        "phi", "ml", "dl", "ds", "u0_earth", "phi_binary", "d_binary", "tE", "FS1_I", "FS2_I", "FB_I", "FS1_L", "FS2_L", "FB_L",
        "name", "u0_earth_s2", "u0_spitzer_s1", "u0_spitzer_s2"
    ])

    # Loop over each angle
    for i, phi_rad in enumerate(phis_rad):
        unique_id = f"{event_id}_{i}"
        d_perp = np.sin(phi_rad) *1*AU
        # print(d_perp)

        #1500000000.0
        #1500000000.0 [2.57667914e+20] [4.14135297e-09] [0.00140569]
        #dt = [1.60383934e-09]

        #d_par = np.cos(phi) *1.*AU

        db = (d_perp/ds_si)/er
        # print(d_perp, ds, er, db)
        # 106066017177.98212 1.543e+20 nan nan
        u0_spitzer_s1 = u0_earth + db

        beta_binary = (d_binary_si/ds_si)/er
        db_binary = beta_binary*np.sin(phi_binary_rad)
        #print(db_binary)

        u0_earth_s1 = u0_earth
        u0_earth_s2 = u0_earth_s1 + db_binary
        u0_spitzer_s2 = u0_spitzer_s1 + db_binary

        # Calculate t0
        # db = d_beta(ds*parsec, d_binary*AU, er)
        # dt = d_tau(ds*parsec, d_binary*AU, er)*tE
        # t0_spitzer = t0 + dt

        t02_earth = 0.0 + binary_timeshift(d_binary, ds, phi_binary, er)
        t0_spitzer_s1 = t0 + binary_timeshift(1*AU, ds, phis[i], er) # Assuming 1 AU distance for Spitzer from Earth
        t02_spitzer_s2 = t0_spitzer_s1 + binary_timeshift(d_binary, ds, phi_binary, er)

        # Generate the fake data
        trange = (-75.0, 75.0)
        sig_I = 10.5 # get a better number for this
        # global earth_data
        earth_data = generate_lightcurve(
            magnification,
            u0_earth_s1,
            t0,
            tE,
            db_binary,
            t02_earth - t0,
            FS1_I,
            FS2_I,
            FB_I,
            OGLE_flux_error,
            trange=trange
        )
        # global spitzer_data
        spitzer_data = generate_lightcurve(
            magnification,
            u0_spitzer_s1,
            t0_spitzer_s1,
            tE,
            db_binary,
            t02_spitzer_s2 - t0_spitzer_s1,
            FS1_L,
            FS2_L,
            FB_L,
            Spitzer_flux_error,
            trange=trange
        )
        if np.isnan(FB_L):
            FB_L = 0.0
        if np.isnan(FB_I):
            FB_I = 0.0

        name = f"_{unique_id}.npy"
        if not os.path.exists("data/lightcurves"):
            os.makedirs("data/lightcurves")
        np.save(f"data/lightcurves/earth_data{name}", earth_data)
        np.save(f"data/lightcurves/spitzer_data{name}", spitzer_data)

        # print(u0_spitzer_s1, t0_spitzer_s1, tE, db_binary, t02_spitzer_s2 - t0_spitzer_s1, FS1_L, FS2_L, FB_L, sig_I)

        # MulensModels
        pspl_earth = mm.Model({'t_0': t0, 'u_0': u0_earth_s1, 't_E': tE})
        pspl_spitzer1 = mm.Model({'t_0': t0_spitzer_s1, 'u_0': u0_spitzer_s1, 't_E': tE})
        pspl_earth_binary = mm.Model({'t_0': t02_earth, 'u_0': u0_earth_s2, 't_E': tE})
        pspl_spitzer1_binary = mm.Model({'t_0': t02_spitzer_s2, 'u_0': u0_spitzer_s2, 't_E': tE})

        # Fitting a flat line to check that the models are significantly preferred to it
        #------------------------------------------------------------------------------
        # Detactability criteria of Deltachi2>500
        chi2_earth_flat, F_best = flat_line_fit(earth_data)
        chi2_spitzer_flat, F_best_Spitzer = flat_line_fit(spitzer_data)

        # Fitting single source models to see if the binary-source model is significantly preferred
        #------------------------------------------------------------------------------------------
        # initial guesses for single source fits
        theta_earth1_guess = np.array([u0_earth_s1, t0, tE, FS1_I, FB_I + FS2_I])
        theta_earth2_guess = np.array([u0_earth_s2, t02_earth, tE, FS2_I, FB_I + FS1_I])
        theta_spitzer1_guess = np.array([u0_spitzer_s1, t0_spitzer_s1, tE, FS1_L, FB_L + FS2_L])
        theta_spitzer2_guess = np.array([u0_spitzer_s2, t02_spitzer_s2, tE, FS2_L, FB_L + FS1_L])

        # earth fits starting from each source individualy and constraining the fit to "t0" near the
        # reference t0 for that source and perspective
        chi2_earth_single1, best1 = single_source_fit(earth_data, theta_earth1_guess, t0_ref=t0)
        chi2_earth_single2, best2 = single_source_fit(earth_data, theta_earth2_guess, t0_ref=t02_earth)

        # Spitzer fits starting from each source individually and constraining the fit to "t0" near the
        # reference t0 for that source and perspective
        chi2_spitzer_single1, best1Spitzer = single_source_fit(spitzer_data, theta_spitzer1_guess, t0_ref=t0_spitzer_s1)
        chi2_spitzer_single2, best2Spitzer = single_source_fit(spitzer_data, theta_spitzer2_guess, t0_ref=t02_spitzer_s2)

        # Evaluating chi2 for the true params binary source model
        #---------------------------------------------------------
        theta_earth = np.array([t0, u0_earth_s1, tE, t02_earth, u0_earth_s2, tE])
        theta_spitzer1 = np.array([t0_spitzer_s1, u0_spitzer_s1, tE, t02_spitzer_s2, u0_spitzer_s2, tE])

        # calculating chi2 using true flux parameters
        chi2_earth_binary = binary_source_chi2(theta_earth, pspl_earth, pspl_earth_binary, earth_data, verbose=False, FS1=FS1_I, FS2=FS2_I, FB=FB_I)
        chi2_spitzer_binary1 = binary_source_chi2(theta_spitzer1, pspl_spitzer1, pspl_spitzer1_binary, spitzer_data, verbose=False, FS1=FS1_L, FS2=FS2_L, FB=FB_L)

        # # calculating chi2 using fitted flux parameters
        # chi2_earth_binary = binary_source_chi2(theta_earth, pspl_earth, pspl_earth_binary, earth_data, verbose=False)
        # chi2_spitzer_binary1 = binary_source_chi2(theta_spitzer1, pspl_spitzer1, pspl_spitzer1_binary, spitzer_data, verbose=False)

        # Storing the parameters in a dataframe
        params_new_dict = {
            "event_id": event_id,
            "unique_id": unique_id,
            "F_best": F_best,
            "F_best_Spitzer": F_best_Spitzer,
            "best1_u0": best1[0],
            "best1_t0": best1[1],
            "best1_tE": best1[2],
            "best1_fs": best1[3],
            "best1_fb": best1[4],
            "best2_u0": best2[0],
            "best2_t0": best2[1],
            "best2_tE": best2[2],
            "best2_fs": best2[3],
            "best2_fb": best2[4],
            "best1Spitzer_u0": best1Spitzer[0],
            "best1Spitzer_t0": best1Spitzer[1],
            "best1Spitzer_tE": best1Spitzer[2],
            "best1Spitzer_fs": best1Spitzer[3],
            "best1Spitzer_fb": best1Spitzer[4],
            "best2Spitzer_u0": best2Spitzer[0],
            "best2Spitzer_t0": best2Spitzer[1],
            "best2Spitzer_tE": best2Spitzer[2],
            "best2Spitzer_fs": best2Spitzer[3],
            "best2Spitzer_fb": best2Spitzer[4],
            "chi2_earth_flat": chi2_earth_flat,
            "chi2_earth_binary": chi2_earth_binary,
            "chi2_earth_single1": chi2_earth_single1,
            "chi2_earth_single2": chi2_earth_single2,
            "chi2_spitzer_flat": chi2_spitzer_flat,
            "chi2_spitzer_binary1": chi2_spitzer_binary1,
            "chi2_spitzer_single1": chi2_spitzer_single1,
            "chi2_spitzer_single2": chi2_spitzer_single2,
            "phi": phis[i],
            "ml": ml,
            "dl": dl,
            "ds": ds,
            "u0_earth": u0_earth,
            "phi_binary": phi_binary,
            "d_binary": d_binary,
            "tE": tE,
            "FS1_I": FS1_I,
            "FS2_I": FS2_I,
            "FB_I": FB_I,
            "FS1_L": FS1_L,
            "FS2_L": FS2_L,
            "FB_L": FB_L,
            "name": name,
            "u0_earth_s2": u0_earth_s2,
            "u0_spitzer_s1": u0_spitzer_s1,
            "u0_spitzer_s2": u0_spitzer_s2
        }

        # Append the new parameters to the full DataFrame
        params.loc[len(params)] = params_new_dict

        # TODO: check for solution swapping <= It shouln't happen now that I have added a sigma=5days t0_ref prior
        # print(f"chi^2 comparison {i}: {chi2_earth_flat}, {chi2_earth_binary}, {chi2_earth_single1}, {chi2_earth_single2}")
        # print(f"Spitzer comparison {i}: {chi2_spitzer_flat}, {chi2_spitzer_binary1}, {chi2_spitzer_single1}, {chi2_spitzer_single2}")

        # Position key:
        # 0: nothing is detected, 1: S1 is detected from Earth, 2: S2 is detected from Earth,
        # 3: S1 is detected from Spitzer, 4: S2 is detected from Spitzer
        # Value key:
        # [0: false/not detectable, 1: true/detectable]

        if ((chi2_earth_flat + chi2_spitzer_flat) - (chi2_earth_binary + chi2_spitzer_binary1)) < 500:
                                                                     # if the chi2 difference between the flat-line model of the truth
                                                                     # is less than 500, the detection is not statistically significant
            is_Spitzer_only_detectable[i][0] = 1  # nothing is detected
        else:                                                        # if the chi2 difference *is* bigger than 500, detection is significant
            is_Spitzer_only_detectable[i][0] = 0  # something is detected

            if (chi2_earth_single1 - chi2_earth_binary) < 160:       # if the chi2 difference between the primary-only, single-lens model and
                                                                     # the truth is less than 160, the detection of the secondary is not significant
                is_Spitzer_only_detectable[i][1] = 1  # S1 is detected from Earth
                is_Spitzer_only_detectable[i][2] = 0  # S2 is not detected from Earth

            elif (chi2_earth_single2 - chi2_earth_binary) < 160:     # if the chi2 difference between the secondary-only, single-lens model and
                                                                     # the truth is less than 160, the detection of the secondary is not significant
                is_Spitzer_only_detectable[i][1] = 0  # S1 is not detected from Earth
                is_Spitzer_only_detectable[i][2] = 1  # S2 is detected from Earth

            else:                                                    # if the chi2 differences between both single-lens models and the truth
                                                                     # are greater than 160, the detection of BOTH sources is significant
                is_Spitzer_only_detectable[i][1] = 1  # S1 is detected from Earth
                is_Spitzer_only_detectable[i][2] = 1  # S2 is detected from Earth

            if (chi2_spitzer_single1 - chi2_spitzer_binary1) < 160:  # if the chi2 difference between the primary-only, single-lens model and
                                                                     # the truth is less than 160, the detection of the secondary is not significant
                is_Spitzer_only_detectable[i][3] = 1  # S1 is detected from Spitzer
                is_Spitzer_only_detectable[i][4] = 0  # S2 is not detected from Spitzer

            elif (chi2_spitzer_single2 - chi2_spitzer_binary1) < 160:  # if the chi2 difference between the secondary-only, single-lens model and
                                                                     # the truth is less than 160, the detection of the secondary is not significant
                is_Spitzer_only_detectable[i][3] = 0  # S1 is not detected from Spitzer
                is_Spitzer_only_detectable[i][4] = 1  # S2 is detected from Spitzer

            else:                                                    # if the chi2 differences between both single-lens models and the truth
                                                                     # are greater than 160, the detection of BOTH sources is significant
                is_Spitzer_only_detectable[i][3] = 1  # S1 is detected from Spitzer
                is_Spitzer_only_detectable[i][4] = 1  # S2 is detected from Spitzer
        # Key:
        # aceg (blue) = S1 detectable from Earth and Spitzer, S2 detectable from Spitzer only.
        # adfh (red) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth only.
        # adeh (green) = S1 detectable from Earth but not Spitzer, S2 not detectable.
        # adfg (yellow) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth and Spitzer.
        # adeg (magenta) = S1 detectable from Earth but not Spitzer, S2 detectable from Spitzer only.
        # acfh (cyan) = S1 detectable from Earth and Spitzer, S2 detectable from Earth only.
        # aceh (orange) = S1 detectable from Earth and Spitzer, S2 not detectable.
        # acfg (purple) = S1 detectable from Earth and Spitzer, S2 detectable from Earth and Spitzer.

        # These are irrelevant, right? I'm putting them in the dictionary anyway just in case.
        # bdfh = S1 not detectable from earth or Spitzer, S2 detectable from earth only.
        # bdeh = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
        # bdfg = S1 not detectable from earth or Spitzer, S2 detectable from Earth and Spitzer.
        # bdeg = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
        # bcfh = S1 not detectable from earth but detectable from Spitzer, S2 detectable from earth only.
        # bceh = S1 not detectable from earth but detectable from Spitzer, S2 not detectable.
        # bceg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Spitzer only.
        # bcfg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Earth and Spitzer.

        # 1: S1 from Earth, 2: S2 Earth, 3: S1 Spitzer, 4: S2 Spitzer
        letters = ""

        if is_Spitzer_only_detectable[i][1] == 1:
            letters = "a"
        else:
            letters = "b"
        if is_Spitzer_only_detectable[i][3] == 1:
            letters += "c"
        else:
            letters += "d"
        if is_Spitzer_only_detectable[i][2] == 0:  # this one is reversed and we are keeping it like that for backwards compatibility
            letters += "e"
        else:
            letters += "f"
        if is_Spitzer_only_detectable[i][4] == 1:
            letters += "g"
        else:
            letters += "h"

        letters_array.append(letters)

        # plot the lightcurves in magnitudes for easier comparison across bands
        plt.close(100)
        plt.figure(figsize=(10, 5), num=100)
        earth_mag = flux_to_mag(earth_data[1], I_BAND_ZEROPOINT)
        earth_mag_err = flux_err_to_mag_err(earth_data[1], earth_data[2])
        spitzer_mag = flux_to_mag(spitzer_data[1], L_BAND_ZEROPOINT)
        spitzer_mag_err = flux_err_to_mag_err(spitzer_data[1], spitzer_data[2])
        plt.errorbar(earth_data[0], earth_mag, yerr=earth_mag_err, label='Earth Data', fmt='x', color='black')
        plt.errorbar(spitzer_data[0], spitzer_mag, yerr=spitzer_mag_err, label='Spitzer Data', fmt='o', color='blue')

        # plot the fitted models
        t = np.linspace(-75, 75, 1000)
        tau = (best1[1]-t)/best1[2]
        ut = u_t(tau, best1[0])
        s1_E_model = magnification(ut)
        s1_E_model = best1[3]*s1_E_model + best1[4]
        s1_E_model_mag = flux_to_mag(s1_E_model, I_BAND_ZEROPOINT)
        label = 'Earth S1 Model'
        if is_Spitzer_only_detectable[i][1]:
            label += ":)"
        else:
            label+= ":("
        plt.plot(t, s1_E_model_mag, label=label, color='black', alpha=0.5)

        tau = (best2[1]-t)/best2[2]
        ut = u_t(tau, best2[0])
        s2_E_model = magnification(ut)
        s2_E_model = best2[3]*s2_E_model + best2[4]
        s2_E_model_mag = flux_to_mag(s2_E_model, I_BAND_ZEROPOINT)
        label = 'Earth S2 Model'
        if is_Spitzer_only_detectable[i][2]:
            label += ":)"
        else:
            label+= ":("
        plt.plot(t, s2_E_model_mag, label=label, color='black', alpha=0.5)

        tau = (best1Spitzer[1]-t)/best1Spitzer[2]
        ut = u_t(tau, best1Spitzer[0])
        s1_S_model = magnification(ut)
        s1_S_model = best1Spitzer[3]*s1_S_model + best1Spitzer[4]
        s1_S_model_mag = flux_to_mag(s1_S_model, L_BAND_ZEROPOINT)
        label = 'Spitzer S1 Model'
        if is_Spitzer_only_detectable[i][3]:
            label += ":)"
        else:
            label+= ":("
        plt.plot(t, s1_S_model_mag, label=label, color='blue', alpha=0.5)

        tau = (best2Spitzer[1]-t)/best2Spitzer[2]
        ut = u_t(tau, best2Spitzer[0])
        s2_S_model = magnification(ut)
        s2_S_model = best2Spitzer[3]*magnification(ut) + best2Spitzer[4]# Corrected calculation
        s2_S_model_mag = flux_to_mag(s2_S_model, L_BAND_ZEROPOINT)
        label = 'Spitzer S2 Model'
        if is_Spitzer_only_detectable[i][4]:
            label += ":)"
        else:
            label+= ":("
        plt.plot(t, s2_S_model_mag, label=label, color='cyan', alpha=0.5)

        # plot the truth models
        pspl_earth, pspl_earth_binary = mm.Model({'t_0': t0, 'u_0': u0_earth_s1, 't_E': tE}), mm.Model({'t_0': t02_earth, 'u_0': u0_earth_s2, 't_E': tE})
        # F = FS1_I * A1 + FS2_I * A2 + FB_I
        A1 = pspl_earth.get_magnification(t)
        A2 = pspl_earth_binary.get_magnification(t)
        F = FS1_I * A1 + FS2_I * A2 + FB_I
        plt.plot(t, flux_to_mag(F, I_BAND_ZEROPOINT), label='Earth Truth Model', color='red', alpha=0.5)

        pspl_spitzer1, pspl_spitzer1_binary = mm.Model({'t_0': t0_spitzer_s1, 'u_0': u0_spitzer_s1, 't_E': tE}), mm.Model({'t_0': t02_spitzer_s2, 'u_0': u0_spitzer_s2, 't_E': tE})
        # F = FS1_L * A1 + FS2_L * A2 + FB_L
        A1 = pspl_spitzer1.get_magnification(t)
        A2 = pspl_spitzer1_binary.get_magnification(t)
        F = FS1_L * A1 + FS2_L * A2 + FB_L
        plt.plot(t, flux_to_mag(F, L_BAND_ZEROPOINT), label='Spitzer Truth Model', color='blue', alpha=0.5)

        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude (mag)')
        plt.gca().invert_yaxis()
        plt.title(f'Lightcurves for {name[1:-4]} - Class: {letters}')
        plt.legend()
        plt.savefig(f"data/lightcurves/lc{name}.png")   

    return letters_array, params #Spitzer_detectable_index

#isod = Spitzer_lc_detectable(
#    0.2, 5000, 8500, 0.25, np.linspace(0,360, 360), 45, 1,
#    30.0,
#    10.0, 15.0, 10.0,
#    10.0, 15.0, 10.0
#    )
#print(isod)

# ml, dl, ds, u0_earth, phi, phi_binary, d_binary,
    # tE,
    # FS1_I, FS2_I, FB_I,
    # FS1_L, FS2_L, FB_L

# Key:
# aceg (blue) = S1 detectable from Earth and Spitzer, S2 detectable from Spitzer only.
# adfh (red) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth only.
# adeh (green) = S1 detectable from Earth but not Spitzer, S2 not detectable.
# adfg (yellow) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth and Spitzer.
# adeg (magenta) = S1 detectable from Earth but not Spitzer, S2 detectable from Spitzer only.
# acfh (cyan) = S1 detectable from Earth and Spitzer, S2 detectable from Earth only.
# aceh (orange) = S1 detectable from Earth and Spitzer, S2 not detectable.
# acfg (purple) = S1 detectable from Earth and Spitzer, S2 detectable from Earth and Spitzer.

# These are irrelevant, right? I'm putting them in the dictionary anyway just in case.
# bdfh = S1 not detectable from earth or Spitzer, S2 detectable from earth only.
# bdeh = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bdfg = S1 not detectable from earth or Spitzer, S2 detectable from Earth and Spitzer.
# bdeg = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bcfh = S1 not detectable from earth but detectable from Spitzer, S2 detectable from earth only.
# bceh = S1 not detectable from earth but detectable from Spitzer, S2 not detectable.
# bceg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Spitzer only.
# bcfg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Earth and Spitzer.


#n = 1 # number of samples you want

#Ds = np.sqrt(np.random.uniform(500**2, 8500**2, n)) # in parsecs
#Dl = np.sqrt(np.random.uniform(2000**2, 5000**2, n)) # also in parsecs
#dbinary_array = np.log(np.random.lognormal(40, 1.5, n)) # in AU
#u0_array = np.linspace(-1, 1, 201) # units of einstein radius
#phi_array = np.linspace(0, 180, 181)

#sd = Spitzer_detectable(1.*M_sol, Dl, Ds, u0_array, phi_array, -90, dbinary_array*AU)
#print(sd)


# In[ ]:


# Returns a string of letters for whether or not the secondary source is detectable
# from Spitzer alone.

import numpy as np
import matplotlib.pyplot as plt
import math

def Spitzer_detectable(
    ml,
    dl,
    ds,
    u0_earth,
    phi,
    phi_binary,
    d_binary,
    event_id=None,  # optional; retained for API compatibility
):
    """

    Tags a cell Spitzer-only detectable if a source is only microlensed as seen
    from Spitzer.

    Inputs:
    ml: mass of the lens in solar masses
    ds: distance to the primary source in parsecs
    dl: distance to the lens in parsecs
    u0_earth: u0 of the event as seen from Earth in units of Einstein radius
    phi: angle between Earth and Spitzer in degrees
    phi_binary: angle of binary separation in degrees
    d_binary: distance between the two sources in AU

    Returns whether or not the secondary source appears lensed from Spitzer only.

    """

    parsec = 3.086e16
    AU = 1.5e11
    M_sol = 2e30
    g = 6.6743e-11
    c = 299792458
    k = 4 * g / (c ** 2)

    ml = np.asarray(ml) * M_sol
    ds = np.asarray(ds) * parsec
    dl = np.asarray(dl) * parsec
    d_binary = np.asarray(d_binary) * AU

    phi_binary = np.asarray(phi_binary) * np.pi / 180.0
    phi = np.asarray(phi) * np.pi / 180.0

    er = np.sqrt(ml * k * ((ds - dl) / (dl * ds)))

    d_perp = np.sin(phi) * 1.0 * AU
    db = (d_perp / ds) / er
    u0_spitzer = u0_earth + db

    beta_binary = (d_binary / ds) / er
    db_binary = beta_binary * np.sin(phi_binary)

    u0_earth_s1 = u0_earth
    u0_earth_s2 = u0_earth_s1 + db_binary
    u0_spitzer_s1 = u0_spitzer
    u0_spitzer_s2 = u0_spitzer_s1 + db_binary

    cond1 = np.where(np.abs(u0_earth_s1) < 1, "a", "b")
    cond2 = np.where(np.abs(u0_spitzer_s1) < 1, "c", "d")
    cond3 = np.where(np.abs(u0_earth_s2) >= 1, "e", "f")
    cond4 = np.where(np.abs(u0_spitzer_s2) < 1, "g", "h")

    all_scalars = (
        np.isscalar(u0_earth_s1)
        and np.isscalar(u0_earth_s2)
        and np.isscalar(u0_spitzer_s1)
        and np.isscalar(u0_spitzer_s2)
    )

    if all_scalars:
        return str(cond1.item()) + str(cond2.item()) + str(cond3.item()) + str(cond4.item())

    return np.char.add(np.char.add(np.char.add(cond1, cond2), cond3), cond4)

# Key:
# aceg (blue) = S1 detectable from Earth and Spitzer, S2 detectable from Spitzer only.
# adfh (red) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth only.
# adeh (green) = S1 detectable from Earth but not Spitzer, S2 not detectable.
# adfg (yellow) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth and Spitzer.
# adeg (magenta) = S1 detectable from Earth but not Spitzer, S2 detectable from Spitzer only.
# acfh (cyan) = S1 detectable from Earth and Spitzer, S2 detectable from Earth only.
# aceh (orange) = S1 detectable from Earth and Spitzer, S2 not detectable.
# acfg (purple) = S1 detectable from Earth and Spitzer, S2 detectable from Earth and Spitzer.

# These are irrelevant, right? I'm putting them in the dictionary anyway just in case.
# bdfh = S1 not detectable from earth or Spitzer, S2 detectable from earth only.
# bdeh = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bdfg = S1 not detectable from earth or Spitzer, S2 detectable from Earth and Spitzer.
# bdeg = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bcfh = S1 not detectable from earth but detectable from Spitzer, S2 detectable from earth only.
# bceh = S1 not detectable from earth but detectable from Spitzer, S2 not detectable.
# bceg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Spitzer only.
# bcfg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Earth and Spitzer.


#n = 1 # number of samples you want

#Ds = np.sqrt(np.random.uniform(500**2, 8500**2, n)) # in parsecs
#Dl = np.sqrt(np.random.uniform(2000**2, 5000**2, n)) # also in parsecs
#dbinary_array = np.log(np.random.lognormal(40, 1.5, n)) # in AU
#u0_array = np.linspace(-1, 1, 201) # units of einstein radius
#phi_array = np.linspace(0, 180, 181)

#sd = Spitzer_detectable(1.*M_sol, Dl, Ds, u0_array, phi_array, -90, dbinary_array*AU)
#print(sd)


# ## Get Table

# In[ ]:


# Load the table
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
full_table = pd.read_csv("data/popsycle_table_with_calcs.csv")
A_max = (full_table["u0"]**2+2)/(abs(full_table["u0"])*np.sqrt(full_table["u0"]**2+4))
#print(table)
# new_table = table[np.where(table["I_S"] - 2.5 * np.log10(A_max) < 17)[0]]

# new_table = table[table["I_S"] - 2.5 * np.log10(A_max) < 17]
# print(new_table.index)
# print(new_table.shape)

# indices = new_table.index
# index = table.index
# print(index[5])
# print(indices[5])

# Print the first few rows of the table
#print(new_table.head())
#print(new_table.shape)

# D_L = new_table['D_L_kpc']
#I_S = new_table['I_S']
#print(I_S)

#print(D_L)
#M_L = new_table['M_L']
#print(sum(M_L)/len(M_L))


# In[ ]:


# filter for magnified primary source L-band of brighter than 17.5 mag
table = full_table[(full_table["L_S"] - 2.5 * np.log10(A_max) < 17.5) | (full_table["L_S2"] - 2.5 * np.log10(A_max) < 17.5)]
table = table.reset_index(drop=True)
print(table.shape)


# In[1]:


table.head()


# In[ ]:


# Loop through the data using the new Spitzer_lc_detectable function.
#Spitzer_lc_detectable(ml, dl, ds, u0_earth, phi, phi_binary, d_binary, tE, FS1_I, FS2_I, FB_I, FS1_L, FS2_L, FB_L)
"""
Tags a cell Spitzer-only detectable if a source is only microlensed as seen
from Spitzer.

Inputs:
ml: mass of the lens in solar masses
ds: distance to the primary source in parsecs
dl: distance to the lens in parsecs
u0_earth: u0 of the event as seen from Earth in units of Einstein radius
phi: angle between Earth and Spitzer in degrees
phi_binary: angle of binary separation in degrees
d_binary: distance between the two sources in AU
tE: Einstein radius crossing time (days)
FS1_I: flux of the primary source seen from Earth
FS2_I: flux of the secondary source seen from Earth
FB_I: flux of the blend seen from Earth
FS1_L: flux of the primary source seen from Spitzer
FS2_L: flux of the secondary source seen from Spitzer
FB_L: flux of the blend seen from Spitzer

Returns whether or not the secondary source appears lensed from Spitzer only.
"""
mu_rel_mas_days = table['mu_rel_mas_yr']/365.25
table['tE'] = table['theta_E_mas']

# for the new table (no distance modulus needed)
def mag_to_flux(m, zp):
    return 10**((zp-m)/2.5)

def flux_to_mag(F, zp):
    """Convert flux to magnitude using the provided zero point."""
    safe_flux = np.clip(np.asarray(F, dtype=float), 1e-20, None)
    return zp - 2.5 * np.log10(safe_flux) 

def flux_err_to_mag_err(F, F_err):
    """Propagate flux uncertainties into magnitude uncertainties."""
    safe_flux = np.clip(np.asarray(F, dtype=float), 1e-20, None)
    return (2.5 / np.log(10)) * (np.asarray(F_err, dtype=float) / safe_flux)

# Initialize empty DataFrame for parameters
# params_array = pd.DataFrame()
# real_data_array_lc = []

# for i in range(30): # Looping through the first row for testing
#     # print(type(table["D_L_kpc"][i]*1000))
#     real_data_lc, params = Spitzer_lc_detectable(
#         ml = table["M_L"][i],
#         dl = table["D_L_kpc"][i]*1000,
#         ds = table["D_S_kpc"][i]*1000,
#         u0_earth = table["u0"][i],
#         phis = np.linspace(0,359,2),
#         phi_binary = table["binary_phi_deg"][i],
#         d_binary = table["binary_sep_au"][i],
#         tE = table['tE'][i],
#         FS1_I = mag_to_flux(table["I_S"][i]),
#         FS2_I = mag_to_flux(table["I_S2"][i]),
#         FB_I = mag_to_flux(table["I_L"][i]),
#         FS1_L = mag_to_flux(table["L_S"][i]),
#         FS2_L = mag_to_flux(table["L_S2"][i]),
#         FB_L = mag_to_flux(table["L_L"][i])
#     )
#     # print(f"real_data_lc{i}: {type(real_data_lc)}")
#     # print(f"real_data_array_lc{i}: {type(real_data_array_lc)}")
#     real_data_array_lc.append(real_data_lc)
#     params_array = pd.concat([params_array, params], ignore_index=True)
#     # print(params_array.shape)

# real_data_array_lc = np.array(real_data_array_lc)
# print(real_data_array_lc)
# params_array.to_csv('output.csv', index=False)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# params_array.head()


# ## First 10 Loop with u0-Based Detectability

# In[ ]:


# Prepare the viewing angles once so every event uses the same grid.
phi_grid = np.linspace(0, 359, 360, dtype=float)

# Collect every (event, phi) result in a tabular structure.
records = []
spitzer_sig = inspect.signature(Spitzer_detectable)
has_event_id = "event_id" in spitzer_sig.parameters

for row in table.itertuples(index=False):
    event_id = str(row.event_id)

    # Build the argument bundle for Spitzer_detectable; include event_id if supported.
    call_args = dict(
        ml=np.array([row.M_L]),
        dl=np.array([row.D_L_kpc * 1000.0]),
        ds=np.array([row.D_S_kpc * 1000.0]),
        u0_earth=np.array([row.u0]),
        phi=phi_grid,
        phi_binary=np.array([row.binary_phi_deg]),
        d_binary=np.array([row.binary_sep_au]),
    )
    if has_event_id:
        call_args["event_id"] = event_id

    # Run the legacy detectability logic and flatten to one (phi) dimension.
    real_data = np.asarray(Spitzer_detectable(**call_args)).reshape(-1)

    for phi_idx, label in enumerate(real_data):
        records.append(
            {
                "unique_id": f"{event_id}_{phi_idx}",
                "event_id": event_id,
                "phi_index": phi_idx,
                "phi_deg": phi_grid[phi_idx],
                "detection_label": str(label),
            }
        )

# Long-format table with an explicit unique identifier for every event/phi pair.
real_data_df = pd.DataFrame.from_records(records)

# Optional: wide representation (events × phis) retained for downstream array-based code.
real_data_wide = real_data_df.pivot(index="event_id", columns="phi_index", values="detection_label")
real_data_array = real_data_wide.to_numpy()
n_samples, n_phis = real_data_array.shape


# In[ ]:


import gc
import inspect
import numpy as np
import pandas as pd
from math import ceil

# Shared φ grid (matches the light-curve run).
phi_grid = np.linspace(0, 359, 60, dtype=float)
chunk_size = 100

records = []
spitzer_sig = inspect.signature(Spitzer_detectable)
has_event_id = "event_id" in spitzer_sig.parameters

num_rows = len(table)
num_chunks = ceil(num_rows / chunk_size)

for chunk_idx in range(num_chunks):
    start = chunk_idx * chunk_size
    stop = min(start + chunk_size, num_rows)
    chunk = table.iloc[start:stop]

    for row in chunk.itertuples(index=False):
        event_id = str(row.event_id)

        call_args = dict(
            ml=np.array([row.M_L]),
            dl=np.array([row.D_L_kpc * 1000.0]),
            ds=np.array([row.D_S_kpc * 1000.0]),
            u0_earth=np.array([row.u0]),
            phi=phi_grid,
            phi_binary=np.array([row.binary_phi_deg]),
            d_binary=np.array([row.binary_sep_au]),
            event_id=event_id,
        )

        real_data = np.asarray(Spitzer_detectable(**call_args)).reshape(-1)

        for phi_idx, label in enumerate(real_data):
            records.append(
                {
                    "unique_id": f"{event_id}_{phi_idx}",
                    "event_id": event_id,
                    "phi_index": phi_idx,
                    "phi_deg": phi_grid[phi_idx],
                    "detection_label": str(label),
                }
            )

    gc.collect()

real_data_df = pd.DataFrame.from_records(records)
real_data_wide = real_data_df.pivot(index="event_id", columns="phi_index", values="detection_label")
real_data_array = real_data_wide.to_numpy()
n_samples, n_phis = real_data_array.shape


# ## Dying cell - Loop with lc-Based Detectability

# In[ ]:


import gc
import numpy as np
import pandas as pd
from math import ceil

phi_grid = np.linspace(0, 359, 60, dtype=float)
chunk_size = 100

all_letters = []
id_records = []

num_rows = len(table)
num_chunks = ceil(num_rows / chunk_size)

for chunk_idx in range(num_chunks):
    start = chunk_idx * chunk_size
    stop = min(start + chunk_size, num_rows)
    chunk = table.iloc[start:stop]

    for row in chunk.itertuples(index=False):
        event_id = str(row.event_id)
        letters_chunk, _ = Spitzer_lc_detectable(
            ml=row.M_L,
            dl=row.D_L_kpc * 1000.0,
            ds=row.D_S_kpc * 1000.0,
            u0_earth=row.u0,
            phis=phi_grid,
            phi_binary=row.binary_phi_deg,
            d_binary=row.binary_sep_au,
            tE=row.theta_E_mas,
            FS1_I=mag_to_flux(row.I_S, I_BAND_ZEROPOINT),
            FS2_I=mag_to_flux(row.I_S2, I_BAND_ZEROPOINT),
            FB_I=mag_to_flux(row.I_L, I_BAND_ZEROPOINT),
            FS1_L=mag_to_flux(row.L_S, L_BAND_ZEROPOINT),
            FS2_L=mag_to_flux(row.L_S2, L_BAND_ZEROPOINT),
            FB_L=mag_to_flux(row.L_L, L_BAND_ZEROPOINT),
            event_id=event_id,
        )

        all_letters.extend(letters_chunk)
        id_records.extend(
            {
                "unique_id": f"{event_id}_{phi_idx}",
                "event_id": event_id,
                "phi_index": phi_idx,
            }
            for phi_idx, _ in enumerate(letters_chunk)
        )

    gc.collect()

all_letters_array = np.array(all_letters)
unique_id_df = pd.DataFrame(id_records)


# ## Graph all the data to compare with u0-based detectability

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Shared code → integer mapping
detectable_codes = {
    "aceg": 1, "adfh": 2, "adeh": 3, "adfg": 4, "adeg": 5,
    "acfh": 6, "aceh": 7, "acfg": 8, "bdfh": 9, "bdeh": 10,
    "bdfg": 11, "bdeg": 12, "bcfh": 13, "bceh": 14,
    "bceg": 15, "bcfg": 16,
}

vectorized_map = np.vectorize(detectable_codes.get)

# Light-curve results (filled bars)
lc_codes = vectorized_map(all_letters)
counts_lc = np.bincount(lc_codes.ravel(), minlength=17)[1:]

# u₀-threshold results (outline overlay)
spitzer_labels = real_data_array.ravel()
spitzer_codes = vectorized_map(spitzer_labels)
counts_u0 = np.bincount(spitzer_codes, minlength=17)[1:]

categories = [
    "S1: B\nS2: S", "S1: E\nS2: E", "S1: E\nS2: N", "S1: E\nS2: B",
    "S1: E\nS2: S", "S1: B\nS2: E", "S1: B\nS2: N", "S1: B\nS2: B",
]
colors = np.array([
    (46/255, 37/255, 133/255), (126/255, 41/255, 84/255),
    (52/255, 116/255, 56/255), (220/255, 204/255, 125/255),
    (93/255, 168/255, 153/255), (147/255, 203/255, 235/255),
    (194/255, 106/255, 119/255), (159/255, 74/255, 150/255),
])

legend_filled = [
    mpatches.Patch(color=colors[i], alpha=0.7, label=label)
    for i, label in enumerate([
        "S1: both, S2: Spitzer only",
        "S1: Earth only, S2: Earth only",
        "S1: Earth only, S2: neither",
        "S1: Earth only, S2: both",
        "S1: Earth only, S2: Spitzer only",
        "S1: both, S2: Earth only",
        "S1: both, S2: neither",
        "S1: both, S2: both",
    ])
]
legend_outline = mpatches.Patch(
    facecolor="none",
    edgecolor="black",
    linewidth=1.3,
    label="u₀ threshold (outline only)",
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(categories, counts_lc[:8], color=colors, alpha=0.7, label="Light-curve fit")
ax.bar(
    categories,
    counts_u0[:8],
    facecolor="none",
    edgecolor="black",
    linewidth=1.3,
    label="u₀ threshold",
)
ax.set_xlabel("Scenarios")
ax.set_ylabel("Frequency")
ax.legend(handles=[*legend_filled, legend_outline], loc="upper left")
plt.tight_layout()
plt.savefig("bargraph_overlay.png", bbox_inches="tight")
plt.show()

total_lc = lc_codes.size
print("Light-curve percentages:")
for i, color_name in enumerate(["Blue", "Red", "Green", "Yellow", "Teal", "Cyan", "Orange", "Purple"], 1):
    pct = counts_lc[i - 1] / total_lc * 100
    print(f"{color_name}: {pct:.4f}")

print((counts_lc[0] / (counts_lc[0] + counts_lc[7] + counts_lc[6]) * 100) * 0.4051)
print(lc_codes.shape)
print(np.sum(counts_lc[:8]))
print(counts_lc[0] + counts_lc[6] + counts_lc[7])
print(np.sum(counts_lc[:8]) / total_lc)
for i, color_name in enumerate(["blue", "red", "green", "yellow", "teal", "cyan", "orange", "purple"], 1):
    print(f"{color_name} number:", counts_lc[i - 1] / total_lc)


# In[ ]:


n_samples = 1
n_phis = 2

for i in range(n_samples):
   for j in range(n_phis):
      print(f"{i}: {real_data_array[i][j]}, {real_data_array_lc[i][j]}")
      # print chi2 values
      row = n_phis*i + j
      df_row = params_array.iloc[row]
      print(f"    delta chi2: {df_row['chi2_earth_flat'] - df_row['chi2_earth_binary']}")
      print(f"              : {df_row['chi2_spitzer_flat'] - df_row['chi2_spitzer_binary1']}")
      # print u0 values
      print(f"            u0: {df_row['u0_earth']}, {df_row['u0_earth_s2']}")
      print(f"              : {df_row['u0_spitzer_s1']}, {df_row['u0_spitzer_s2']}")

# Key:
# aceg (blue) = S1 detectable from Earth and Spitzer, S2 detectable from Spitzer only.
# adfh (red) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth only.
# adeh (green) = S1 detectable from Earth but not Spitzer, S2 not detectable.
# adfg (yellow) = S1 detectable from Earth but not Spitzer, S2 detectable from Earth and Spitzer.
# adeg (magenta) = S1 detectable from Earth but not Spitzer, S2 detectable from Spitzer only.
# acfh (cyan) = S1 detectable from Earth and Spitzer, S2 detectable from Earth only.
# aceh (orange) = S1 detectable from Earth and Spitzer, S2 not detectable.
# acfg (purple) = S1 detectable from Earth and Spitzer, S2 detectable from Earth and Spitzer.

# These are irrelevant, right? I'm putting them in the dictionary anyway just in case.
# bdfh = S1 not detectable from earth or Spitzer, S2 detectable from earth only.
# bdeh = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bdfg = S1 not detectable from earth or Spitzer, S2 detectable from Earth and Spitzer.
# bdeg = S1 not detectable from earth or Spitzer, S2 detectable from Spitzer only.
# bcfh = S1 not detectable from earth but detectable from Spitzer, S2 detectable from earth only.
# bceh = S1 not detectable from earth but detectable from Spitzer, S2 not detectable.
# bceg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Spitzer only.
# bcfg = S1 not detectable from earth but detectable from Spitzer, S2 detectable from Earth and Spitzer.


# In[ ]:


params = np.load("params_array.npy")
print(params.shape)


# In[ ]:


ptest = params_array[1]
print(ptest)
print(f"Spitzer data: {spitzer_data}\n")
print(f"Earth data: {earth_data}\n")


# In[ ]:


# Plotting the distribution of different numbers/colors for the real data (in the same manner as the simulation above).

# Initialize the array.
real_data_array = np.array(Spitzer_lc_detectable(ml=np.array([table["M_L"][indices[0]]]),
                                   dl=np.array([table["D_L_kpc"][indices[0]]*1000]),
                                   ds=np.array([table["D_S_kpc"][indices[0]]*1000]),
                                   u0_earth=np.array([table["u0"][indices[0]]]),
                                   phi=np.linspace(0,360,2), # Set phi to 2 points to match the expected dimension
                                   phi_binary=np.array(table["binary_phi_deg"][indices[0]]),
                                   d_binary=np.array([table["binary_sep_au"][indices[0]]])))

# Run the data.
for i in indices[1:]:
    real_data = Spitzer_detectable(ml=np.array([table["M_L"][i]]),
                                   dl=np.array([table["D_L_kpc"][i]*1000]),
                                   ds=np.array([table["D_S_kpc"][i]*1000]),
                                   u0_earth=np.array([table["u0"][i]]),
                                   phi=np.linspace(0,360,2), # Set phi to 2 points to match the expected dimension
                                   phi_binary=np.array(table["binary_phi_deg"][i]),
                                   d_binary=np.array([table["binary_sep_au"][i]]))
    real_data_array = np.vstack((real_data_array, real_data))

# Dictionary to assign each output to a number.
Spitzer_detectable_outputs = {"aceg":1, "adfh":2, "adeh":3, "adfg":4, "adeg":5,
                              "acfh":6, "aceh":7, "acfg":8, "bdfh":9, "bdeh":10,
                              "bdfg":11, "bdeg":12, "bcfh":13, "bceh":14,
                              "bceg":15, "bcfg":16}

for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])):
        real_data_array[i,j] = Spitzer_detectable_outputs[real_data_array[i,j]]

real_data_array = real_data_array.astype(int)
#Initialize the array.
"""
real_data_array = np.array(Spitzer_detectable_new(ml=np.array([table["M_L"][0]]),
                                   dl=np.array([table["D_L_kpc"][0]*1000]), ds=np.array([table["D_S_kpc"][0]*1000]),
                                   u0_spitzer=np.array([table["u0"][0]]), phi=np.linspace(0,360,360), phi_binary = -90,
                                   d_binary=np.array([table["binary_sep_au"][0]])))

# Run the data.
for i in range(1, len(table)):
    real_data = Spitzer_detectable_new(ml=np.array([table["M_L"][i]]),
                                   dl=np.array([table["D_L_kpc"][i]*1000]), ds=np.array([table["D_S_kpc"][i]*1000]),
                                   u0_spitzer=np.array([table["u0"][i]]), phi=np.linspace(0,360,360), phi_binary = -90,
                                   d_binary=np.array([table["binary_sep_au"][i]]))
    real_data_array = np.vstack((real_data_array, real_data))
    """


# Count the values for each color.
test_count1 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 1:
            test_count1 += 1

test_count2 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 2:
            test_count2 += 1

test_count3 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 3:
            test_count3 += 1

test_count4 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 4:
            test_count4 += 1

test_count5 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 5:
            test_count5 += 1

test_count6 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 6:
            test_count6 += 1

test_count7 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 7:
            test_count7 += 1

test_count8 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 8:
            test_count8 += 1

test_count9 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 9:
            test_count9 += 1

test_count10 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 10:
            test_count10 += 1

test_count11 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 11:
            test_count11 += 1

test_count12 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 12:
            test_count12 += 1

test_count13 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 13:
            test_count13 += 1

test_count14 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 14:
            test_count14 += 1

test_count15 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 15:
            test_count15 += 1

test_count16 = 0
for i in range(len(real_data_array)):
    for j in range(len(real_data_array[0])): # Use the actual size of the second dimension
        if real_data_array[i,j] == 16:
            test_count16 += 1

print(100*test_count1/(len(real_data_array)*len(real_data_array[0]))) # Use actual size for division

# Make the plot.
categories = ["blue", "red", "green", "yellow", "magenta", "cyan",
                   "orange", "purple"]#, "darkblue", "darkred", "darkgreen", "gold",
                   #"darkmagenta", "darkcyan", "darkorange", "black"]
counts = [test_count1, test_count2, test_count3, test_count4, test_count5,
          test_count6, test_count7, test_count8]#, test_count9, test_count10,
          #test_count11, test_count12, test_count13, test_count14, test_count15,
          #test_count16]
colors = np.array(["blue", "red", "green", "yellow", "magenta", "cyan",
                   "orange", "purple"])#, "darkblue", "darkred", "darkgreen", "gold",
                   #"darkmagenta", "darkcyan", "darkorange", "black"])
# for the legend
blue_patch = mpatches.Patch(color='blue', label="S1 detectable from Spitzer, S2 detectable from Spitzer only.")
red_patch = mpatches.Patch(color='red', label="S1 not detectable from Spitzer, S2 detectable from Earth only.")
green_patch = mpatches.Patch(color='green', label="S1 not detectable from Spitzer, S2 not detectable.")
yellow_patch = mpatches.Patch(color='yellow', label="S1 not detectable from Spitzer, S2 detectable from Earth and Spitzer.")
magenta_patch = mpatches.Patch(color='magenta', label="S1 not detectable from Spitzer, S2 detectable from Spitzer only.")
cyan_patch = mpatches.Patch(color='cyan', label="S1 detectable from Spitzer, S2 detectable from Earth only.")
orange_patch = mpatches.Patch(color='orange', label="S1 detectable from Spitzer, S2 not detectable.")
purple_patch = mpatches.Patch(color='purple', label="S1 detectable from Spitzer, S2 detectable from Earth and Spitzer.")

# labeldict = {"blue":"S2 detectable from Spitzer only.",
             # "red":"S2 detectable from Earth only.",
             # "green":"S2 not detectable.",
             # "yellow":"S2 detectable from Earth and Spitzer.",
             # "magenta":"S2 detectable from Spitzer only.",
             # "cyan":"S2 detectable from Earth only.",
             # "orange":"S2 not detectable.",
             # "purple":"S2 detectable from Earth and Spitzer."}

# Plot the graph.
plt.bar(categories, counts, color=colors)#, label=labeldict[categories])
plt.xlabel('Scenarios')
plt.ylabel('Frequency')
plt.legend(handles=[blue_patch, red_patch, green_patch, yellow_patch, magenta_patch, cyan_patch,
                   orange_patch, purple_patch], bbox_to_anchor=(0.5, -0.15), loc="upper center")
# Adjust the plot's right margin to make space for the legend
#plt.subplots_adjust(right=0.7)

plt.savefig("bargraph.png", bbox_inches='tight')
plt.show()


print((test_count1/(test_count1 + test_count8 + test_count7)*100)*0.4051) # cut
print(real_data_array.shape)

#print(len(real_data_array))
#print(real_data_array.shape)
