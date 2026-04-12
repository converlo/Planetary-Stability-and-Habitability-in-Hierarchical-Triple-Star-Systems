"""
Numerical framework for orbital stability and radiative habitability
in hierarchical triple star systems hosting a circumbinary planet.

This module provides a set of tools to model the orbital dynamics and
the irradiation environment of a circumbinary planet in a
hierarchical triple stellar system composed of an inner binary and an
outer stellar companion in a coplanar configuration. The dynamical evolution is computed with the
REBOUND N-body simulation library, while the thermal environment is estimated from
the combined stellar flux received across a two-dimensional spatial grid.

The workflow is designed to support exploratory studies of the overlap
between dynamical stability regions and radiative habitable zones in
multiple-star systems. In particular, the code allows the user to:
(1) generate hierarchical triple configurations,
(2) evaluate the long-term survival of circumbinary planetary orbits,
(3) estimate stability fractions over randomized orbital phases,
(4) map stable orbital domains as a function of planetary semi-major axis,
(5) compute time-dependent equilibrium temperature maps,
(6) visualize the joint structure of stability zone (SZ) and habitable
zones (HZ) through an animation.

For visualization purposes, all positions are expressed in a reference
frame centered on the barycenter of the inner binary rather than on the
global barycenter of the triple system. This choice keeps the stability
zone visually fixed in the animation and facilitates the comparison
between orbital dynamics and thermal structure. In addition, the
planetary orbital timescale used for sampling is estimated from the
Keplerian approximation
    P = sqrt(a_p**3 / (m_A + m_B)),
which is adopted because the osculating orbital period returned by
REBOUND may be poorly defined for perturbed circumbinary trajectories.

The present implementation is intended for research-oriented numerical
experiments and figure production for studies of exoplanet stability 
and habitability in multiple-star environments.
"""

import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random as rd

def simulation(parameters, phase, a_p, integrator = "whfast"):
    """
    Build a hierarchical triple-star REBOUND simulation with a circumbinary planet.

    The system consists of:
    - an inner binary composed of stars A and B,
    - an outer stellar companion C orbiting the barycenter of the inner binary,
    - a circumbinary planet p orbiting the same inner-binary barycenter.

    All bodies are initialized from the orbital elements provided in
    ``parameters`` and ``phase``.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the physical and orbital parameters of the system.
        Expected structure is:
            parameters["binary"]["m_A"], ["m_B"], ["a_AB"], ["e_AB"]
            parameters["companion"]["m_C"], ["a_C"], ["e_C"], ["inc_C"]
            parameters["planet"]["m_p"], ["e_p"], ["inc_p"]
        Masses are assumed in solar masses, semi-major axes in AU, and
        angles in radians.
    phase : dict
        Dictionary containing the initial true anomalies and arguments of
        pericenter for the binary, companion, and planet:
            phase["binary"]["f_B"], ["w_B"]
            phase["companion"]["f_C"], ["w_C"]
            phase["planet"]["f_p"], ["w_p"]
    a_p : float
        Semi-major axis of the circumbinary planet, in AU.
    integrator : str, optional
        REBOUND integrator to use. Default is ``"whfast"``.

    Returns
    -------
    sim : rebound.Simulation
        Initialized REBOUND simulation in units of AU, solar mass, and year.
    """
    # Body parameters extraction
    m_A = parameters["binary"]["m_A"]
    m_B = parameters["binary"]["m_B"]
    a_AB = parameters["binary"]["a_AB"]
    e_AB = parameters["binary"]["e_AB"]

    m_C = parameters["companion"]["m_C"]
    a_C = parameters["companion"]["a_C"]
    e_C = parameters["companion"]["e_C"]
    inc_C = parameters["companion"]["inc_C"]

    m_p = parameters["planet"]["m_p"]
    e_p = parameters["planet"]["e_p"]
    inc_p = parameters["planet"]["inc_p"]


    # Phase extraction
    f_B = phase["binary"]["f_B"]
    w_B = phase["binary"]["w_B"]

    f_C = phase["companion"]["f_C"]
    w_C = phase["companion"]["w_C"]

    f_p = phase["planet"]["f_p"]
    w_p = phase["planet"]["w_p"]


    # Simulation initialization
    sim = rebound.Simulation()
    sim.units = ("AU", "msun", "year")
    sim.integrator = integrator

    # Binary
    sim.add(hash="A",m=m_A)
    sim.add(hash="B", m=m_B, e=e_AB, a=a_AB, f=f_B, omega=w_B)

    cdmAB = sim.com(0, 2) # Creation of the barycenter of the inner binary

    # Companion
    sim.add(hash="C", m=m_C, e=e_C, a=a_C, f=f_C, omega=w_C, inc=inc_C, primary=cdmAB)

    # Circumbinary planet
    sim.add(hash="p", m=m_p, e=e_p, a=a_p, f=f_p, omega=w_p, inc=inc_p, primary=cdmAB)

    sim.move_to_com() # Moving to the reference frame of the barycenter

    return sim


def positions(parameters, phase, a_p, integrator = "whfast", i_max=500, n_orbits=10):
    """
    Compute the time-dependent positions of all bodies in the inner-binary frame.

    The simulation is integrated over a time span corresponding to a chosen
    number of planetary orbital periods. At each output time, the positions
    of stars A, B, companion C, and planet p are stored in a frame centered
    on the barycenter of the inner binary.

    Parameters
    ----------
    parameters : dict
        System parameters passed to :func:`simulation`.
    phase : dict
        Initial orbital phases passed to :func:`simulation`.
    a_p : float
        Planetary semi-major axis, in AU.
    integrator : str, optional
        REBOUND integrator. Default is ``"whfast"``.
    i_max : int, optional
        Number of output snapshots. Default is 500.
    n_orbits : float, optional
        Number of planetary orbital periods covered by the integration.
        Default is 10.

    Returns
    -------
    pos : ndarray of shape (i_max, 4, 2)
        Array containing the Cartesian positions of the four bodies in the
        plane of the inner binary:
        ``pos[:, 0, :]`` = star A,
        ``pos[:, 1, :]`` = star B,
        ``pos[:, 2, :]`` = companion C,
        ``pos[:, 3, :]`` = planet p.

    Notes
    -----
    The sampling times are based on the Keplerian estimate

        P = sqrt(a_p**3 / (m_A + m_B))

    rather than on the osculating period returned by REBOUND. This choice is
    more robust for perturbed circumbinary orbits, for which the orbital
    elements may become noisy or poorly defined.

    For visualization purposes, positions are recentered on the barycenter of
    the inner binary at each time step. This keeps the stability zone fixed in
    animated representations.
    """
    # Creation of the hierarchical triple-star simulation with a circumbinary planet
    sim = simulation(parameters, phase, a_p, integrator = integrator)

    # Store particles and parameters of interest
    A = sim.particles["A"]
    B = sim.particles["B"]
    C = sim.particles["C"]
    p = sim.particles["p"]
    
    m_A = parameters["binary"]["m_A"]
    m_B = parameters["binary"]["m_B"]

    # Time step based on 1/20 of the shortest relevant orbital period
    P_B = sim.particles["B"].P
    sim.dt = P_B/20

    # Keplerian estimate of the planetary orbital period around the inner binary
    P_p = np.sqrt(a_p**3 / (m_A + m_B))
    
    times = np.linspace(0, n_orbits*P_p, i_max)
    pos = np.zeros((i_max, 4, 2))

    # Integrate for the n_orbits periods and stored i_max times the positions in the reference frame of the binary
    for i, t in enumerate(times):
        sim.integrate(t)
        
        x_com_AB = (m_A * A.x + m_B * B.x) / (m_A + m_B)
        y_com_AB = (m_A * A.y + m_B * B.y) / (m_A + m_B)

        pos[i, 0] = [A.x - x_com_AB, A.y - y_com_AB]
        pos[i, 1] = [B.x - x_com_AB, B.y - y_com_AB]
        pos[i, 2] = [C.x - x_com_AB, C.y - y_com_AB]
        pos[i, 3] = [p.x - x_com_AB, p.y - y_com_AB]

    return pos


def integration(parameters, phase, a_p, integrator="whfast", n=10000, r=10):
    """
    Test the long-term dynamical stability of a circumbinary planet.

    The system is integrated for a duration corresponding to ``n`` planetary
    orbital periods. The orbit is classified as unstable if the planetary
    eccentricity reaches or exceeds unity, or if the planetary semi-major axis
    exceeds a prescribed multiple of the companion semi-major axis.

    Parameters
    ----------
    parameters : dict
        System parameters passed to :func:`simulation`.
    phase : dict
        Initial orbital phases passed to :func:`simulation`.
    a_p : float
        Planetary semi-major axis, in AU.
    integrator : str, optional
        REBOUND integrator. Default is ``"whfast"``.
    n : int, optional
        Number of planetary orbital periods used as the integration horizon.
        Default is 10000.
    r : float, optional
        Ejection threshold factor. The orbit is considered unstable if
        ``a_planet > r * a_companion``. Default is 10.

    Returns
    -------
    bool
        ``True`` if the orbit survives the full integration interval under the
        adopted criterion, ``False`` otherwise.

    Notes
    -----
    This function implements a pragmatic stability criterion intended for large
    parameter sweeps. It does not distinguish between different classes of
    instability such as collisions or ejections.
    """
    
    sim = simulation(parameters, phase, a_p, integrator=integrator)

    # Store particles and parameters of interests
    C = sim.particles["C"]
    p = sim.particles["p"]
    #print(f"a_p={a_p:.3f}, p.P={p.P}, sim.t={sim.t}")
    
    #if not np.isfinite(p.P) or p.P <= 0:
        #print(f"Orbital period invalid for a_p={a_p:.3f}: P={p.P}")
        #return False
    
    m_A = parameters["binary"]["m_A"]
    m_B = parameters["binary"]["m_B"]
    
    # Keplerian estimate of the planetary orbital period around the inner binary
    P = np.sqrt(a_p**3 / (m_A + m_B))
    t_max = P * n
    #print(f"t_max={t_max}")
    
    # Time step based on 1/20 of the shortest relevant orbital period
    P_B = sim.particles["B"].P
    sim.dt = P_B / 20
    dt = sim.dt

    steps = 0

    while sim.t < t_max:
        # Advance the integration by one fixed time step
        sim.integrate(sim.t + dt)
        steps += 1

        # Osculating orbits relative to the barycenter of the inner binary
        orbite_p = p.orbit(primary=sim.com(0,2))
        orbit_C = C.orbit(primary=sim.com(0,2))
        
        # Instability criterion:
            # - hyperbolic planetary orbit
            # - planetary semi-major axis larger than a prescribed fraction of the companion orbit
        if orbite_p.e >= 1 or orbite_p.a > r * orbit_C.a:
            #print(f"Instable : a_p={a_p:.3f}, t={sim.t:.3f}, steps={steps}, e={orbite_p.e:.3f}, a={orbite_p.a:.3f}")
            return False

    #print(f"Stable jusqu'au bout : a_p={a_p:.3f}, steps={steps}")
    # The orbit is considered stable if it survives the full integration time
    return True


def random_phase():
    """
    Draw a random set of orbital phases for the binary, companion, and planet.

    The true anomaly and argument of pericenter of each orbit are sampled
    independently from a uniform distribution over [0, 2π).

    Returns
    -------
    dict
        Nested dictionary containing randomized values of ``f`` and ``w`` for
        the binary, the companion, and the planet.

    Notes
    -----
    This randomization is used to estimate the fraction of dynamically stable
    realizations associated with a given planetary semi-major axis.
    """
    return {
        "binary": {
            "f_B": rd.uniform(0, 2*np.pi),
            "w_B": rd.uniform(0, 2*np.pi)
        },
        "companion": {
            "f_C": rd.uniform(0, 2*np.pi),
            "w_C": rd.uniform(0, 2*np.pi)
        },
        "planet": {
            "f_p": rd.uniform(0, 2*np.pi),
            "w_p": rd.uniform(0, 2*np.pi)
        }
    }


def stability_fraction(parameters, a_p, integrator="whfast", N=30, n=10000, r=10):
    """
    Estimate the fraction of stable realizations for a given planetary orbit.

    For a fixed planetary semi-major axis, the stability of the system is
    evaluated over ``N`` independent realizations of the initial orbital
    phases. The returned value corresponds to the fraction of realizations
    that remain stable over the specified integration time.

    Parameters
    ----------
    parameters : dict
        System parameters passed to :func:`integration`.
    a_p : float
        Planetary semi-major axis, in AU.
    integrator : str, optional
        REBOUND integrator. Default is ``"whfast"``.
    N : int, optional
        Number of random phase realizations. If ``N == 0``, a single
        zero-phase configuration is evaluated. Default is 30.
    n : int, optional
        Number of planetary orbital periods used in each integration.
        Default is 10000.
    r : float, optional
        Ejection threshold factor passed to :func:`integration`. Default is 10.

    Returns
    -------
    float or int
        Fraction of stable realizations if ``N > 0``. If ``N == 0``, returns
        either 0 or 1 depending on the stability of the zero-phase case.

    Notes
    -----
    This quantity can be interpreted as an empirical stability probability
    with respect to the random distribution of initial orbital phases.
    """
    N_stable = 0
    step = 0
    
    # Reference configuration with all orbital angles set to zero
    if N==0:
      phase_zero = {
            "binary": {"f_B": 0, "w_B": 0},
            "companion": {"f_C": 0, "w_C": 0},
            "planet": {"f_p": 0, "w_p": 0}
          }
      # Evaluate the stability of the zero-phase configuration
      if integration(parameters=parameters, phase=phase_zero, a_p=a_p, 
                     integrator=integrator, n=n, r=r):
            N_stable = 1
      return N_stable
    else :
      for _ in range(N):
          # Draw a random set of initial orbital phases
          phase = random_phase()
          
          # Count the realization if the orbit remains stable
          if integration(parameters=parameters, phase=phase, a_p=a_p, 
                         integrator=integrator, n=n, r=r):
            N_stable += 1
            
          step += 1  
          print(f"step : {step} / {N}")
          
      # Empirical fraction of stable realizations
      return N_stable / N


def stability_zone(parameters, a_p_range, integrator="whfast", N=30, n=10000, r=10, k=25):
    """
    Compute the stability fraction as a function of planetary semi-major axis.

    The function samples ``k`` values of the planetary semi-major axis within
    ``a_p_range`` and evaluates the corresponding stability fraction using
    :func:`stability_fraction`.

    Parameters
    ----------
    parameters : dict
        System parameters.
    a_p_range : sequence of two float
        Lower and upper bounds of the explored planetary semi-major axis range,
        in AU.
    integrator : str, optional
        REBOUND integrator. Default is ``"whfast"``.
    N : int, optional
        Number of random phase realizations per sampled semi-major axis.
        Default is 30.
    n : int, optional
        Number of planetary orbital periods used for each integration.
        Default is 10000.
    r : float, optional
        Ejection threshold factor. Default is 10.
    k : int, optional
        Number of sampled semi-major axis values. Default is 25.

    Returns
    -------
    ndarray of shape (k, 2)
        Array whose first column contains the sampled planetary semi-major axes
        and whose second column contains the corresponding stability fractions.

    Notes
    -----
    This function provides the raw numerical data used to define and visualize
    the circumstellar or circumbinary stability zone.
    """
    
    results = []
    
    # Uniform sampling of the explored semi-major-axis interval
    a_p_list = np.linspace(a_p_range[0], a_p_range[1], k)

    for a_p in a_p_list:
        # Estimate the stability fraction for the current planetary orbit
        p = stability_fraction(parameters=parameters, a_p=a_p, 
                               integrator=integrator, N=N, n=n, r=r)
        results.append((a_p, p))
        print(f"a_p = {a_p}")
        
    # Return the stability scan as a two-column array
    return np.array(results)


def stability_zone_boundary(results, q=0.9):
    """
    Extract contiguous stability intervals from a stability scan.

    A semi-major axis is considered part of the stability zone if its stability
    fraction is strictly greater than the threshold ``q``. Contiguous stable
    samples are then grouped into intervals.

    Parameters
    ----------
    results : ndarray of shape (N, 2)
        Output of :func:`stability_zone`, with semi-major axis in the first
        column and stability fraction in the second.
    q : float, optional
        Stability threshold used to define the boundary. Default is 0.9.

    Returns
    -------
    list of tuple
        List of intervals ``(a_min, a_max)`` defining the stable regions.

    Notes
    -----
    The boundary is extracted from a discretely sampled stability curve.
    Therefore, its precision depends on the resolution of the semi-major-axis
    grid used to generate ``results``.
    """
    a_p = results[:, 0]
    p = results[:, 1]

    # Keep only the semi-major axes that satisfy the stability threshold
    stable_a_p = results[p > q, 0]

    if len(stable_a_p) == 0:
        print("no stable planet")
        return []

    zones = []
    start = stable_a_p[0]

    for i in range(1, len(stable_a_p)):
        # Start a new zone if the gap exceeds the sampling step significantly
        if stable_a_p[i] - stable_a_p[i-1] > (a_p[1] - a_p[0]) * 1.5:
            zones.append((start, stable_a_p[i-1]))
            start = stable_a_p[i]

    # Close the final stable interval
    zones.append((start, stable_a_p[-1]))
    
    # Remove degenerate intervals
    zones = [(a_min, a_max) for a_min, a_max in zones if a_max > a_min]
    
    print(f"The stability zone is : {zones}")
    return zones


def stability_plot_bar(parameters, results, q=0.9):
    """
    Plot the stability fraction as a bar diagram and highlight stable regions.

    The figure displays the stability fraction as a function of planetary
    semi-major axis, the semi-major axis of the outer companion, and the
    intervals identified as stable according to :func:`stability_zone_boundary`.

    Parameters
    ----------
    parameters : dict
        System parameters. Only the companion semi-major axis is used for
        annotation.
    results : ndarray of shape (N, 2)
        Stability scan returned by :func:`stability_zone`.
    q : float, optional
        Stability threshold used to define highlighted stable regions.
        Default is 0.9.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax : matplotlib.axes.Axes
        Matplotlib axes object.

    Notes
    -----
    This representation is useful for quickly identifying the approximate
    extent of dynamically allowed planetary orbits in the explored parameter
    range.
    """
    fig, ax = plt.subplots()

    a_C = parameters["companion"]["a_C"]
    
    # Mark the semi-major axis of the outer stellar companion
    ax.axvline(x=a_C, linestyle='--', label='Companion semi-major axis a_C (AU)') #affichage semimajor axis compagnon
    ax.legend()

    # Extract and highlight the stable intervals
    zones = stability_zone_boundary(results, q=q)

    for a_min, a_max in zones:
        ax.axvspan(a_min, a_max, alpha=0.2)

    a_p = results[:,0]
    p = results[:,1]

    # Bar width set by the semi-major-axis sampling step
    dx = a_p[1] - a_p[0]

    ax.bar(a_p, p, width=dx, edgecolor='black', align='center')

    ax.set_xlabel("Planet semi-major axis a_p (AU)")
    ax.set_ylabel("Fraction of stable orbit")
    ax.set_title("Stability plot bar")
    ax.set_ylim(0, 1)

    ax.grid(True)

    return fig, ax


def luminosity(m):
    """
    Estimate stellar luminosity from stellar mass using a piecewise mass-luminosity relation.

    Parameters
    ----------
    m : float
        Stellar mass in solar masses.

    Returns
    -------
    float
        Stellar luminosity in watts.

    Notes
    -----
    The adopted prescription is a simplified piecewise mass-luminosity relation
    commonly used for main-sequence stars. It is intended for approximate
    radiative estimates and should not be interpreted as a detailed stellar
    evolution model.
    """
    L_SUN = 3.828e26
    
    # Approximate main-sequence mass-luminosity relation
    if m < 0.43:
        L = 0.23 * m**2.3*L_SUN
    elif m < 2:
        L = m**4*L_SUN
    elif m < 55:
        L = 1.4 * m**3.5*L_SUN
    else:
        L = 32000 * m*L_SUN
        
    return L


def temperature(parameters, positions, xlim, ylim, R=50, A=0.3, greenhouse_factor=0.61):
    """
    Compute equilibrium temperature maps from the combined stellar irradiation.

    A two-dimensional Cartesian grid is constructed over the domain defined by
    ``xlim`` and ``ylim``. At each time step, the equilibrium temperature is
    computed from the sum of the radiative fluxes of the three stars, assuming
    blackbody balance with prescribed Bond albedo and greenhouse efficiency
    factor.

    Parameters
    ----------
    parameters : dict
        System parameters used to infer stellar luminosities.
    positions : ndarray of shape (n_frames, 4, 2)
        Time-dependent body positions returned by :func:`positions`.
    xlim : tuple of float
        Minimum and maximum x-coordinates of the spatial domain, in AU.
    ylim : tuple of float
        Minimum and maximum y-coordinates of the spatial domain, in AU.
    R : int, optional
        Spatial resolution of the grid in each dimension. Default is 50.
    A : float, optional
        Bond albedo of the planet. Default is 0.3.
    greenhouse_factor : float, optional
        Effective greenhouse factor entering the radiative equilibrium formula.
        Default is 0.61.

    Returns
    -------
    X : ndarray of shape (R, R)
        x-coordinate meshgrid.
    Y : ndarray of shape (R, R)
        y-coordinate meshgrid.
    T : ndarray of shape (n_frames, R, R)
        Equilibrium temperature maps in kelvin.

    Notes
    -----
    The temperature model is intentionally simplified and does not include
    atmospheric circulation, thermal inertia, spectral dependence of albedo,
    seasonal effects, eclipsing, etc. It should be interpreted as a first-order radiative 
    estimation.
    """
    SIGMA = 5.670374419e-8       # Stefan-Boltzmann (W m^-2 K^-4)
    AU = 1.495978707e11          # (m)
    
    # Build the 2D spatial grid in the orbital plane
    x = np.linspace(xlim[0], xlim[1], R)
    y = np.linspace(ylim[0], ylim[1], R)
    X, Y = np.meshgrid(x, y)

    # Stellar luminosities inferred from stellar masses
    L_A = luminosity(parameters["binary"]["m_A"])
    L_B = luminosity(parameters["binary"]["m_B"])
    L_C = luminosity(parameters["companion"]["m_C"])

    eps = 1e-12
    n_frames = positions.shape[0]
    T = np.zeros((n_frames, R, R))

    for i in range(n_frames):
        # Extract instantaneous stellar positions in the inner-binary frame
        x_A, y_A = positions[i, 0]
        x_B, y_B = positions[i, 1]
        x_C, y_C = positions[i, 2]

        # Distance from each grid cell to each star
        d_A = np.sqrt((X - x_A)**2 + (Y - y_A)**2 + eps)
        d_B = np.sqrt((X - x_B)**2 + (Y - y_B)**2 + eps)
        d_C = np.sqrt((X - x_C)**2 + (Y - y_C)**2 + eps)

        # Equilibrium temperature from the summed stellar irradiation
        T_eq = ((1 - A) / (16 * np.pi * SIGMA * greenhouse_factor) *
                (L_A/(d_A*AU)**2 +
                 L_B/(d_B*AU)**2 +
                 L_C/(d_C*AU)**2)
               )**(1/4)

        T[i] = T_eq

    return X, Y, T


def animation(parameters, a_p, xlim, ylim, R, sz_boundary, HZ, planet_type, integrator="whfast", i_anim=100, n_anim=3, i_hab=1000, n_hab=100,  save=False, filename="animation.gif"):
    """
    Animate the orbital motion, stability zone, and habitable zone structure.

    This function generates a two-panel figure showing:
    - a time-dependent equilibrium temperature map,
    - the positions and recent trajectories of the stars and planet,
    - the precomputed stability zone boundaries,
    - the habitable zone derived from the thermal history over a longer run,
    - a side panel summarizing the physical parameters of the system.

    Parameters
    ----------
    parameters : dict
        System parameters.
    a_p : float
        Planetary semi-major axis used in the animation, in AU.
    xlim : tuple of float
        Minimum and maximum x-coordinates of the displayed domain, in AU.
    ylim : tuple of float
        Minimum and maximum y-coordinates of the displayed domain, in AU.
    R : int
        Spatial resolution of the temperature map.
    sz_boundary : list of tuple
        Stability zone intervals returned by :func:`stability_zone_boundary`.
    HZ : {"PHZ", "AHZ", "EHZ"}
        Type of habitable zone to display:
        ``"PHZ"`` for Permanently Habitable Zone,
        ``"AHZ"`` for Averaged Habitable Zone,
        ``"EHZ"`` for Extended Habitable Zone.
        (See Eggl 2012)
    planet_type : str
        Key of the selected planet profile in ``planet_profiles``.
    integrator : str, optional
        REBOUND integrator. Default is ``"whfast"``.
    i_anim : int, optional
        Number of frames in the animation sequence. Default is 100.
    n_anim : float, optional
        Number of planetary orbits covered by the animation. Default is 3.
    i_hab : int, optional
        Number of snapshots used to compute the habitable zone. Default is 1000.
    n_hab : float, optional
        Number of planetary orbits used to estimate the habitable zone.
        Default is 100.
    save : bool, optional
        If ``True``, save the animation to disk. Default is ``False``.
    filename : str, optional
        Output filename used when ``save=True``. Default is ``"animation.gif"``.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Matplotlib animation object.

    Notes
    -----
    The habitable zone is computed from a longer thermal sequence than the one
    displayed in the animation, in order to separate diagnostic visualization
    from statistical thermal characterization.

    The displayed reference frame is centered on the barycenter of the inner
    binary. This makes the stability zone stationary in the figure and improves
    visual comparison between the orbital architecture and the radiative domain.
    """
    # Planet-dependent radiative parameters
    A = planet_profiles[planet_type]["A"]
    greenhouse_factor = planet_profiles[planet_type]["greenhouse_factor"]
    
    # Use a single phase realization for both HZ computation and animation
    phase = random_phase()

    # Long integration used to define the habitable zone
    pos_hz = positions(parameters, phase, a_p, integrator=integrator,
                       i_max=i_hab, n_orbits=n_hab)
    X, Y, T_hz = temperature(parameters, pos_hz, xlim, ylim, R, A, greenhouse_factor)

    # Shorter integration used for the displayed animation
    pos_anim = positions(parameters, phase, a_p, integrator=integrator,
                         i_max=i_anim, n_orbits=n_anim)
    _, _, T_anim = temperature(parameters, pos_anim, xlim, ylim, R, A, greenhouse_factor)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.8], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    
    # Summary of the system parameters shown in the side panel
    param_text = (
    f"Binary\n"
    f"mass m_A = {parameters['binary']['m_A']} solar mass\n"
    f"mass m_B = {parameters['binary']['m_B']} solar mass\n"
    f"semi-major axis a_AB = {parameters['binary']['a_AB']} AU\n"
    f"eccentricity e_AB = {parameters['binary']['e_AB']}\n\n"
    f"Companion\n"
    f"mass m_C = {parameters['companion']['m_C']} solar mass\n"
    f"semi-major axis a_C = {parameters['companion']['a_C']} AU\n"
    f"e_C = {parameters['companion']['e_C']}\n\n"
    f"Planet\n"
    f"Planet type = {planet_type}\n"
    f"mass m_p = {parameters['planet']['m_p']} solar mass\n"
    f"semi-major axis a_p = {a_p} AU\n"
    f"eccentricity e_p = {parameters['planet']['e_p']}\n"
    f"albedo A = {A:.2f}\n"
    f"green house effect factor greenhouse_factor = {greenhouse_factor:.2f}\n"
    )

    ax_info.text(
    0.02, 0.98, param_text,
    transform=ax_info.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="black")
    )

    # Initial temperature map
    im = ax.imshow(
        T_anim[0],
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin='lower',
        cmap='inferno',
        vmin=0, vmax=1500,
        animated=True
    )
    plt.colorbar(im, ax=ax, label="Equilibrium temperature (K)")

    # Markers for stars and planet
    star_A, = ax.plot([], [], marker='*', color='w', markersize=6,  linestyle='None', label="A", zorder=5)
    star_B, = ax.plot([], [], marker='*', color='w', markersize=6,  linestyle='None', label="B", zorder=5)
    star_C, = ax.plot([], [], marker='*', color='w', markersize=4,  linestyle='None', label="C", zorder=5)
    planet, = ax.plot([], [], marker='o', color='cyan', markersize=4,  linestyle='None', label="Planet", zorder=6)

    # Draw the precomputed stability-zone boundaries
    theta = np.linspace(0, 2*np.pi, 400)

    for a_min, a_max in sz_boundary:
        x_inner = a_min * np.cos(theta)
        y_inner = a_min * np.sin(theta)
        x_outer = a_max * np.cos(theta)
        y_outer = a_max * np.sin(theta)

        ax.plot(x_inner, y_inner, color='black', alpha=0.5, lw=1.5, zorder=3)
        ax.plot(x_outer, y_outer, color='black', alpha=0.5, lw=1.5, zorder=3)

    # Short orbital trails used for visual clarity
    trail_A, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_B, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_C, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_p, = ax.plot([], [], color='cyan', alpha=0.8, lw=0.8)
       
    # Construct the habitable zone from the long thermal evolution depending of the HZ type chosen
    if HZ == 'PHZ':
        T_min = np.min(T_hz, axis=0)
        T_max = np.max(T_hz, axis=0)
        zone = (T_min > 273) & (T_max < 373)
        print("Number of points in HZ :", np.sum(zone))
    elif HZ == 'AHZ':
        T_mean = np.mean(T_hz, axis=0)
        zone = (T_mean > 273) & (T_mean < 373)
        print("Number of points in HZ :", np.sum(zone))
    elif HZ == 'EHZ':
        zone = np.any((T_hz > 273) & (T_hz < 373), axis=0)
        print("Number of points in HZ :", np.sum(zone))
    else:
        raise ValueError("HZ doit valoir 'PHZ' ou 'AHZ' ou 'EHZ'")

    # Overlay the habitable zone on the temperature map
    ax.contourf(
        X, Y, zone.astype(float),
        levels=[0.5, 1.5],
        colors=['deepskyblue'],
        alpha=0.4,
        zorder=2
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_aspect('equal')
    ax.legend()
    

    def update(frame):
        # Update the temperature map
        im.set_data(T_anim[frame])

        # Update body positions
        star_A.set_data([pos_anim[frame, 0, 0]], [pos_anim[frame, 0, 1]])
        star_B.set_data([pos_anim[frame, 1, 0]], [pos_anim[frame, 1, 1]])
        star_C.set_data([pos_anim[frame, 2, 0]], [pos_anim[frame, 2, 1]])
        planet.set_data([pos_anim[frame, 3, 0]], [pos_anim[frame, 3, 1]])
        
        # Trail lengths chosen independently for inner binary, companion, and planet
        window_AB = 200
        window_C = 300
        window_p = 300
        
        start_AB = max(0, frame - window_AB)
        start_C = max(0, frame - window_C)
        start_p = max(0, frame - window_p)
        
        trail_A.set_data(pos_anim[start_AB:frame+1, 0, 0], pos_anim[start_AB:frame+1, 0, 1])
        trail_B.set_data(pos_anim[start_AB:frame+1, 1, 0], pos_anim[start_AB:frame+1, 1, 1])
        trail_C.set_data(pos_anim[start_C:frame+1, 2, 0], pos_anim[start_C:frame+1, 2, 1])
        trail_p.set_data(pos_anim[start_p:frame+1, 3, 0], pos_anim[start_p:frame+1, 3, 1])

        ax.set_title(f"Frame {frame} - stability zone and {HZ}")

        return im, star_A, star_B, star_C, planet, trail_A, trail_B, trail_C, trail_p

    # Create and save the animation as a .gif file
    ani = anim.FuncAnimation(fig, update, frames=len(T_anim), interval=50)
    
    if save:
        ani.save(filename, writer="pillow", fps=20)
    
    plt.show()
    
    return ani

# Reference planetary profiles used for the radiative equilibrium model.
planet_profiles = {
    "earth": {"A": 0.30, "greenhouse_factor": 0.61, "m_p": 1e-3},
    "venus": {"A": 0.75, "greenhouse_factor": 0.02, "m_p": 0.8e-3},
    "super earth": {"A": 0.30, "greenhouse_factor": 0.61, "m_p": 5e-3}
}

planet_type = "earth"

# Example system used for validation and exploratory tests.
parameters_test = {
    "binary": {"m_A": 2.0, "m_B": 1.5, "a_AB": 0.2, "e_AB": 0},
    "companion": {"m_C": 0.2, "a_C": 7.0, "e_C": 0.1, "inc_C": 0},
    "planet": {"m_p": planet_profiles[planet_type]["m_p"], "e_p": 0, "inc_p": np.pi}
}

# Approximate orbital configuration inspired by HD 188753.
# The system is treated here in a simplified coplanar configuration and null companion's eccentricity
HD188753 = {
    "binary": {"m_A": 0.86, "m_B": 0.66, "a_AB": 0.648, "e_AB": 0.175},
    "companion": {"m_C": 0.99, "a_C": 12.3, "e_C": 0.1, "inc_C": 0},
    "planet": {"m_p": planet_profiles[planet_type]["m_p"], "e_p": 0, "inc_p": 0}
}

# SZ and HZ parameters
integrator = "whfast"
n_stab = 5000
n_hab = 1000
i_hab = 5000
r = 5
N = 30
k = 50
q = 0.9
HZ = 'PHZ'

# Range of semi-major axis used for the calculation of the SZ
a_p_range = [HD188753["binary"]["a_AB"], HD188753["companion"]["a_C"]]

# Calculation of the SZ boundary
sz_boundary = stability_zone_boundary(
    stability_zone(
        HD188753,
        a_p_range=a_p_range,
        integrator=integrator,
        N=N,
        n=n_stab,
        r=r,
        k=k
    ),
    q=q
)

# Animation parameters
a_min, a_max = sz_boundary[0]
a_p = rd.uniform(a_min, a_max)
l = 40 # size of the map
xlim = (-l/2, l/2)
ylim = (-l/2, l/2)
i_anim = 500
n_anim = 10 
R = 1000

# Create an animation
ani = animation(
    parameters=HD188753,
    a_p=a_p,
    xlim=xlim,
    ylim=ylim,
    R=R,
    sz_boundary=sz_boundary,
    planet_type=planet_type,
    HZ=HZ,
    integrator=integrator,
    i_anim=i_anim,
    n_anim=n_anim,
    i_hab=i_hab,
    n_hab=n_hab,
    save=True,
    filename="hz_sz_animation.gif"
)
