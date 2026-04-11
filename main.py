"""
Dans cette version l'animation est centré auour du barycentre du bianire plutôt qu'autour du barycentre du système triple afin de garder une sz fixe, le seule changement intervient dans la focntion position
Utilisation de P=np.sqrt(a_p**3 / (m_A + m_B)) orbite képlerienne parce que l'orbite est tres mal defini par rebound
"""

import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random as rd

#%%

def simulation(parameters, phase, a_p, integrator = "whfast"):
  # Parameters extraction
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

    cdmAB = sim.com(0, 2)

    # Companion
    sim.add(hash="C", m=m_C, e=e_C, a=a_C, f=f_C, omega=w_C, inc=inc_C, primary=cdmAB)

    # Circumbinary planet
    sim.add(hash="p", m=m_p, e=e_p, a=a_p, f=f_p, omega=w_p, inc=inc_p, primary=cdmAB)

    sim.move_to_com()

    return sim

def positions(parameters, phase, a_p, integrator = "whfast", i_max=100, n_orbits=100):
    sim = simulation(parameters, phase, a_p, integrator = integrator)

    A = sim.particles["A"]
    B = sim.particles["B"]
    C = sim.particles["C"]
    p = sim.particles["p"]
    
    m_A = parameters["binary"]["m_A"]
    m_B = parameters["binary"]["m_B"]

    if sim.integrator.lower() == "whfast":
        P_B = sim.particles["B"].P
        sim.dt = P_B/20

    P = np.sqrt(a_p**3 / (m_A + m_B))
    times = np.linspace(0, n_orbits*P, i_max) # On garde une période planétaire pour l'animation
    pos = np.zeros((i_max, 4, 2)) #nombre de time step, nombre de corps, nombre de dimension

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
    sim = simulation(parameters, phase, a_p, integrator=integrator)

    C = sim.particles["C"]
    p = sim.particles["p"]
    #print(f"a_p={a_p:.3f}, p.P={p.P}, sim.t={sim.t}")
    
    #if not np.isfinite(p.P) or p.P <= 0:
        #print(f"Orbital period invalid for a_p={a_p:.3f}: P={p.P}")
        #return False
    
    m_A = parameters["binary"]["m_A"]
    m_B = parameters["binary"]["m_B"]
    
    P = np.sqrt(a_p**3 / (m_A + m_B))
    t_max = P * n
    #print(f"t_max={t_max}")
    
    if sim.integrator.lower() == "whfast":
        P_B = sim.particles["B"].P
        sim.dt = P_B / 20
        dt = sim.dt

    steps = 0

    while sim.t < t_max:
        sim.integrate(sim.t + dt)
        steps += 1

        orbite_p = p.orbit(primary=sim.com(0,2))
        orbit_C = C.orbit(primary=sim.com(0,2))

        if orbite_p.e >= 1 or orbite_p.a > r * orbit_C.a:
            #print(f"Instable : a_p={a_p:.3f}, t={sim.t:.3f}, steps={steps}, e={orbite_p.e:.3f}, a={orbite_p.a:.3f}")
            return False

    #print(f"Stable jusqu'au bout : a_p={a_p:.3f}, steps={steps}")
    return True

def random_phase():
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
    N_stable = 0
    step = 0
    if N==0:
      phase_zero = {
            "binary": {"f_B": 0, "w_B": 0},
            "companion": {"f_C": 0, "w_C": 0},
            "planet": {"f_p": 0, "w_p": 0}
          }

      if integration(parameters=parameters, phase=phase_zero, a_p=a_p, integrator=integrator, n=n, r=r):
            N_stable = 1
      return N_stable
    else :
      for _ in range(N):
          phase = random_phase()
          if integration(parameters=parameters, phase=phase, a_p=a_p, integrator=integrator, n=n, r=r):
              
            N_stable += 1
          step += 1  
          print(f"step : {step} / {N}")
      return N_stable / N

def stability_zone(parameters, a_p_range, integrator="whfast", N=30, n=10000, r=10, k=25):
    results = []
    a_p_list = np.linspace(a_p_range[0], a_p_range[1], k)

    for a_p in a_p_list:
        p = stability_fraction(parameters=parameters, a_p=a_p, integrator=integrator, N=N, n=n, r=r)
        results.append((a_p, p))
        print(f"a_p = {a_p}")
    return np.array(results)

def stability_zone_boundary(results, q=0.9):
    if len(results) == 0:
        print("no stable planet")
        return []

    a_p = results[:, 0]
    p = results[:, 1]

    stable_a_p = results[p > q, 0]

    if len(stable_a_p) == 0:
        print("no stable planet")
        return []

    zones = []
    start = stable_a_p[0]

    for i in range(1, len(stable_a_p)):
        if stable_a_p[i] - stable_a_p[i-1] > (a_p[1] - a_p[0]) * 1.5:
            zones.append((start, stable_a_p[i-1]))
            start = stable_a_p[i]

    zones.append((start, stable_a_p[-1]))
    zones = [(a_min, a_max) for a_min, a_max in zones if a_max > a_min]
    
    return zones


def stability_plot_bar(parameters, results, q=0.9):
    fig, ax = plt.subplots()

    a_C = parameters["companion"]["a_C"]
    ax.axvline(x=a_C, linestyle='--', label='Companion semi-major axis a_C (AU)') #affichage semimajor axis compagnon
    ax.legend()

    zones = stability_zone_boundary(results)

    for a_min, a_max in zones:
        ax.axvspan(a_min, a_max, alpha=0.2)

    a_p = results[:,0]
    p = results[:,1]

    dx = a_p[1] - a_p[0]

    ax.bar(a_p, p, width=dx, edgecolor='black', align='center')

    ax.set_xlabel("Planet semi-major axis a_p (AU)")
    ax.set_ylabel("Fraction of stable orbit")
    ax.set_title("Stability plot bar")
    ax.set_ylim(0, 1)

    ax.grid(True)

    return fig, ax

def luminosity(m):
    L_SUN = 3.828e26
    if m < 0.43:
        L = 0.23 * m**2.3*L_SUN
    elif m < 2:
        L = m**4*L_SUN
    elif m < 55:
        L = 1.4 * m**3.5*L_SUN
    else:
        L = 32000 * m*L_SUN
    return L


SIGMA = 5.670374419e-8       # Stefan-Boltzmann (W m^-2 K^-4)
AU = 1.495978707e11          # (m)

def temperature(parameters, positions, xlim, ylim, R=50, A=0.3, ghe=0.61): # A -> albedo 0.3 par defaut (albedo terre) ; e -> effet de serre 0.61 pour la terre trouver source et verifier formule pour effet de serre, opacité IR ??
    x = np.linspace(xlim[0], xlim[1], R)
    y = np.linspace(ylim[0], ylim[1], R)
    X, Y = np.meshgrid(x, y)

    L_A = luminosity(parameters["binary"]["m_A"])
    L_B = luminosity(parameters["binary"]["m_B"])
    L_C = luminosity(parameters["companion"]["m_C"])

    eps = 1e-12
    n_frames = positions.shape[0]
    T = np.zeros((n_frames, R, R))

    for i in range(n_frames):
        x_A, y_A = positions[i, 0]
        x_B, y_B = positions[i, 1]
        x_C, y_C = positions[i, 2]

        d_A = np.sqrt((X - x_A)**2 + (Y - y_A)**2 + eps)
        d_B = np.sqrt((X - x_B)**2 + (Y - y_B)**2 + eps)
        d_C = np.sqrt((X - x_C)**2 + (Y - y_C)**2 + eps)

        T_eq = ((1 - A) / (16 * np.pi * SIGMA * ghe) *
                (L_A/(d_A*AU)**2 +
                 L_B/(d_B*AU)**2 +
                 L_C/(d_C*AU)**2)
               )**(1/4)

        T[i] = T_eq

    return X, Y, T

def animation(parameters, a_p, xlim, ylim, R, sz_boundary, HZ, planet_type, integrator="whfast", i_anim=100, n_anim=3, i_hab=1000, n_hab=100,  save=False, filename="animation.mp4"):
    A = planet_profiles[planet_type]["A"]
    ghe = planet_profiles[planet_type]["ghe"]
    
    phase = random_phase()

    pos_hz = positions(parameters, phase, a_p, integrator=integrator, i_max=i_hab, n_orbits=n_hab)
    X, Y, T_hz = temperature(parameters, pos_hz, xlim, ylim, R, A, ghe)

    pos_anim = positions(parameters, phase, a_p, integrator=integrator, i_max=i_anim, n_orbits=n_anim)
    _, _, T_anim = temperature(parameters, pos_anim, xlim, ylim, R, A, ghe)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.8], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    
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
    f"green house effect factor ghe = {ghe:.2f}\n"
    )

    ax_info.text(
    0.02, 0.98, param_text,
    transform=ax_info.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="black")
    )

    im = ax.imshow(
        T_anim[0],
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin='lower',
        cmap='inferno',
        vmin=0, vmax=1500,
        animated=True
    )
    plt.colorbar(im, ax=ax, label="Equilibrium temperature (K)")

    # étoiles + planète
    star_A, = ax.plot([], [], marker='*', color='w', markersize=6,  linestyle='None', label="A", zorder=5)
    star_B, = ax.plot([], [], marker='*', color='w', markersize=6,  linestyle='None', label="B", zorder=5)
    star_C, = ax.plot([], [], marker='*', color='w', markersize=4,  linestyle='None', label="C", zorder=5)
    planet, = ax.plot([], [], marker='o', color='cyan', markersize=4,  linestyle='None', label="Planet", zorder=6)

    # SZ (statique)
    theta = np.linspace(0, 2*np.pi, 400)

    for a_min, a_max in sz_boundary:
        x_inner = a_min * np.cos(theta)
        y_inner = a_min * np.sin(theta)
        x_outer = a_max * np.cos(theta)
        y_outer = a_max * np.sin(theta)

        ax.plot(x_inner, y_inner, color='black', alpha=0.5, lw=1.5, zorder=3)
        ax.plot(x_outer, y_outer, color='black', alpha=0.5, lw=1.5, zorder=3)
        
        #ax.fill(
        #    np.concatenate([x_inner, x_outer[::-1]]),
        #    np.concatenate([y_inner, y_outer[::-1]]),
        #    color='black',
        #    alpha=0.2,
        #    zorder=0,
        #    linewidth=0
        #)

    trail_A, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_B, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_C, = ax.plot([], [], color='white', alpha=0.8, lw=0.8)
    trail_p, = ax.plot([], [], color='cyan', alpha=0.8, lw=0.8)
       
    if HZ == 'PHZ':
        T_min = np.min(T_hz, axis=0)
        T_max = np.max(T_hz, axis=0)
        zone = (T_min > 273) & (T_max < 373)
        print("Nombre de points dans HZ :", np.sum(zone))
    elif HZ == 'AHZ':
        T_mean = np.mean(T_hz, axis=0)
        zone = (T_mean > 273) & (T_mean < 373)
        print("Nombre de points dans HZ :", np.sum(zone))
    elif HZ == 'EHZ':
        zone = np.any((T_hz > 273) & (T_hz < 373), axis=0)
        print("Nombre de points dans HZ :", np.sum(zone))
    else:
        raise ValueError("HZ doit valoir 'PHZ' ou 'AHZ' ou 'EHZ'")

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
        im.set_data(T_anim[frame])

        star_A.set_data([pos_anim[frame, 0, 0]], [pos_anim[frame, 0, 1]])
        star_B.set_data([pos_anim[frame, 1, 0]], [pos_anim[frame, 1, 1]])
        star_C.set_data([pos_anim[frame, 2, 0]], [pos_anim[frame, 2, 1]])
        planet.set_data([pos_anim[frame, 3, 0]], [pos_anim[frame, 3, 1]])
        
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

    ani = anim.FuncAnimation(fig, update, frames=len(T_anim), interval=50)
    
    if save:
        ani.save(filename, writer="pillow", fps=20)
    
    plt.show()
    
    return ani


#%%

planet_profiles = {
    "earth": {"A": 0.30, "ghe": 0.61, "m_p": 1e-3},
    "venus": {"A": 0.75, "ghe": 0.02, "m_p": 0.8e-3},
    "super earth": {"A": 0.30, "ghe": 0.61, "m_p": 5e-3}
}

planet_type = "earth"

# Paramètres orbitaux de test
parameters_test = {
    "binary": {"m_A": 2.0, "m_B": 1.5, "a_AB": 0.2, "e_AB": 0},
    "companion": {"m_C": 0.2, "a_C": 7.0, "e_C": 0.1, "inc_C": 0},
    "planet": {"m_p": planet_profiles[planet_type]["m_p"], "e_p": 0, "inc_p": np.pi}
}

# Paramètres orbitaux du système HD 188753
HD188753 = {
    "binary": {"m_A": 0.86, "m_B": 0.66, "a_AB": 0.648, "e_AB": 0.175},
    "companion": {"m_C": 0.99, "a_C": 12.3, "e_C": 0.1, "inc_C": 0}, # inclinaison de 10,9° pour simplifier on prendra un sytème coplanaire 
    "planet": {"m_p": planet_profiles[planet_type]["m_p"], "e_p": 0, "inc_p": 0}
}

# Paramètres HZ et SZ
integrator = "whfast" # choix de l'intégrateur
n_stab = 5000 # nombre d'orbites utilisés pour la SZ
n_hab = 1000 # nombre d'orbites utilisés pour la HZ
i_hab = 5000 # nombre de frames utilisées pour calculer la HZ
r = 50 # critère d'instabilités pour l'éjection planétaire
N = 30 # nombre de tirage aléatoire sur la phase
k = 50 # résolution spatiale 
q = 0.9 # critère de stabilité
HZ = 'PHZ' # type of hz (see Eggl 2012)

a_p_range = [HD188753["binary"]["a_AB"], HD188753["companion"]["a_C"]]

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

# Paramètres animation
a_min, a_max = sz_boundary[0]
a_p = rd.uniform(a_min, a_max) # demi-grand axe de la planète pris aléatoirement dans la première des zone de stabilité pour l'animation
l = 40 # taille de la map
xlim = (-l/2, l/2)
ylim = (-l/2, l/2)
i_anim = 500 # résolution temporelle
n_anim = 10 # nombre d'orbites planétaires utilisées pour l'animation
R = 1000 # résolution spatiale

print(sz_boundary)
#%%
"""
planet_profiles = {
    "earth": {"A": 0.30, "ghe": 0.61, "m_p": 1e-3},
    "venus": {"A": 0.75, "ghe": 0.02, "m_p": 0.8e-3},
    "super earth": {"A": 0.30, "ghe": 0.61, "m_p": 5e-3}
}

planet_type = "earth"

HD188753 = {
    "binary": {"m_A": 0.86, "m_B": 0.66, "a_AB": 0.648, "e_AB": 0.175},
    "companion": {"m_C": 0.99, "a_C": 12.3, "e_C": 0.1, "inc_C": 0}, # inclinaison de 10,9° pour simplifier on prendra un sytème coplanaire 
    "planet": {"m_p": planet_profiles[planet_type]["m_p"], "e_p": 0, "inc_p": 0}
}

sz_boundary = [(np.float64(2.074775510204082), np.float64(3.0259591836734696))]

a_min, a_max = sz_boundary[0]
a_p = rd.uniform(a_min, a_max) # demi-grand axe de la planète pris aléatoirement dans la première des zone de stabilité pour l'animation
l = 40 # taille de la map
xlim = (-l/2, l/2)
ylim = (-l/2, l/2)
i_anim = 500 # résolution temporelle
n_anim = 10 # nombre d'orbites planétaires utilisées pour l'animation
R = 1000 # résolution spatiale

# Paramètres HZ et SZ
integrator = "whfast" # choix de l'intégrateur
n_stab = 5000 # nombre d'orbites utilisés pour la SZ
n_hab = 1000 # nombre d'orbites utilisés pour la HZ
i_hab = 5000 # nombre de frames utilisées pour calculer la HZ
r = 50 # critère d'instabilités pour l'éjection planétaire
N = 30 # nombre de tirage aléatoire sur la phase
k = 50 # résolution spatiale 
q = 0.9 # critère de stabilité
HZ = 'PHZ' # type of hz (see Eggl 2012)
"""
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
