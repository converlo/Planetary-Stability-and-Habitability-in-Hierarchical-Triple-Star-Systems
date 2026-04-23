# Planetary Stability and Habitability in Hierarchical Triple Star Systems

This repository provides a research-oriented Python framework designed to study the overlap between orbital stability and radiative habitability in hierarchical triple star systems hosting circumbinary planets. The code combines N-body integrations performed with REBOUND with simplified equilibrium-temperature mapping to explore the conditions under which long-term dynamical stability and potentially habitable thermal environments may coexist.

This project is developed by **Loïc Converset** and **Elie Desmartin** within the **ExTReM thematic program** of the Graduate School at **Université Grenoble Alpes (UGA)**. The work is carried out at the **Institut de Planétologie et d’Astrophysique de Grenoble (IPAG)**, as part of the **Odyssey team** and the **Stellar-MADE project**, under the supervision of **Mario Sucerquia**, **Romain Grane**, and **Nicolás Cuello**.

## Project overview

The objective of this project is to investigate how dynamical stability constraints and stellar irradiation jointly shape potentially habitable regions in hierarchical triple stellar systems. The current implementation focuses on circumbinary planets orbiting the inner binary, while an outer stellar companion acts as both a gravitational perturber and an additional radiative source.

The framework allows the user to:
- generate hierarchical triple-star configurations,
- integrate orbital evolution with REBOUND,
- estimate stability fractions over randomized initial orbital phases,
- derive approximate stability-zone boundaries,
- compute equilibrium temperature maps from the combined flux of three stars,
- visualize the overlap between stability zones (SZ) and habitable zones (HZ).


## Requirements

The code requires Python 3 and the following packages:
- `rebound`
- `numpy`
- `random`
- `os`
- `matplotlib`

You can install the main dependencies with:

```bash
pip install rebound numpy matplotlib os random
