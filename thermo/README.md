# Precipitation Thermodynamics Using CALPHAD

## Sublattice Substitutions

The reference CALPHAD database for Cr-Nb-Ni provides phases with the
following sublattice models, in descending order of observation frequency:

- γ, FCC: (Cr, Nb, Ni)<sub>1</sub>.
- δ, NbNi<sub>3</sub>: (**Nb**, Ni)<sub>1/4</sub> (Cr, Nb, **Ni**)<sub>3/4</sub>.
- λ, NbCr<sub>2</sub>: (Cr, **Nb**)<sub>1/3</sub>(**Cr**, Nb, Ni)<sub>2/3</sub>.
- μ, Nb<sub>6</sub>Ni<sub>7</sub>: **Nb**<sub>6/13</sub>(Cr, Nb, **Ni**)<sub>7/13</sub>.
- β, BCC: (Cr, Nb, Ni)<sub>1</sub>.
- Liquid: (Cr, Nb, Ni)<sub>1</sub>.

Based on X-ray and electron diffraction characterization of additively
manufactured parts, the γ, δ, and λ phases are of the most interest
(neither β nor μ are routinely observed). In order to produce a one-to-one
mapping from system to sublattice composition -- avoiding solving for
internal equilibrium when using the phase-field model -- the sublattice
definitions are modified to create clean segregation of constituents. For
δ phase, Nb is eliminated from the **Ni** sublattice; for λ phase, Nb is
eliminated from the **Cr** sublattice. This redefinition does change the
phase diagram slightly, but its important characteristics are preserved.

These changes are reiterated and implemented in
[CALPHAD_energies.py](CALPHAD_energies.py).

## Paraboloid Representations

The Kim-Kim-Suzuki interface model assumed in this work requires smooth,
continuous functions defined throughout the Gibbs simplex. CALPHAD models
do not provide that: rather, there are usually regions where CALPHAD free
energy functions are undefined. To resolve this incompatibility, and to
simplify linear algebra later on, a second-order Taylor series expansion is
used to approximate the free energy of each phase.

1. A three-phase coexistence field of interest is identified using Pandat
   to compute the phase diagram from the simplified CALPHAD database.
2. The compositions corresponding to the vertices of the coexistence
   triangle were extracted, corresponding to the composition at equilibrium
   of each coexisting phase, using ImageJ and Pandat.
3. An isothermal temperature of 873°C is chosen, to reflect the manufacturer's
   recommended stress relieving heat treatment.
4. The free energy of each phase is approximated using a Taylor series
   expansion about its equilibrium coexistence composition:
   
   ```latex
   G_{\alpha}(x_1, x_2) \approx
   + 0.5 * \frac{\partial^2 G(x_1^e, x_2^e)}{\partial (x_1)^2}
           \left(x_1 - x_1^e\right)^2
   +       \frac{\partial^2 G(x_1^e, x_2^e)}{\partial x_1 \partial x_2}
                \left(x_1 - x_1^e\right)\left(x_2 - x_2^e\right)
   + 0.5 * \frac{\partial^2 G(x_1^e, x_2^e)}{\partial (x_2)^2}
           \left(x_2 - x_2^e\right)^2
   ```
   Note that the term $G_0 = G(x_1^e, x_2^e)$ has been excluded, so that
   each phase has a free energy of precisely zero at its equilibrium
   coexistence composition. PyCalphad is used to read the CALPHAD database,
   and SymPy is used to compute the paraboloid expressions and write them
   to disk as C code. The Gibbs free energies are divided by the molar
   volume of FCC Ni to convert from a molar basis to the volumetric form
   expected by phase-field models.

## Initial Condition

Rapid solidification of the melt pool during additive manufacturing produces a
cellular/dendritic microstructure, with microsegregation of solute elements
enriching the interdendritic regions (last to solidify). This enrichment favors
precipitation of secondary phases, which is the focus of this research. The
initial condition is therefore a rectangular 2D window 1 μm across with a
Gaussian [bell curve](enrichment.c) composition profile in both Cr and Nb
creating a composition "peak" down the centerline. At $t=0$, the entire system
is in the γ phase.

## Nucleation

For the sake of simplicity, the classical theory of nucleation is
implemented in an order-parameter-only homogeneous precipitation model.
Tuning of model parameters is done using the
[check-nucleation](check-nucleation.cpp) program, which is much faster
to run than the full model. It randomly selects a composition within the
enrichment range, and prints the probability of nucleating each secondary
phase (along with other useful debugging information).
