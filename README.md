# Multi-pulse ion-electron(-photon) entanglement

Research code for driven ion-electron entanglement with Gaussian or flat-top pulse sequences, including a one-photon spontaneous-emission extension.

## Structure

- `common/general_functions.py` — atomic-unit conversions and generic helpers.
- `common/entanglement.py` — ion-electron and ion-electron-photon entropy calculations.
- `multipulse/functions.py` — pulse envelopes, pulse areas, Rabi amplitudes, depletion, and ion-electron continuum amplitudes.
- `multipulse/energy_grids.py` — adaptive photoelectron and photon-energy grids.
- `multipulse/spontaneous_emission.py` — amplitudes `c_a`, `c_b`, `c_c`, and `c_d` including spontaneous emission.
- `multipulse/plotting.py` — reusable plotting functions with labels, legends, and consistent formatting.
- `multipulse/Run_Multipulse.ipynb` — interactive parameter definition, example scans, and analysis.

## Notebook workflow

Open `multipulse/Run_Multipulse.ipynb`, define the system and pulse sequence, and run the desired sections. The notebook uses `%autoreload 2`, so saved changes to the `.py` modules are picked up automatically during development.

The optional secondary photon grid in `def_eps_l` uses graded spacing: it is dense where it joins the central grid and becomes progressively sparser in the outer wings. This avoids an artificial large trapezoidal contribution at an abrupt grid-spacing boundary.

## Dependencies

Python with NumPy, SciPy, Matplotlib, and Jupyter.
