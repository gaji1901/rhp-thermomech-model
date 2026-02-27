# Coupled Thermo-Mechanical Simulation Model for Rotating Heat Pipes (RHP)

This repository contains the complete Python implementation (~2200 lines)
of the thermo-mechanical simulation model described in the associated research paper.

The model includes:
- Coupled thermo-mechanical pressure balance
- Hybrid liquid transport model (Darcy + film flow)
- Rossby-number based correction
- Acceleration-dependent boiling suppression
- Evaluation of performance limits (capillary, sonic, entrainment, boiling, viscous)

---

## Installation

Tested with Python 3.11.

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Model

If using the GUI:

```bash
python rhp_model.py
```

If using command line execution:

```bash
python run_model.py --fluid Water --rpm 20000 --tempC 100
```

---

## Reproducibility

The version corresponding to the results presented in the paper is tagged as:

```
v1.0-paper
```

---

## Author

GitHub: https://github.com/gaji1901

---

## License

See LICENSE file.