# Coupled Thermo-Mechanical Simulation Model for Rotating Heat Pipes (RHP)

This repository contains the complete Python implementation (~2200 lines) of a coupled thermo-mechanical simulation model for rotating heat pipes (RHP), developed within the context of an academic research project.

The model combines thermo-mechanical pressure balance, hybrid liquid transport modeling, and performance limit evaluation to simulate rotating heat pipe behavior under varying operating conditions.

---

## Features

- Coupled capillary–centrifugal pressure balance  
- Hybrid liquid return model (Darcy wick flow + parallel film flow)  
- Rossby-number-based viscosity correction  
- Acceleration-dependent boiling suppression  
- Performance limit evaluation:
  - Capillary limit  
  - Sonic limit  
  - Entrainment limit  
  - Viscous limit  
  - Boiling limit  
- Graphical User Interface (GUI)

---

## Installation

Tested with Python 3.11.

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Model

### German Version
```bash
python rhp_model.py
```

### English Version
```bash
python rhp_model_en.py
```

---

## Reproducibility

The version corresponding to the results presented in the associated thesis/paper is archived and identified by the tag:

`v1.0-paper`

This tagged version ensures full traceability and reproducibility of the reported numerical results.

---

## Repository Structure

```
rhp_model.py        # German version
rhp_model_en.py     # English version
requirements.txt    # Python dependencies
.gitignore          # Ignored files
README.md
```

---

