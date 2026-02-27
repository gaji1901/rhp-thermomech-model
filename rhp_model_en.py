import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # Allows embedding of Matplotlib plots in Tkinter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
import json # For saving and loading configurations
import os   # For file system operations

# Optional import for Excel export functionality
try:
    import pandas as pd
except ImportError:
    pd = None

# Import the CoolProp library to query exact thermophysical property data
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("Error: CoolProp is not installed. Please run 'pip install CoolProp'.")
    exit()

# ==============================================================================
# CONSTANTS & SYSTEM SETTINGS
# ==============================================================================
ROSSBY_FACTOR_VAPOR = 0.5   # Empirical factor to account for swirl effects in the vapor phase
ROSSBY_FACTOR_LIQUID = 0.15 # Empirical factor for the backflow of the liquid film under rotation
DESIGN_Q_TARGET = 100.0     # Reference load [W] for the initialization of iterations
NUCLEATION_RADIUS_DEFAULT = 2.0e-6 # Standard radius for nucleation sites during boiling (Nucleate Boiling)

# ==============================================================================
# PART 1: DATA STRUCTURES FOR PARAMETERS & GEOMETRY
# ==============================================================================

@dataclass
class HeatPipeParameters:
    """
    Data class for storing all geometric and physical
    properties of the heat pipe.
    """
    length: float           # Total length of the pipe [m]
    d_out: float            # Outer diameter [m]
    d_in: float             # Inner diameter (vapor core) [m]
    
    l_evap: float           # Length of the evaporator zone [m]
    l_adiab: float          # Length of the adiabatic (transport) zone [m]
    l_cond: float           # Length of the condenser zone [m]
    
    r_min: float            # Axis offset at the condenser (eccentricity) [m]
    cone_angle_deg: float   # Internal cone angle to promote fluid transport [deg]
    inclination_deg: float  # Global inclination of the system in space [deg]
    
    pore_radius: float      # Effective pore radius of the capillary structure [m]
    permeability: float     # Permeability of the wick [m^2]
    wick_area: float        # Cross-sectional area of the wick [m^2]
    wick_thickness: float   # Thickness of the sintered structure [m]
    porosity: float         # Porosity of the sintered material (0.0 - 1.0)
    
    k_solid: float          # Thermal conductivity of the pipe material [W/mK]
    
    # Material properties for transient calculations (Default: Copper)
    rho_solid: float = 8960.0  # Density of the solid [kg/m^3]
    cp_solid: float = 385.0    # Specific heat capacity of the solid [J/kgK]

    def get_tilt_rad(self):
        """Returns the internal cone angle in radians."""
        return np.radians(self.cone_angle_deg)
    
    def get_global_inclination_rad(self):
        """Returns the global system inclination in radians."""
        return np.radians(self.inclination_deg)

    @property
    def l_eff(self):
        """
        Calculates the effective length for friction losses.
        Definition: 0.5 * L_evaporator + L_adiabatic + 0.5 * L_condenser.
        """
        return 0.5 * self.l_evap + self.l_adiab + 0.5 * self.l_cond

# ==============================================================================
# PART 2: MATERIAL PROPERTIES (COOLPROP WRAPPER)
# ==============================================================================

class WorkingFluid:
    """
    Class for managing the working fluid and querying temperature-dependent
    material properties via the CoolProp interface.
    """
    def __init__(self, name="Water"):
        self.name = name

    def get_properties(self, T_k):
        """
        Determines all relevant thermophysical properties for a given
        temperature T [Kelvin].
        """
        fluid_map = {"Water": "Water", "Methanol": "Methanol", "Ammonia": "Ammonia"}
        fluid_string = fluid_map.get(self.name, "Water")
        
        # Temperature limitation to prevent numerical divergence near the critical point
        T_crit = PropsSI('T_CRITICAL', fluid_string)
        T = np.clip(T_k, 200.0, T_crit - 5.0)

        try:
            # Query density for liquid (Q=0) and vapor (Q=1)
            rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid_string)
            rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid_string)
            
            # Query viscosity
            mu_l = PropsSI('V', 'T', T, 'Q', 0, fluid_string)
            mu_v = PropsSI('V', 'T', T, 'Q', 1, fluid_string)
            
            # Other thermodynamic properties
            sigma = PropsSI('I', 'T', T, 'Q', 0, fluid_string) # Surface tension
            k_l = PropsSI('L', 'T', T, 'Q', 0, fluid_string)   # Thermal conductivity liquid
            p_sat = PropsSI('P', 'T', T, 'Q', 0, fluid_string) # Saturation vapor pressure
            
            # Enthalpy for calculating the heat of vaporization
            h_gas = PropsSI('H', 'T', T, 'Q', 1, fluid_string)
            h_liq = PropsSI('H', 'T', T, 'Q', 0, fluid_string)
            h_fg = h_gas - h_liq
            
            # Specific heat capacity of the liquid
            cp_l = PropsSI('C', 'T', T, 'Q', 0, fluid_string)
            
            # Calculation of the isentropic exponent (Gamma) for speed of sound limits
            cp_v = PropsSI('C', 'T', T, 'Q', 1, fluid_string)
            cv_v = PropsSI('O', 'T', T, 'Q', 1, fluid_string) # 'O' corresponds to Cv (mass basis)
            gamma = cp_v / cv_v
            
            # Gas data
            molar_mass = PropsSI('MOLAR_MASS', fluid_string)
            p_crit = PropsSI('P_CRITICAL', fluid_string)
            
            R_univ = 8.31446
            R_spec = R_univ / molar_mass

            return {
                'rho_l': rho_l, 'rho_v': rho_v, 'mu_l': mu_l, 'mu_v': mu_v,
                'sigma': sigma, 'h_fg': h_fg, 'k_l': k_l, 'p_sat': p_sat,
                'cp_l': cp_l, 'gamma': gamma,
                'R_spec': R_spec, 'p_crit': p_crit, 'molar_mass': molar_mass
            }
        except Exception as e:
            raise ValueError(f"CoolProp Error for {self.name} at {T:.1f} K: {str(e)}")

# ==============================================================================
# PART 3: MECHANICAL MODEL (PRESSURE BALANCE)
# ==============================================================================

class RotatingHeatPipeModel:
    """
    Models the hydrodynamic processes in the rotating heat pipe,
    especially the pressure balance between driving and resisting forces.
    """
    def __init__(self, params: HeatPipeParameters):
        self.p = params
        self.gravity = 9.81

    def calc_omega(self, rpm):
        """Converts rotational speed [RPM] to angular velocity [rad/s]."""
        return rpm * (2 * np.pi / 60)

    def calc_film_state(self, mass_kg, rho_l, rpm):
        """
        Analyzes the fill state of the heat pipe.
        Calculates whether the fluid is fully bound in the wick or if a
        free film or pool is formed.
        """
        if rho_l <= 0: return 0, 0, "Error"
        vol_liq = mass_kg / rho_l
        r_wick_outer = self.p.d_in / 2 + self.p.wick_thickness
        vol_void_total = self.p.wick_area * self.p.length * self.p.porosity
        saturation = (vol_liq / vol_void_total) * 100.0
        
        vol_liq_effective = vol_liq / self.p.porosity
        term = vol_liq_effective / (np.pi * self.p.length)
        
        # Check for overfilling (pool formation in the inner space)
        if (r_wick_outer**2 - term) < 0:
            delta = self.p.wick_thickness + 0.0005
            status = "OVERFILLED (Pool formation)"
        else:
            r_inner_fluid = np.sqrt(r_wick_outer**2 - term)
            delta = r_wick_outer - r_inner_fluid
            if delta < self.p.wick_thickness * 0.95:
                status = "UNDERFILLED (Dry-out risk)"
            elif delta > self.p.wick_thickness * 1.05:
                status = "OVERFILLED"
            else:
                status = "OPTIMAL"
        return delta, saturation, status

    def force_centrifugal(self, rpm, rho_liq):
        """
        Calculates the driving pressure buildup from centrifugal force.
        This is the main driving mechanism in rotating heat pipes with a cone.
        """
        omega = self.calc_omega(rpm)
        alpha = self.p.get_tilt_rad()
        
        # The radius of the fluid is calculated from axis offset + half pipe diameter
        r_cond = self.p.r_min + (self.p.d_in / 2.0)
        
        # Projection of the axial length onto the radius increase through the cone
        dr = self.p.length * np.sin(alpha)
        r_evap = r_cond + dr
        
        # Integration of the centrifugal force over the radius: 0.5 * rho * w^2 * (r_max^2 - r_min^2)
        dp = 0.5 * rho_liq * (omega**2) * (r_evap**2 - r_cond**2)
        return dp

    def force_capillary_max(self, sigma):
        """Calculates the maximum capillary pressure based on the pore radius."""
        return 2 * sigma / self.p.pore_radius

    def loss_vapor_advanced(self, m_dot, rho_vap, mu_vap, rpm):
        """
        Calculates the pressure drop in the vapor flow.
        Takes friction effects and Coriolis forces (via Rossby number)
        into account at high rotational speeds.
        """
        if m_dot <= 1e-9: return 0.0
        r_v = self.p.d_in / 2.0
        v = m_dot / (rho_vap * (np.pi * r_v**2))
        omega = self.calc_omega(rpm)
        
        # Correction factor for rotating flows based on the Rossby number
        if omega > 1.0 and v > 0:
            Ro = v / (omega * self.p.d_in)
            coriolis_factor = 1.0 + ROSSBY_FACTOR_VAPOR * (Ro + 0.05)**(-0.5)
        else:
            coriolis_factor = 1.0
            
        mu_eff = mu_vap * coriolis_factor
        
        # Determination of the flow regime (laminar/turbulent)
        Re = (rho_vap * v * self.p.d_in) / mu_eff
        if Re < 2300:
            dp = (8 * mu_eff * self.p.l_eff * m_dot) / (np.pi * rho_vap * r_v**4)
        else:
            f = 0.3164 * (Re**(-0.25)) # Blasius equation for turbulent flow
            dp = f * (self.p.l_eff / self.p.d_in) * 0.5 * rho_vap * v**2
        return dp

    def loss_liquid_hybrid(self, m_dot, rho_liq, mu_liq, rpm):
        """
        Calculates the pressure drop in the liquid return flow.
        Uses a parallel model of Darcy flow (through the wick) and
        film flow (over the wick) if there is an excess of fluid.
        """
        if m_dot <= 1e-9: return 0.0, "Static"

        omega = self.calc_omega(rpm)
        r_avg = self.p.d_in / 2.0
        
        # Rossby correction for liquid viscosity
        v_approx = m_dot / (rho_liq * self.p.wick_area)
        if omega > 1.0 and v_approx > 0:
            Ro_l = v_approx / (omega * (self.p.d_in/2))
            visc_factor_l = 1.0 + ROSSBY_FACTOR_LIQUID * (Ro_l + 0.1)**(-0.5)
        else:
            visc_factor_l = 1.0
            
        mu_eff_liq = mu_liq * visc_factor_l
        
        # Effective acceleration (Centrifugal or Gravity)
        if omega < 1.0:
            g_eff = 9.81
        else:
            g_eff = omega**2 * r_avg
        
        # 1. Conductance for flow through the wick (Darcy's Law)
        G_wick = (self.p.permeability * self.p.wick_area * rho_liq) / (mu_eff_liq * self.p.l_eff)
        
        # 2. Calculation of theoretical film thickness at total mass flow (Nusselt approach)
        term = (3 * mu_eff_liq * m_dot) / ((rho_liq**2) * g_eff * 2 * np.pi * r_avg)
        delta_theor = term**(1/3)
        
        if delta_theor > self.p.wick_thickness:
            # Fluid exits the wick -> Parallel film flow
            delta_excess = delta_theor - self.p.wick_thickness
            
            # Conductance for film flow
            G_film = (rho_liq * 2 * np.pi * (self.p.d_in/2.0) * delta_excess**3) / (3 * mu_eff_liq * self.p.l_eff)
            
            G_total = G_wick + G_film
            mode = "Parallel (Wick+Film)"
        else:
            # Fluid only flows in the wick
            G_total = G_wick
            mode = "Darcy (Wick)"
            
        dp = m_dot / G_total
        return dp, mode

    def loss_coriolis(self, m_dot, rpm, rho_liq):
        """
        Placeholder for explicit Coriolis losses.
        (Currently implicitly accounted for via Rossby factors in viscosity).
        """
        return 0.0

    def force_gravity(self, rho_liq):
        """Calculates the influence of gravity based on global inclination."""
        beta = self.p.get_global_inclination_rad()
        h = self.p.length * np.sin(beta)
        return rho_liq * self.gravity * h

    def check_balance(self, dp_k, dp_omega, dp_v, dp_l, dp_grav, dp_cor):
        """
        Checks the system's pressure balance.
        Driving forces: Capillary pressure + Centrifugal force
        Resisting forces: Pressure losses (Vapor/Liquid) + Gravity + Coriolis
        """
        drive = dp_k + dp_omega
        resist = dp_v + dp_l + dp_grav + dp_cor
        return drive - resist

# ==============================================================================
# PART 3b: THERMAL RESISTANCE MODEL
# ==============================================================================

class ThermalNetwork:
    """
    Calculates thermal resistances and heat transfer coefficients (HTC)
    in the system.
    """
    def __init__(self, model: RotatingHeatPipeModel, fluid: WorkingFluid):
        self.model = model
        self.fluid = fluid

    def get_accel(self, rpm, diameter):
        """Calculates radial centrifugal acceleration."""
        omega = self.model.calc_omega(rpm)
        r = diameter / 2.0
        return omega**2 * r

    def calc_h_natural_convection(self, props, rpm, d_in, dT, L_char):
        """
        Calculates the HTC for natural convection in a strong centrifugal field.
        Uses correlations for rotating systems (Rayleigh number).
        """
        a_c = self.get_accel(rpm, d_in)
        
        rho = props['rho_l']
        mu = props['mu_l']
        k = props['k_l']
        cp = props['cp_l']
        
        # Volumetric expansion coefficient (approximation)
        beta = 0.001 

        if L_char <= 0: L_char = 1e-5
        
        # Calculation of dimensionless numbers
        Pr = (mu * cp) / k
        Gr = (rho**2 * a_c * beta * dT * L_char**3) / (mu**2)
        Ra = Gr * Pr
        
        if Ra < 1e-9: return 100.0 # Minimum value to avoid singularities
        
        # Song/Marto correlation for rotating systems
        Nu_n = 0.133 * (Ra**0.375)
        h_nat = (Nu_n * k) / L_char
        return h_nat

    def calc_h_cond_film_exact(self, props, rpm, m_dot):
        """
        Calculates the heat transfer coefficient (HTC) during condensation.
        Accounts for film thinning due to centrifugal forces (drainage).
        """
        omega = self.model.calc_omega(rpm)
        rho_l = props['rho_l']
        mu_l = props['mu_l']
        k_l = props['k_l']
        
        r_avg = self.model.p.d_in / 2.0
        alpha = self.model.p.get_tilt_rad() # Cone angle
        sin_alpha = np.sin(alpha)
        
        # Limit angle from which the pumping effect becomes dominant
        TRANSITION_ANGLE = 0.0008 
        
        if sin_alpha < 1e-6: sin_alpha = 1e-6 

        # Determining the base HTC at a standstill
        if alpha < TRANSITION_ANGLE:
            # 0 degrees: Very good contact at standstill (saturation)
            h_static = 25000.0
        else:
            # > 0 degrees: Fluid gathers at the bottom due to gravity, lower HTC at the top
            h_static = 10600.0

        if m_dot <= 1e-9: return h_static

        omega_calc = max(0.1, omega) 
        
        # Nusselt film thickness calculation under centrifugal force
        numerator = 3 * mu_l * m_dot
        denominator = 2 * np.pi * r_avg * (rho_l**2) * (omega_calc**2) * r_avg * sin_alpha
        delta_flow = (numerator / denominator)**(1.0/3.0) 
        
        # Limitation for pool formation at 0 degrees
        vol_void = self.model.p.wick_area * self.model.p.length * self.model.p.porosity
        vol_fluid_cond = vol_void * 0.5 
        A_inner = np.pi * self.model.p.d_in * self.model.p.length
        delta_pool_limit = vol_fluid_cond / A_inner
        
        # HTC for completely flooded state
        h_pool_saturated = max(k_l / delta_pool_limit, 10000.0)

        # Case differentiation for behavior during rotation
        if alpha < TRANSITION_ANGLE:
            # Case 0 degrees: Degradation due to "flooding" (centrifugal force pushes fluid into structure)
            r_rotation = self.model.p.r_min + (self.model.p.d_in / 2.0)
            g_force = (omega**2 * r_rotation) / 9.81
            decay = 1.0 - np.exp(-g_force / 150.0) 
            h_combined = h_static * (1.0 - decay) + h_pool_saturated * decay
            
        else:
            # Case > 0 degrees: Improvement due to "drainage" (film becomes thinner)
            delta = delta_flow
            delta = max(delta, 1e-7) # Physically minimum film thickness
            h_rot = (k_l / delta) * 0.7 # Correction for shear effects
            
            # Vectorial addition
            h_combined = (h_static**2 + h_rot**2)**0.5
        
        return h_combined

    def calc_resistances(self, T_op, rpm, q_in_watts=100.0):
        """
        Main function for calculating the thermal network.
        Determines total resistance and effective thermal conductivity.
        """
        props = self.fluid.get_properties(T_op)
        p = self.model.p
        omega = self.model.calc_omega(rpm)
        
        # 1. Calculation of mass flow from thermal load
        m_dot = q_in_watts / props['h_fg']
        
        # Consideration of "Pool Blocking" in the condenser (Geometric overfilling)
        vol_void = p.wick_area * p.length * p.porosity
        mass_internal = vol_void * props['rho_l']
        delta_calc, saturation_pct, _ = self.model.calc_film_state(mass_internal, props['rho_l'], rpm)
        
        # Determining the active condenser length (reduction due to sump formation)
        l_cond_active = p.l_cond
        l_pool = 0.0
        
        if saturation_pct > 100.0:
            vol_excess = max(0.0, (saturation_pct - 100.0) / 100.0) * vol_void
            cross_section_inner = np.pi * (p.d_in / 2.0)**2
            l_pool = vol_excess / cross_section_inner
            l_pool = min(l_pool, p.l_cond * 0.99)
            l_cond_active = p.l_cond - l_pool

        # 2. Condenser HTC (Using exact film model)
        h_cond = self.calc_h_cond_film_exact(props, rpm, m_dot)
        
        # 3. Evaporator HTC (Mixed convection)
        dT_guess = 5.0 # Starting value for iteration
        h_evap = 500.0
        
        k_s = p.k_solid
        k_l = props['k_l']
        eps = p.porosity
        
        # Effective conductivity of wick material
        num = k_l + 2*k_s - 2*eps*(k_s - k_l)
        den = k_l + 2*k_s + eps*(k_s - k_l)
        k_wick_eff = k_s * (num / den)
        
        # Determination of characteristic length for dimensionless numbers
        if delta_calc > p.wick_thickness:
             L_char = delta_calc
        else:
             L_char = p.wick_thickness
             
        g_acc = omega**2 * (p.d_in / 2.0)
        
        # Wetting model and sinter influence (Physical approach)
        # Wetting calculation dependent on hydrostatics and capillary force
        r_rot_center = self.model.p.r_min + (self.model.p.d_in / 2.0)
        a_centrifugal = omega**2 * r_rot_center
        
        # Hydrostatic pressure difference across the pipe
        dp_hydro_trans = props['rho_l'] * a_centrifugal * self.model.p.d_in
        dp_cap_max = 2.0 * props['sigma'] / self.model.p.pore_radius

        # Wetting factor (Transversal Dry-Out)
        wetting_factor = 1.0 

        if dp_hydro_trans > dp_cap_max:
            # Capillary force insufficient to distribute fluid around circumference against centrifugal force
            base_pool_fraction = 0.35 
            climb_ratio = dp_cap_max / dp_hydro_trans
            wetting_factor = base_pool_fraction + (1.0 - base_pool_fraction) * climb_ratio
            wetting_factor = max(0.3, wetting_factor)
        
        # Influence of axial component (inclination/cone) on wetting
        boost_angle = self.model.p.get_tilt_rad()
        
        if boost_angle < 0.001 and abs(self.model.p.inclination_deg) < 89.0:
             boost_angle = np.radians(self.model.p.inclination_deg)

        if boost_angle > 0.001 and omega > 50.0:
            axial_drive = (omega**2 * r_rot_center * np.sin(boost_angle))
            reflood_factor = 1.0 + (axial_drive / 500.0) 
            wetting_factor = wetting_factor * reflood_factor
            
            # Consideration of "flow-through effect" in porous sintered structures
            if self.model.p.porosity < 0.9 and self.model.p.porosity > 0.1:
                sinter_boost = 1.0 + (axial_drive / 200.0) * 0.8 
                wetting_factor = wetting_factor * sinter_boost
            else:
                wetting_factor = min(1.0, wetting_factor)

        else:
            wetting_factor = min(1.0, wetting_factor)

        # Calculation of effective evaporator area
        fill_ratio = min(1.0, max(0.0, saturation_pct / 100.0))
        A_evap_effective = (np.pi * self.model.p.d_in * self.model.p.l_evap) * wetting_factor * fill_ratio
        
        # Iterative calculation of heat transfer in the evaporator
        for _ in range(3):
            # A) Boiling (Cooper correlation)
            pr = props['p_sat'] / props['p_crit']
            pr = np.clip(pr, 0.001, 0.99)
            M_g_mol = props['molar_mass'] * 1000.0
            
            q_flux = max(q_in_watts, 0.1) / max(A_evap_effective, 1e-6)
            
            h_boiling = 55.0 * (pr**0.12) * ((-np.log10(pr))**(-0.55)) * (M_g_mol**(-0.5)) * (q_flux**0.67)
            
            # B) Natural Convection (Song et al.)
            h_nat = self.calc_h_natural_convection(props, rpm, p.d_in, dT_guess, L_char)
            
            # C) Forced Convection (Film flow)
            h_forced = k_l / L_char 
            
            # Combined model (Mixed convection)
            h_mixed = (h_forced**3.5 + h_nat**3.5)**(1/3.5)
            
            # Damping of boiling due to high g-forces (Boiling suppression)
            g_critical = 250.0 * 9.81 
            g_width = 50.0 * 9.81 
            
            if g_acc > 0:
                S_suppression = 1.0 / (1.0 + np.exp((g_acc - g_critical) / g_width))
            else:
                S_suppression = 1.0
            
            # Blending between boiling and convection
            n_blend = 3.0
            h_evap = ( (h_boiling * S_suppression)**n_blend + h_mixed**n_blend )**(1.0 / n_blend)
                
            dT_new = q_flux / h_evap
            dT_guess = 0.5 * dT_guess + 0.5 * dT_new
            
        # 4. Assembly of resistances (Resistor Network)
        r_o = p.d_out / 2
        r_i_wick = p.d_in / 2
        r_if_wall = r_i_wick + p.wick_thickness
        
        # Heat conduction through the pipe wall
        R_wall_evap = np.log(r_o / r_if_wall) / (2 * np.pi * p.l_evap * p.k_solid)
        R_wall_cond = np.log(r_o / r_if_wall) / (2 * np.pi * p.l_cond * p.k_solid)
        
        # Heat conduction through the wick
        R_wick_evap = np.log(r_if_wall / r_i_wick) / (2 * np.pi * p.l_evap * k_wick_eff)
        R_wick_cond = np.log(r_if_wall / r_i_wick) / (2 * np.pi * p.l_cond * k_wick_eff)
        
        # Phase transition resistance evaporator
        R_film_evap = 1.0 / (h_evap * A_evap_effective)
        
        # Phase transition resistance condenser (Differentiating Active vs. Pool)
        if l_cond_active > 1e-6:
            r_cond_start = p.d_in / 2.0 
            r_cond_end = r_cond_start + p.l_cond * np.tan(p.get_tilt_rad())
            s_slant = np.sqrt(p.l_cond**2 + (r_cond_end - r_cond_start)**2)
            A_cond_exact = np.pi * (r_cond_start + r_cond_end) * s_slant
            
            ratio_active = l_cond_active / p.l_cond
            A_active = A_cond_exact * ratio_active
            
            R_film_active = 1.0 / (h_cond * A_active)
        else:
            R_film_active = 1e9 

        # Resistance of the sump area (Pool)
        if l_pool > 1e-6:
            h_pool_conduction = k_l / (p.d_in / 4.0)
            R_pool = 1.0 / (h_pool_conduction * (np.pi * p.d_in * l_pool))
        else:
            R_pool = 1e9

        # Parallel connection in the condenser
        R_film_cond = 1.0 / ( (1.0/R_film_active) + (1.0/R_pool) )
        
        R_vapor = 0.0001
        
        # Sum of radial resistances
        R_radial = (R_wall_evap + R_wick_evap + R_film_evap +
                    R_vapor +
                    R_film_cond + R_wick_cond + R_wall_cond)
                    
        # Consideration of axial conduction through pipe wall (parallel path)
        A_wall = np.pi * (r_o**2 - r_if_wall**2)
        if A_wall > 1e-9:
            R_axial = p.length / (p.k_solid * A_wall)
            R_total = 1.0 / (1.0/R_radial + 1.0/R_axial)
        else:
            R_total = R_radial
            
        A_cross = np.pi * (r_o**2)
        k_pipe_eff = p.length / (A_cross * R_total)
        
        return R_total, k_pipe_eff

# ==============================================================================
# PART 4: PERFORMANCE LIMITS
# ==============================================================================

class PerformanceLimits:
    """
    Calculates the physical operating limits of the heat pipe.
    """
    def __init__(self, model: RotatingHeatPipeModel, fluid: WorkingFluid):
        self.model = model
        self.fluid = fluid

    def limit_capillary(self, T_op, rpm):
        """
        Calculates the capillary limit (hydrodynamic limit).
        Iteratively searches for the heat load Q where driving pressures
        equal resistances.
        """
        props = self.fluid.get_properties(T_op)
        def residual(Q):
            if Q <= 0: return 1.0
            m_dot = Q / props['h_fg']
            dp_k = self.model.force_capillary_max(props['sigma'])
            dp_omega = self.model.force_centrifugal(rpm, props['rho_l'])
            dp_v = self.model.loss_vapor_advanced(m_dot, props['rho_v'], props['mu_v'], rpm)
            
            dp_l, _ = self.model.loss_liquid_hybrid(m_dot, props['rho_l'], props['mu_l'], rpm)
            
            dp_cor = self.model.loss_coriolis(m_dot, rpm, props['rho_l'])
            dp_grav = self.model.force_gravity(props['rho_l'])
            
            # Check for transversal dry-out (centrifugal force vs. capillary force at circumference)
            omega = self.model.calc_omega(rpm)
            r_pipe = self.model.p.d_in / 2.0
            r_rotation_center = self.model.p.r_min + r_pipe
            accel_field = omega**2 * r_rotation_center
            
            dp_hydro_transversal = props['rho_l'] * accel_field * (2 * r_pipe)
            dp_cap_max = 2 * props['sigma'] / self.model.p.pore_radius 
            
            # Reduction of available pumping pressure if capillary is loaded by lateral acceleration
            if dp_hydro_transversal > dp_cap_max:
                dp_k_available = max(0.0, dp_cap_max - dp_hydro_transversal)
                dp_k = dp_k_available 
            
            return self.model.check_balance(dp_k, dp_omega, dp_v, dp_l, dp_grav, dp_cor)

        # Bisection method for finding roots
        low, high = 0.0, 500000.0
        if residual(high) > 0: return high
        for _ in range(60):
            mid = (low + high) / 2
            if abs(residual(mid)) < 1.0: return mid
            if residual(mid) > 0: low = mid
            else: high = mid
        return mid

    def limit_boiling(self, T_op, rpm):
        """
        Calculates the boiling limit.
        Limit at which bubble formation in the wick blocks liquid flow.
        Accounts for high g-forces suppressing boiling.
        """
        props = self.fluid.get_properties(T_op)
        
        r_rot = self.model.p.d_in / 2.0
        omega = self.model.calc_omega(rpm)
        accel = omega**2 * r_rot
        g_force = accel / 9.81
        
        # Above ~250g convection dominates, classic boiling limit is no longer relevant
        if g_force > 250.0:
            return 1e9 

        k_s = self.model.p.k_solid
        k_l = props['k_l']
        eps = self.model.p.porosity
        num = k_l + 2*k_s - 2*eps*(k_s - k_l)
        den = k_l + 2*k_s + eps*(k_s - k_l)
        k_eff = k_s * (num / den)
        r_n = NUCLEATION_RADIUS_DEFAULT 
        L_evap = self.model.p.l_evap
        r_o = self.model.p.d_out / 2
        r_i = self.model.p.d_in / 2
        
        dp_rot_hydro = 0.5 * props['rho_l'] * omega**2 * (r_o**2 - r_i**2)
        sigma_term = 2 * props['sigma'] / r_n
        pressure_resisting_bubbles = sigma_term + dp_rot_hydro
        
        r_wick_interface = r_i + self.model.p.wick_thickness
        geo_factor = np.log(r_wick_interface / r_i)
        
        if geo_factor < 1e-9: geo_factor = 1e-9

        numerator = 2 * np.pi * L_evap * k_eff * T_op * pressure_resisting_bubbles
        denominator = props['h_fg'] * props['rho_v'] * geo_factor
        if denominator == 0: return 1e9
        return numerator / denominator

    def limit_sonic(self, T_op):
        """
        Calculates the sonic limit.
        Vapor velocity reaches Mach 1 at evaporator exit.
        """
        props = self.fluid.get_properties(T_op)
        r_v = self.model.p.d_in / 2
        A_v = np.pi * r_v**2
        
        gamma = props['gamma'] 
        
        a = np.sqrt(gamma * props['R_spec'] * T_op)
        Q_s = A_v * props['rho_v'] * props['h_fg'] * a * np.sqrt(1/(2*(gamma+1)))
        return Q_s

    def limit_entrainment(self, T_op):
        """
        Calculates the entrainment limit.
        High vapor velocities tear liquid droplets from the wick.
        """
        props = self.fluid.get_properties(T_op)
        A_v = np.pi * (self.model.p.d_in/2)**2
        l_char = self.model.p.pore_radius * 2
        term = (props['sigma'] * props['rho_v']) / l_char
        Q_ent = A_v * props['h_fg'] * np.sqrt(term)
        return Q_ent

    def limit_viscous(self, T_op):
        """
        Calculates the viscous limit.
        Relevant at very low temperatures (high vapor viscosity).
        """
        props = self.fluid.get_properties(T_op)
        d_v = self.model.p.d_in
        A_v = np.pi * (d_v / 2)**2
        numerator = A_v * ((d_v/2)**2) * props['h_fg'] * props['rho_v'] * props['p_sat']
        denominator = 16 * props['mu_v'] * self.model.p.l_eff
        return numerator / denominator

    def get_all_limits(self, T_op, rpm):
        """Combines all performance limits into a dictionary."""
        return {
            "Capillary": self.limit_capillary(T_op, rpm),
            "Sonic": self.limit_sonic(T_op),
            "Entrainment": self.limit_entrainment(T_op),
            "Boiling": self.limit_boiling(T_op, rpm),
            "Viscous": self.limit_viscous(T_op)
        }

# ==============================================================================
# PART 5: GUI AND VISUALIZATION
# ==============================================================================

class HeatPipeGUI:
    """
    Graphical user interface (Tkinter) to control the simulation.
    Includes input masks, real-time validation, plotting, and export.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Rotating Heat Pipes - Simulation Tool ")
        self.root.geometry("1200x900") 
        
        # Configuration for save/load
        self.settings_file = "rhp_settings.json"
        self.saved_data = {}
        self.load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Design settings
        style = ttk.Style()
        try:
            style.theme_use('clam') 
        except:
            pass 
            
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333333")
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#2c3e50")
        
        main_container = ttk.Frame(root, padding="15")
        main_container.pack(fill="both", expand=True)
        
        # Header area
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(header_frame, text="Calculation Model for Rotating Heat Pipes", style="Header.TLabel").pack(side="left")
        ttk.Button(header_frame, text="ℹ Assumptions", command=self.show_assumptions, width=12).pack(side="right")

        # Action bar
        action_bar = ttk.LabelFrame(main_container, text=" Actions ", padding="5")
        action_bar.pack(fill="x", pady=(0, 10))

        ttk.Button(action_bar, text="▶ Calculate & Plot", command=self.run_analysis).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="⚙ Optimization", command=self.run_optimization).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="📈 Pressure Profile", command=self.run_pressure_plot).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="⏱ Transient (Startup)", command=self.run_transient).pack(side="left", padx=5, pady=2)
        
        ttk.Button(action_bar, text="💾 Export Excel", command=self.export_to_excel).pack(side="left", padx=(20, 5), pady=2)

        self.vars = {}
        self.widgets = {}
        
        self.current_fig = None

        # Scrollable main area
        scroll_frame = ttk.Frame(main_container)
        scroll_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(scroll_frame, background="#f0f0f0", highlightthickness=0)
        
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        content_frame = ttk.Frame(canvas, style="TFrame")

        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Mouse wheel support
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Two-column grid layout
        left_col = ttk.Frame(content_frame)
        left_col.grid(row=0, column=0, sticky="nw", padx=(0, 20))
        
        right_col = ttk.Frame(content_frame)
        right_col.grid(row=0, column=1, sticky="nsew", rowspan=2)

        content_frame.columnconfigure(1, weight=1) 
        right_col.columnconfigure(0, weight=1)     
        right_col.rowconfigure(2, weight=1)        

        # --- LEFT COLUMN: INPUTS ---

        # 1. Operating Parameters
        group_ops = ttk.LabelFrame(left_col, text=" 1. Operating Parameters ", padding="15 10")
        group_ops.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_ops, "Fluid", "Water", ["Water", "Methanol", "Ammonia"])
        self.create_input_row(group_ops, "Heat Load [W]", "100.0", None) 
        self.create_input_row(group_ops, "Speed [RPM]", "24000", None)
        self.create_input_row(group_ops, "Temperature [°C]", "60.0", None, info_cmd=self.show_temp_info)

        # 2. Geometry
        group_geo = ttk.LabelFrame(left_col, text=" 2. Geometry (Pipe) ", padding="15 10")
        group_geo.pack(fill="x", pady=(0, 15))
        
        btn_geo_info = ttk.Button(group_geo, text="ℹ Geometry Help", command=self.show_geo_info, width=15)
        btn_geo_info.pack(anchor="e", pady=(0, 10))

        self.create_input_row(group_geo, "Length L (Total) [mm]", "250.0", None)
        self.create_input_row(group_geo, "L_Evaporator [mm]", "50.0", None)
        self.create_input_row(group_geo, "L_Adiabatic [mm]", "150.0", None)
        self.create_input_row(group_geo, "L_Condenser [mm]", "50.0", None)
        ttk.Separator(group_geo, orient="horizontal").pack(fill="x", pady=10)
        self.create_input_row(group_geo, "D_outer [mm]", "12.0", None)
        self.create_input_row(group_geo, "D_inner [mm]", "8.0", None)
        
        self.create_input_row(group_geo, "Wall thickness (auto) [mm]", "1.0", None)

        # 3. Structure & Material
        group_struct = ttk.LabelFrame(left_col, text=" 3. Wick & Material ", padding="15 10")
        group_struct.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_struct, "Wick thickness [mm]", "1.0", None)
        self.create_input_row(group_struct, "Pore radius [µm]", "50.0", None)
        self.create_input_row(group_struct, "Permeability [1e-10 m²]", "1.0", None)
        self.create_input_row(group_struct, "Wick porosity [0-1]", "0.5", None)
        self.create_input_row(group_struct, "Thermal cond. wall [W/mK]", "380.0", None)
        
        # Automatic calculation of wall thickness
        if "Wall thickness (auto) [mm]" in self.widgets:
            w_widget = self.widgets["Wall thickness (auto) [mm]"]
            
            style.map("Grey.TEntry", 
                      fieldbackground=[("readonly", "#e0e0e0")],  
                      foreground=[("readonly", "#333333")])       
            
            w_widget.configure(style="Grey.TEntry", state="readonly", cursor="no")
        
        self.vars["D_outer [mm]"].trace("w", self.auto_calc_wall)
        self.vars["D_inner [mm]"].trace("w", self.auto_calc_wall)
        self.vars["Wick thickness [mm]"].trace("w", self.auto_calc_wall)


        # 4. Rotation Setup
        group_rot = ttk.LabelFrame(left_col, text=" 4. Rotation Setup ", padding="15 10")
        group_rot.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_rot, "Rotation mode", "Eccentric", ["Eccentric", "Centered"])
        self.create_input_row(group_rot, "Axis offset (r_min) [mm]", "15.0", None)
        self.create_input_row(group_rot, "Cone angle [deg]", "3.0", None)
        self.create_input_row(group_rot, "Inclination angle (Global) [deg]", "90.0", None)
        
        self.vars["Rotation mode"].trace("w", self.toggle_mode)

        # --- RIGHT COLUMN: VISUALIZATION & RESULTS ---
        
        # Geometry Preview
        self.sketch_frame = ttk.LabelFrame(right_col, text=" Geometry Preview (for rough visualization only) ", padding="10")
        self.sketch_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.geo_canvas = tk.Canvas(self.sketch_frame, width=500, height=120, bg="white", highlightthickness=0)
        self.geo_canvas.pack(fill="both", expand=True)

        # Results Display (Table)
        self.group_res = ttk.LabelFrame(right_col, text=" Results & Analysis ", padding="15 10")
        self.group_res.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        res_container = ttk.Frame(self.group_res)
        res_container.pack(fill="both", expand=True)

        btn_copy = ttk.Button(res_container, text="📋", width=3, command=self.copy_tree_value)
        btn_copy.pack(side="right", fill="y", padx=(2,0)) 

        text_scroll = ttk.Scrollbar(res_container)
        text_scroll.pack(side="right", fill="y")

        columns = ("Parameter", "Value", "Unit")
        self.res_tree = ttk.Treeview(res_container, columns=columns, show="headings", height=15, 
                                     yscrollcommand=text_scroll.set)
        
        self.res_tree.heading("Parameter", text="Parameter")
        self.res_tree.heading("Value", text="Value")
        self.res_tree.heading("Unit", text="Unit")
        
        self.res_tree.column("Parameter", width=200, anchor="w")
        self.res_tree.column("Value", width=100, anchor="center")
        self.res_tree.column("Unit", width=80, anchor="center")
        
        self.res_tree.pack(side="left", fill="both", expand=True)
        
        text_scroll.config(command=self.res_tree.yview)

        # Plot Area
        self.plot_frame = ttk.LabelFrame(right_col, text=" Diagrams ", padding="5")
        self.plot_frame.grid(row=2, column=0, sticky="nsew")
        
        ttk.Label(self.plot_frame, text="Plots will appear here after calculation", foreground="grey").pack(pady=20)
        
        ttk.Frame(left_col, height=50).pack(fill="x")

        # Initialization of traces for real-time validation and sketch update
        sketch_vars = ["Length L (Total) [mm]", "L_Evaporator [mm]", "L_Adiabatic [mm]", "L_Condenser [mm]", 
                       "D_outer [mm]", "D_inner [mm]", "Wick thickness [mm]",
                       "Axis offset (r_min) [mm]", "Cone angle [deg]", "Inclination angle (Global) [deg]"]
        
        for sv in sketch_vars:
            if sv in self.vars:
                self.vars[sv].trace("w", self.update_sketch)
                self.vars[sv].trace("w", self.perform_visual_validation)

        if "Length L (Total) [mm]" in self.vars:
            self.vars["Length L (Total) [mm]"].trace("w", self.auto_adjust_adiabat)

        self.root.after(100, self.update_sketch)
        self.root.after(100, self.perform_visual_validation)


    def copy_tree_value(self):
        """Copies the contents of the results table to the clipboard."""
        try:
            lines = ["Parameter\tValue\tUnit"]
            for child in self.res_tree.get_children():
                item = self.res_tree.item(child)
                vals = item['values']
                if vals:
                    line = "\t".join([str(v) for v in vals])
                    lines.append(line)
            
            full_text = "\n".join(lines)
            
            self.root.clipboard_clear()
            self.root.clipboard_append(full_text)
            self.root.update()
            messagebox.showinfo("Copied", "Table has been copied to clipboard.")
        except Exception:
            pass

    def create_input_row(self, parent, lbl, val, opts, info_cmd=None):
        """Creates a standardized input row with label and input field."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        
        ttk.Label(row, text=lbl, width=30).pack(side="left")
        
        if info_cmd:
            ttk.Button(row, text="?", width=3, command=info_cmd).pack(side="right", padx=(5,0))

        initial_value = val
        if lbl in self.saved_data:
            initial_value = self.saved_data[lbl]

        if opts:
            var = tk.StringVar(value=initial_value)
            cb = ttk.Combobox(row, textvariable=var, values=opts, state="readonly", width=18)
            cb.pack(side="right", padx=5)
            self.vars[lbl] = var
            self.widgets[lbl] = cb
        else:
            var = tk.StringVar(value=initial_value)
            entry = ttk.Entry(row, textvariable=var, width=20)
            entry.pack(side="right", padx=5)
            self.vars[lbl] = var
            self.widgets[lbl] = entry

    def load_settings(self):
        """Loads user settings from a JSON file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    self.saved_data = json.load(f)
            except:
                pass 

    def on_closing(self):
        """Saves current inputs when closing the program."""
        try:
            data = {k: v.get() for k, v in self.vars.items()}
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=4)
        except:
            pass
        self.root.destroy()
    
    def perform_visual_validation(self, *args):
        """Performs a visual validation of the inputs (color coding for errors)."""
        COLOR_OK = "black"
        COLOR_ERR = "red"

        # Length validation
        try:
            L = float(self.vars["Length L (Total) [mm]"].get())
            l_e = float(self.vars["L_Evaporator [mm]"].get())
            l_a = float(self.vars["L_Adiabatic [mm]"].get())
            l_c = float(self.vars["L_Condenser [mm]"].get())

            if abs(L - (l_e + l_a + l_c)) > 0.1 or l_e < 0 or l_a < 0 or l_c < 0 or L <= 0:
                c_len = COLOR_ERR
            else:
                c_len = COLOR_OK
            
            self.widgets["Length L (Total) [mm]"].configure(foreground=c_len)
            self.widgets["L_Evaporator [mm]"].configure(foreground=c_len)
            self.widgets["L_Adiabatic [mm]"].configure(foreground=c_len)
            self.widgets["L_Condenser [mm]"].configure(foreground=c_len)

        except ValueError:
            pass 

        # Diameter validation
        try:
            da = float(self.vars["D_outer [mm]"].get())
            di = float(self.vars["D_inner [mm]"].get())
            wick = float(self.vars["Wick thickness [mm]"].get())

            if di >= da or (da - di - 2*wick) <= 0 or da <= 0 or di <= 0:
                c_dia = COLOR_ERR
            else:
                c_dia = COLOR_OK
            
            self.widgets["D_outer [mm]"].configure(foreground=c_dia)
            self.widgets["D_inner [mm]"].configure(foreground=c_dia)
            self.widgets["Wick thickness [mm]"].configure(foreground=c_dia)

        except ValueError:
            pass
    
    def auto_adjust_adiabat(self, *args):
        """Automatically adjusts adiabatic zone length when total length is changed."""
        try:
            l_total = float(self.vars["Length L (Total) [mm]"].get())
            l_evap = float(self.vars["L_Evaporator [mm]"].get())
            l_cond = float(self.vars["L_Condenser [mm]"].get())
            
            new_adiabat = l_total - l_evap - l_cond
            
            if new_adiabat >= 0:
                self.vars["L_Adiabatic [mm]"].set(f"{new_adiabat:.1f}")
        except ValueError:
            pass

    def update_sketch(self, *args):
        """Updates geometric preview based on current parameters."""
        try:
            l_evap = float(self.vars["L_Evaporator [mm]"].get())
            l_adiab = float(self.vars["L_Adiabatic [mm]"].get())
            l_cond = float(self.vars["L_Condenser [mm]"].get())
            
            da = float(self.vars["D_outer [mm]"].get())
            di_start = float(self.vars["D_inner [mm]"].get()) 
            wick_thickness = float(self.vars["Wick thickness [mm]"].get())
            
            r_min_val = float(self.vars["Axis offset (r_min) [mm]"].get())
            cone_angle = float(self.vars["Cone angle [deg]"].get())
            inc_val = float(self.vars["Inclination angle (Global) [deg]"].get())

            total_l = l_evap + l_adiab + l_cond
            if total_l <= 0: return

            self.geo_canvas.delete("all")
            c_w = self.geo_canvas.winfo_width()
            c_h = self.geo_canvas.winfo_height()
            if c_w < 10: c_w = 500
            if c_h < 10: c_h = 150 

            margin_x = 50
            margin_y = 20
            
            draw_w = c_w - 2 * margin_x
            scale_x = draw_w / total_l
            
            max_r_total = r_min_val + da/2.0 + (total_l * np.tan(np.radians(cone_angle)))
            scale_y = (c_h * 0.4) / (max_r_total if max_r_total > 0 else da)
            
            scale_y = min(scale_y, 10.0) 

            axis_y = c_h / 2.0

            def get_r_inner(x_mm):
                return (di_start / 2.0) + x_mm * np.tan(np.radians(cone_angle))

            # Draw rotation axis
            self.geo_canvas.create_line(10, axis_y, c_w - 10, axis_y, fill="black", dash=(10, 4), width=1)
            
            current_x_px = margin_x
            current_x_mm = 0.0
            
            sections = [
                (l_cond, "Condenser", "#e3f2fd", "cond"), 
                (l_adiab, "Adiabatic", "#f5f5f5", "adiab"),   
                (l_evap, "Evaporator", "#ffebee", "evap")   
            ]

            for l_sect, name, col, stype in sections:
                w_px = l_sect * scale_x
                
                r_i_start = get_r_inner(current_x_mm)
                r_i_end = get_r_inner(current_x_mm + l_sect)
                
                y_offset_px = r_min_val * scale_y
                
                r_o_px = (da / 2.0) * scale_y
                
                r_wick_start_px = (r_i_start + wick_thickness) * scale_y
                r_wick_end_px   = (r_i_end + wick_thickness) * scale_y
                
                r_i_start_px = r_i_start * scale_y
                r_i_end_px   = r_i_end * scale_y

                cy = axis_y - y_offset_px 

                x1 = current_x_px
                x2 = current_x_px + w_px
                
                # Pipe wall (top and bottom)
                self.geo_canvas.create_rectangle(x1, cy - r_o_px, x2, cy - r_wick_start_px, 
                                                 fill="#90a4ae", outline="black", tags="wall")
                self.geo_canvas.create_rectangle(x1, cy + r_wick_start_px, x2, cy + r_o_px, 
                                                 fill="#90a4ae", outline="black", tags="wall")

                # Wick structure
                pts_wick_top = [
                    x1, cy - r_wick_start_px,
                    x2, cy - r_wick_end_px,
                    x2, cy - r_i_end_px,
                    x1, cy - r_i_start_px
                ]
                self.geo_canvas.create_polygon(pts_wick_top, fill="#cfd8dc", outline="grey")

                pts_wick_bot = [
                    x1, cy + r_wick_start_px,
                    x2, cy + r_wick_end_px,
                    x2, cy + r_i_end_px,
                    x1, cy + r_i_start_px
                ]
                self.geo_canvas.create_polygon(pts_wick_bot, fill="#cfd8dc", outline="grey")

                # Vapor core
                pts_vapor = [
                    x1, cy - r_i_start_px,
                    x2, cy - r_i_end_px,
                    x2, cy + r_i_end_px,
                    x1, cy + r_i_start_px
                ]
                self.geo_canvas.create_polygon(pts_vapor, fill=col, outline="")

                # Vertical separating lines
                self.geo_canvas.create_line(x1, cy - r_o_px, x1, cy + r_o_px, fill="#555")
                self.geo_canvas.create_line(x2, cy - r_o_px, x2, cy + r_o_px, fill="#555")

                if w_px > 30:
                    self.geo_canvas.create_text(x1 + w_px/2, cy, text=name, font=("Segoe UI", 7), fill="#333")

                # Dimensioning
                dim_y = axis_y + 10 
                self.geo_canvas.create_line(x1, dim_y, x2, dim_y, arrow=tk.BOTH, fill="#444", width=1)
                self.geo_canvas.create_text(x1 + w_px/2, dim_y + 8, text=f"{l_sect:.0f}", font=("Segoe UI", 7))

                current_x_px += w_px
                current_x_mm += l_sect

            # Caps (visual closures)
            start_x = margin_x
            end_x = current_x_px
            cy = axis_y - r_min_val * scale_y
            ro = (da/2.0) * scale_y
            
            self.geo_canvas.create_oval(start_x - 5, cy - ro, start_x + 5, cy + ro, fill="#cfd8dc", outline="black")
            r_end_total = ro 
            self.geo_canvas.create_oval(end_x - 5, cy - r_end_total, end_x + 5, cy + r_end_total, fill="#cfd8dc", outline="black")

            # Visualization of the inclination angle
            arrow_start_x = current_x_px 
            arrow_start_y = axis_y
            arrow_len = 40

            self.geo_canvas.create_line(arrow_start_x, arrow_start_y, arrow_start_x + 50, arrow_start_y, 
                                        fill="#888", dash=(4, 2))

            rad = np.radians(inc_val)
            dx = arrow_len * np.cos(rad)
            dy = arrow_len * np.sin(rad) 

            tip_x = arrow_start_x + dx
            tip_y = arrow_start_y - dy

            self.geo_canvas.create_line(arrow_start_x, arrow_start_y, tip_x, tip_y, 
                                        arrow=tk.LAST, width=2, fill="#c0392b")

            self.geo_canvas.create_text(tip_x + 5, tip_y, 
                                        text=f"{inc_val:.1f}°", 
                                        anchor="w", font=("Segoe UI", 8, "bold"), fill="#c0392b")
            
        except Exception as e:
            pass

    
    def auto_calc_wall(self, *args):
        """Automatically calculates wall thickness based on outer, inner diameter and wick thickness."""
        try:
            s_da = self.vars["D_outer [mm]"].get()
            s_di = self.vars["D_inner [mm]"].get()
            s_wick = self.vars["Wick thickness [mm]"].get()
            
            if not s_da or not s_di or not s_wick: return

            da = float(s_da)
            di = float(s_di)
            wick = float(s_wick)
            
            # Formula: D_outer = D_inner + 2*Wick + 2*Wall
            wall = (da - di - 2 * wick) / 2.0
            
            self.vars["Wall thickness (auto) [mm]"].set(f"{wall:.2f}")
            
            if wall <= 0:
                self.widgets["Wall thickness (auto) [mm]"].configure(foreground="red")
            else:
                self.widgets["Wall thickness (auto) [mm]"].configure(foreground="#333333")
                
        except ValueError:
            pass 

    def toggle_mode(self, *args):
        """Enables or disables the axis offset input depending on the rotation mode."""
        mode = self.vars["Rotation mode"].get()
        target_field = "Axis offset (r_min) [mm]"
        
        if target_field not in self.widgets: return
        
        widget = self.widgets[target_field]
        var = self.vars[target_field]
        
        if mode == "Centered":
            var.set("0.0")
            widget.configure(state="disabled")
        else:
            widget.configure(state="normal")
            try:
                if float(var.get()) == 0.0:
                    var.set("15.0")
            except ValueError:
                pass

    def show_temp_info(self):
        """Shows information on critical temperatures of the working fluids."""
        msg = ("Temperature Limit (Critical Point):\n\n"
               "Ammonia: max. 131 °C\n"
               "Methanol: max. 238 °C\n"
               "Water:   max. 326 °C (Code limit)\n\n"
               "Above these values, the fluid is supercritical\n"
               "and vaporization is no longer possible.")
        messagebox.showinfo("Temperature Limits", msg)

    def show_assumptions(self):
        """Shows the assumptions and simplifications underlying the model."""
        info_text = (
            "Main Model Assumptions for Validation:\n\n"
            "1. Thermo-Mechanics:\n"
            "   Decoupled calculation of pressure balance (limits) and thermal resistance.\n\n"
            "2. Wick Structure (Hybrid Approach):\n"
            "   Differentiation between Darcy flow (in sinter) and annular flow (film) at high RPMs.\n\n"
            "3. Phase Transition:\n"
            "   Condensation HTC scales physically correct with Omega^2 (Centrifugal force).\n"
            "   Boiling limit considers rotating hydrostatic pressure.\n"
            "   Addition: Neglect of interfacial shear stress (vapor counterflow) in condensation calculation (Nusselt film theory).\n\n"
            "4. Rotation:\n"
            "   Centrifugal force pumps fluid from condenser (r_min) to evaporator (r_max).\n\n"
            "5. Vapor Flow (Coriolis via approach 2):\n"
            "   Integration of Coriolis effects via the Rossby number directly into the friction factor."
        )
        messagebox.showinfo("Model Basics & Assumptions", info_text)

    def show_geo_info(self):
        """Explains the geometric relationships of the pipe dimensions."""
        msg = (
            "Geometry Plausibility:\n\n"
            "The wall thickness is now automatically calculated:\n"
            "  Wall = (D_outer - D_inner - 2 * Wick) / 2\n\n"
            "Please ensure that D_outer is large enough!"
        )
        messagebox.showinfo("Geometry Formula", msg)

    def check_geometry(self, da, di, t_wick, t_wall):
        """Checks the geometric consistency of the inputs."""
        if t_wall <= 0:
            messagebox.showerror("Geometry Error", 
                                 f"The calculated wall thickness is {t_wall:.2f} mm (<= 0)!\n"
                                 "Please increase D_outer or decrease D_inner/Wick.")
            return False
        return True

    def get_model_objects(self):
        """
        Extracts all input values, converts units, and initializes
        the model objects for the simulation.
        """
        v = {k: val.get() for k, val in self.vars.items()}
        
        da_check = float(v["D_outer [mm]"])
        di_check = float(v["D_inner [mm]"])
        wick_check = float(v["Wick thickness [mm]"])
        wall_check = float(v["Wall thickness (auto) [mm]"])
        
        if not self.check_geometry(da_check, di_check, wick_check, wall_check):
            raise ValueError("Invalid geometry. Aborting.")

        L_mm = float(v["Length L (Total) [mm]"])
        l_evap_mm = float(v["L_Evaporator [mm]"])
        l_adiab_mm = float(v["L_Adiabatic [mm]"])
        l_cond_mm = float(v["L_Condenser [mm]"])

        if abs((l_evap_mm + l_adiab_mm + l_cond_mm) - L_mm) > 0.1:
            msg = (f"Sum error of lengths!\n"
                   f"Evaporator ({l_evap_mm}) + Adiabatic ({l_adiab_mm}) + Condenser ({l_cond_mm})\n"
                   f"= {l_evap_mm + l_adiab_mm + l_cond_mm:.2f} mm\n"
                   f"!= Total length {L_mm} mm\n\n"
                   "Please adjust the partial lengths.")
            messagebox.showerror("Input Error Length", msg)
            raise ValueError("Length sum invalid. Aborting.")

        # Conversion to SI units
        L = L_mm / 1000.0
        l_evap_m = l_evap_mm / 1000.0
        l_adiab_m = l_adiab_mm / 1000.0
        l_cond_m = l_cond_mm / 1000.0

        Da = float(v["D_outer [mm]"]) / 1000.0
        Di = float(v["D_inner [mm]"]) / 1000.0
        r_pore = float(v["Pore radius [µm]"]) * 1e-6
        perm = float(v["Permeability [1e-10 m²]"]) * 1e-10
        wick_thick = float(v["Wick thickness [mm]"]) / 1000.0
        porosity = float(v["Wick porosity [0-1]"])
        k_solid = float(v["Thermal cond. wall [W/mK]"])
        inc_deg = float(v["Inclination angle (Global) [deg]"])
        cone_deg = float(v["Cone angle [deg]"])
        
        r_min_val = float(v["Axis offset (r_min) [mm]"]) / 1000.0
        
        wick_area = np.pi * ((Di/2 + wick_thick)**2 - (Di/2)**2)
        
        params = HeatPipeParameters(
            length=L, d_out=Da, d_in=Di,
            l_evap=l_evap_m, l_adiab=l_adiab_m, l_cond=l_cond_m,
            r_min=r_min_val, cone_angle_deg=cone_deg, inclination_deg=inc_deg,
            pore_radius=r_pore, permeability=perm, wick_area=wick_area,
            wick_thickness=wick_thick, porosity=porosity, k_solid=k_solid
        )
        
        fluid = WorkingFluid(v["Fluid"])
        model = RotatingHeatPipeModel(params)
        limits = PerformanceLimits(model, fluid)
        thermal = ThermalNetwork(model, fluid)
        
        return params, fluid, model, limits, thermal, v

    def export_to_excel(self):
        """Exports simulation results and parameters to an Excel file."""
        if pd is None:
            messagebox.showerror("Library Missing", "Pandas is missing.")
            return
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperature [°C]"]) + 273.15
            rpm_pt = float(v["Speed [RPM]"])
            
            Q_input = float(v["Heat Load [W]"])

            res_pt = limits.get_all_limits(T_k, rpm_pt)
            R_th, k_eff = thermal.calc_resistances(T_k, rpm_pt, q_in_watts=Q_input)
            
            limiting_factor = min(res_pt, key=res_pt.get)
            max_Q = res_pt[limiting_factor]
            
            props = fluid.get_properties(T_k)
            vol_wick = params.wick_area * params.length
            vol_void = vol_wick * params.porosity
            mass_charge_g = vol_void * props['rho_l'] * 1000.0
            mass_calc = mass_charge_g
            delta_film, sat_pct, status_film = model.calc_film_state(mass_calc / 1000.0, props['rho_l'], rpm_pt)

            rpms = np.linspace(0, 30000, 50)
            data_curves = {"RPM": rpms}
            limit_names = ["Capillary", "Boiling", "Sonic", "Entrainment", "Viscous"]
            
            for lname in limit_names:
                vals = []
                for r in rpms:
                    res = limits.get_all_limits(T_k, r)
                    vals.append(res[lname])
                data_curves[lname] = vals
            
            input_list = [{"Parameter": k, "Value": val} for k, val in v.items()]
            df_inputs = pd.DataFrame(input_list)
            
            res_list = [
                {"Result": "Limiting Factor", "Value": limiting_factor},
                {"Result": "Max. Power [W]", "Value": max_Q},
                {"Result": "R_th [K/W] (at load)", "Value": R_th},
                {"Result": "k_eff [W/mK]", "Value": k_eff},
                {"Result": "Fill amount (100%) [g]", "Value": mass_charge_g},
                {"Result": "Calc: Film thickness [mm]", "Value": delta_film * 1000.0},
                {"Result": "Calc: Fill state", "Value": status_film},
            ]
            
            for lname, lval in res_pt.items():
                if lval >= 1e8:
                    if lname == "Boiling":
                        val_to_write = "Not relevant (> 250g)"
                    elif lname == "Viscous":
                        val_to_write = "Not relevant (Viscous)"
                    else:
                        val_to_write = "Not relevant"
                else:
                    val_to_write = lval
                    
                res_list.append({"Result": f"Limit {lname} [W]", "Value": val_to_write})
            
            df_results = pd.DataFrame(res_list)
            df_curves = pd.DataFrame(data_curves)

            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")], title="Export Results")

            if not file_path: return

            with pd.ExcelWriter(file_path) as writer:
                df_inputs.to_excel(writer, sheet_name="Parameter", index=False)
                df_results.to_excel(writer, sheet_name="Results_Point", index=False)
                df_curves.to_excel(writer, sheet_name="RPM_Curves", index=False)
            
            messagebox.showinfo("Export Successful", f"Data was saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def run_analysis(self):
        """Performs the steady-state analysis and visualizes the performance limits."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperature [°C]"]) + 273.15
            rpm_pt = float(v["Speed [RPM]"])
            
            Q_input = float(v["Heat Load [W]"])
            
            res_pt = limits.get_all_limits(T_k, rpm_pt)
            limiting_factor = min(res_pt, key=res_pt.get)
            max_Q = res_pt[limiting_factor]
            
            R_th, k_eff = thermal.calc_resistances(T_k, rpm_pt, q_in_watts=Q_input)
            
            dT_at_max = max_Q * R_th
            
            props = fluid.get_properties(T_k)
            vol_wick = params.wick_area * params.length
            vol_void = vol_wick * params.porosity
            mass_charge_g = vol_void * props['rho_l'] * 1000.0
            mass_calc = mass_charge_g
            delta_film, sat_pct, status_film = model.calc_film_state(mass_calc / 1000.0, props['rho_l'], rpm_pt)
            
            # Show results
            for item in self.res_tree.get_children():
                self.res_tree.delete(item)
            
            self.res_tree.insert("", "end", values=(f"Results for {rpm_pt:.0f} RPM:", "", ""))
            self.res_tree.insert("", "end", values=("Limiting Factor", limiting_factor, ""))
            self.res_tree.insert("", "end", values=("Max. Power Q_max", f"{max_Q:.2f}", "W"))
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Fill Amount (100% Saturation)", f"{mass_charge_g:.2f}", "g"))
            self.res_tree.insert("", "end", values=("Eff. Film Thickness", f"{delta_film*1e6:.1f}", "µm"))
            self.res_tree.insert("", "end", values=("Fill State", status_film, ""))
            
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Therm. Resistance", f"{R_th:.4f}", "K/W"))
            self.res_tree.insert("", "end", values=("Eff. Thermal Conductivity", f"{k_eff:.1f}", "W/mK"))
            self.res_tree.insert("", "end", values=("Temp. Difference at Q_max", f"{dT_at_max:.1f}", "K"))
            
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Details Performance Limits:", "", ""))
            
            for k, val in res_pt.items():
                if val >= 1e8:
                    if k == "Boiling":
                        disp_val = "Not relevant (> 250g)"
                    elif k == "Viscous":
                        disp_val = "Not relevant (Viscous)"
                    else:
                        disp_val = "Not relevant"
                    disp_unit = "-"
                else:
                    disp_val = f"{val:.1f}"
                    disp_unit = "W"
                self.res_tree.insert("", "end", values=(f" - {k}", disp_val, disp_unit))

            # Calculate curve profiles
            rpms = np.linspace(0, 30000, 50)
            res_map = {"Capillary": [], "Boiling": [], "Sonic": [], "Entrainment": [], "Viscous": []}
            for r in rpms:
                res = limits.get_all_limits(T_k, r)
                for k in res_map:
                    res_map[k].append(res[k])
            
            # Plot preparation
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            self.plot_frame.update_idletasks() 
            w_px = self.plot_frame.winfo_width()
            h_px = self.plot_frame.winfo_height()
            
            if w_px > 50 and h_px > 50:
                 fig_w = w_px / 100.0  
                 fig_h = (h_px - 45) / 100.0
                 fig_w = max(fig_w, 3.0)
                 fig_h = max(fig_h, 2.0)
            else:
                 fig_w, fig_h = 6.5, 5.0 

            fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
            
            ax = fig.add_subplot(111)
            
            ax.plot(rpms, res_map["Capillary"], label="Capillary limit", linewidth=2)
            ax.plot(rpms, res_map["Boiling"], label="Boiling limit", linestyle='-.')
            ax.plot(rpms, res_map["Entrainment"], label="Entrainment limit", linestyle=':')
            ax.plot(rpms, res_map["Sonic"], label="Sonic limit", linestyle='--')
            
            safe_q = np.minimum.reduce([res_map[k] for k in res_map])
            ax.fill_between(rpms, 0, safe_q, color='green', alpha=0.1, label="Safe area")
            ax.plot(rpm_pt, max_Q, 'ro', markersize=8, label="Operating point")
            
            ax.set_yscale("log")
            ax.set_ylim(1, 100000)
            ax.set_xlabel("Speed [RPM]")
            ax.set_ylabel("Power [W]")
            ax.set_title("Operating Limits of the Rotating Heat Pipe")
            ax.legend(fontsize='small', loc='best') 
            ax.grid(True)
            
            fig.tight_layout()
            
            self.current_fig = fig

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_optimization(self):
        """Performs parameter optimization for the cone angle."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperature [°C]"]) + 273.15
            rpm = float(v["Speed [RPM]"])

            # Calculation of max geometrically possible angle
            r_out = params.d_out / 2.0
            r_in_start = params.d_in / 2.0
            min_wall_thickness = 0.0005 
            
            max_radial_growth = r_out - r_in_start - params.wick_thickness - min_wall_thickness
            
            if max_radial_growth <= 0:
                max_angle_deg = 0.1
                messagebox.showwarning("Geometry Warning", 
                    "D_outer is too small for a cone optimization.\n"
                    "There is no space for an expansion!")
            else:
                max_angle_rad = np.arctan(max_radial_growth / params.length)
                max_angle_deg = np.degrees(max_angle_rad)

            search_limit = min(10.0, max_angle_deg)
            
            angles = np.linspace(0, search_limit, 20)

            q_vals = []
            valid_angles = []

            for ang in angles:
                model.p.cone_angle_deg = ang
                
                res = limits.get_all_limits(T_k, rpm)
                limit_val = min(res.values())
                
                q_vals.append(limit_val)
                valid_angles.append(ang)

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            self.plot_frame.update_idletasks()
            w_px = self.plot_frame.winfo_width()
            h_px = self.plot_frame.winfo_height()
            
            if w_px > 50 and h_px > 50:
                 fig_w = w_px / 100.0
                 fig_h = (h_px - 45) / 100.0
                 fig_w = max(fig_w, 3.0)
                 fig_h = max(fig_h, 2.0)
            else:
                 fig_w, fig_h = 6.5, 5.0

            fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
            ax = fig.add_subplot(111)

            ax.plot(valid_angles, q_vals, 'b-o')
            
            best_idx = np.argmax(q_vals)
            best_ang = valid_angles[best_idx]
            max_val = q_vals[best_idx]
            
            ax.plot(best_ang, max_val, 'r*', markersize=12, label=f"Optimum: {best_ang:.2f}°")
            
            ax.set_xlabel("Cone angle [deg]")
            ax.set_ylabel("Max Q [W]")
            ax.set_title(f"Optimization (Geom. Limit: {max_angle_deg:.2f}°)")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            
            self.current_fig = fig
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_pressure_plot(self):
        """Visualizes the axial pressure profile of vapor and liquid."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperature [°C]"]) + 273.15
            rpm = float(v["Speed [RPM]"])
            
            res_pt = limits.get_all_limits(T_k, rpm)
            
            Q_limit = min(res_pt.values())
            Q_input = float(v["Heat Load [W]"])
            
            Q_plot = min(Q_input, Q_limit) if Q_limit > 1.0 else Q_input
            
            m_dot = Q_plot / fluid.get_properties(T_k)['h_fg']
            
            z = np.linspace(0, params.length, 100)
            props = fluid.get_properties(T_k)
            
            # Vapor pressure profile (incl. friction losses)
            dp_v = model.loss_vapor_advanced(m_dot, props['rho_v'], props['mu_v'], rpm)
            pv = props['p_sat'] - dp_v * (z/params.length) 
            
            omega = model.calc_omega(rpm)
            alpha = params.get_tilt_rad()
            
            r_z = params.r_min + (params.d_in / 2.0) + z * np.sin(alpha)
            
            r_start = params.r_min + (params.d_in / 2.0)
            
            # Liquid pressure profile (Saturation + Centrifugal pressure - friction)
            p_rot = 0.5 * props['rho_l'] * omega**2 * (r_z**2 - r_start**2)
            
            dp_l, _ = model.loss_liquid_hybrid(m_dot, props['rho_l'], props['mu_l'], rpm)
            p_fric = dp_l * (z / params.length)
            
            pl = props['p_sat'] + p_rot - p_fric
            
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            self.plot_frame.update_idletasks()
            w_px = self.plot_frame.winfo_width()
            h_px = self.plot_frame.winfo_height()
            
            if w_px > 50 and h_px > 50:
                 fig_w = w_px / 100.0
                 fig_h = (h_px - 45) / 100.0
                 fig_w = max(fig_w, 3.0)
                 fig_h = max(fig_h, 2.0)
            else:
                 fig_w, fig_h = 6.5, 5.0

            fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
            ax = fig.add_subplot(111)

            ax.plot(z*1000, pv/1000, 'r--', label='Vapor')
            ax.plot(z*1000, pl/1000, 'b-', label='Liquid')
            
            ax.set_xlabel("z [mm] (Cond -> Evap)")
            ax.set_ylabel("Pressure [kPa]")
            ax.set_title(f"Pressure profile at {Q_plot:.1f} W ({rpm:.0f} rpm)")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            self.current_fig = fig

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(e) 

    def run_transient(self):
        """Simulates the startup behavior over time."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            
            t_max = 120.0  # Simulation duration [s]
            t_ramp = 5.0   # Time until target RPM [s]
            
            target_rpm = float(v["Speed [RPM]"])
            T_start = 20.0 + 273.15 
            
            Q_load = float(v["Heat Load [W]"]) 
            
            R_ext_cooling = 0.05 
            T_amb = 20.0 + 273.15
            
            # Calculation of thermal masses
            vol_wall = np.pi * ((params.d_out/2)**2 - (params.d_in/2 + params.wick_thickness)**2) * params.length
            mass_wall = vol_wall * params.rho_solid
            
            mass_fixtures_evap = 1.5  
            cp_fixtures = 460.0       
            C_heater_block = mass_fixtures_evap * cp_fixtures
            
            props_start = fluid.get_properties(T_start)
            vol_wick_void = params.wick_area * params.length * params.porosity
            mass_fluid_total = vol_wick_void * props_start['rho_l'] * 1.2 
            
            C_total_wall = mass_wall * params.cp_solid
            C_total_fluid = mass_fluid_total * props_start['cp_l']
            
            C_evap = (C_total_wall + C_total_fluid) * 0.5 + C_heater_block
            C_cond = (C_total_wall + C_total_fluid) * 0.5
            
            t_curr = 0.0
            T_evap_hist = [T_start]
            T_cond_hist = [T_start]
            rpm_hist = [0]
            t_hist = [0.0]
            
            T_e_curr = T_start
            T_c_curr = T_start
            
            # Time step method
            while t_curr < t_max:
                if t_curr < t_ramp:
                    current_rpm = target_rpm * (t_curr / t_ramp)
                else:
                    current_rpm = target_rpm
                
                T_mean = (T_e_curr + T_c_curr) / 2
                
                if current_rpm < 100:
                    k_cu = params.k_solid
                    A_cross = np.pi * ((params.d_out/2)**2 - (params.d_in/2)**2)
                    R_internal = params.length / (k_cu * A_cross)
                else:
                    try:
                        R_internal, _ = thermal.calc_resistances(T_mean, current_rpm, q_in_watts=Q_load)
                    except:
                        R_internal = 10.0
                
                tau_evap = C_evap * R_internal
                tau_cond = C_cond * R_ext_cooling
                tau_min = min(tau_evap, tau_cond)
                
                dt = min(0.1, tau_min * 0.4)
                
                dQ_in = Q_load
                dQ_exchange = (T_e_curr - T_c_curr) / R_internal
                dT_e = (dQ_in - dQ_exchange) * dt / C_evap
                
                dQ_out = (T_c_curr - T_amb) / R_ext_cooling
                dT_c = (dQ_exchange - dQ_out) * dt / C_cond
                
                T_e_curr += dT_e
                T_c_curr += dT_c
                t_curr += dt
                
                T_evap_hist.append(T_e_curr)
                T_cond_hist.append(T_c_curr)
                rpm_hist.append(current_rpm)
                t_hist.append(t_curr)
            
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            self.plot_frame.update_idletasks()
            w_px = self.plot_frame.winfo_width()
            h_px = self.plot_frame.winfo_height()
            
            if w_px > 50 and h_px > 50:
                 fig_w = w_px / 100.0
                 fig_h = (h_px - 45) / 100.0
                 fig_w = max(fig_w, 3.0)
                 fig_h = max(fig_h, 2.0)
            else:
                 fig_w, fig_h = 6.5, 5.0 
            
            fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)

            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            
            ax1.plot(t_hist, np.array(T_evap_hist) - 273.15, 'r-', label='T_Evap')
            ax1.plot(t_hist, np.array(T_cond_hist) - 273.15, 'b-', label='T_Cond')
            ax1.set_ylabel('Temp [°C]')
            ax1.set_title(f'Transient Startup Behavior (at Heat Load={Q_load} W)')
            ax1.grid(True)
            ax1.legend(fontsize='small')
            
            ax2.plot(t_hist, rpm_hist, 'k--')
            ax2.set_ylabel('RPM')
            ax2.set_xlabel('Time [s]')
            ax2.grid(True)
            
            fig.tight_layout()

            self.current_fig = fig

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()

        except Exception as e:
            messagebox.showerror("Error Transient", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = HeatPipeGUI(root)
    root.mainloop()