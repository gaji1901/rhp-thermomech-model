import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # Ermöglicht die Einbettung von Matplotlib-Plots in Tkinter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
import json # Für das Speichern und Laden von Konfigurationen
import os   # Für Dateisystemoperationen

# Optionaler Import für Excel-Export-Funktionalität
try:
    import pandas as pd
except ImportError:
    pd = None

# Import der CoolProp-Bibliothek zur Abfrage exakter thermophysikalischer Stoffdaten
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("Fehler: CoolProp ist nicht installiert. Bitte 'pip install CoolProp' ausführen.")
    exit()

# ==============================================================================
# KONSTANTEN & SYSTEMEINSTELLUNGEN
# ==============================================================================
ROSSBY_FACTOR_VAPOR = 0.5   # Empirischer Faktor zur Berücksichtigung von Drall-Effekten in der Dampfphase
ROSSBY_FACTOR_LIQUID = 0.15 # Empirischer Faktor für den Rückfluss des Flüssigkeitsfilms unter Rotation
DESIGN_Q_TARGET = 100.0     # Referenzlast [W] für die Initialisierung von Iterationen
NUCLEATION_RADIUS_DEFAULT = 2.0e-6 # Standardradius für Keimstellen beim Sieden (Nucleate Boiling)

# ==============================================================================
# TEIL 1: DATENSTRUKTUREN FÜR PARAMETER & GEOMETRIE
# ==============================================================================

@dataclass
class HeatPipeParameters:
    """
    Datenklasse zur Speicherung aller geometrischen und physikalischen
    Eigenschaften des Wärmerohrs.
    """
    length: float           # Gesamtlänge des Rohrs [m]
    d_out: float            # Außendurchmesser [m]
    d_in: float             # Innendurchmesser (Dampfkern) [m]
    
    l_evap: float           # Länge der Verdampferzone [m]
    l_adiab: float          # Länge der Adiabatenzone (Transportzone) [m]
    l_cond: float           # Länge der Kondensatorzone [m]
    
    r_min: float            # Achsversatz am Kondensator (Exzentrizität) [m]
    cone_angle_deg: float   # Interner Konuswinkel zur Förderung des Fluidtransports [Grad]
    inclination_deg: float  # Globale Neigung des Systems im Raum [Grad]
    
    pore_radius: float      # Effektiver Porenradius der Kapillarstruktur [m]
    permeability: float     # Permeabilität (Durchlässigkeit) des Dochts [m^2]
    wick_area: float        # Querschnittsfläche des Dochts [m^2]
    wick_thickness: float   # Dicke der Sinterstruktur [m]
    porosity: float         # Porosität des Sintermaterials (0.0 - 1.0)
    
    k_solid: float          # Wärmeleitfähigkeit des Rohrmaterials [W/mK]
    
    # Materialwerte für transiente Berechnungen (Standard: Kupfer)
    rho_solid: float = 8960.0  # Dichte des Feststoffs [kg/m^3]
    cp_solid: float = 385.0    # Spezifische Wärmekapazität des Feststoffs [J/kgK]

    def get_tilt_rad(self):
        """Gibt den internen Konuswinkel in Bogenmaß zurück."""
        return np.radians(self.cone_angle_deg)
    
    def get_global_inclination_rad(self):
        """Gibt die globale Systemneigung in Bogenmaß zurück."""
        return np.radians(self.inclination_deg)

    @property
    def l_eff(self):
        """
        Berechnet die effektive Länge für Reibungsverluste.
        Definition: 0.5 * L_Verdampfer + L_Adiabat + 0.5 * L_Kondensator.
        """
        return 0.5 * self.l_evap + self.l_adiab + 0.5 * self.l_cond

# ==============================================================================
# TEIL 2: STOFFWERTE (COOLPROP WRAPPER)
# ==============================================================================

class WorkingFluid:
    """
    Klasse zur Verwaltung des Arbeitsmediums und Abfrage temperaturabhängiger
    Stoffwerte über die CoolProp-Schnittstelle.
    """
    def __init__(self, name="Water"):
        self.name = name

    def get_properties(self, T_k):
        """
        Ermittelt alle relevanten thermophysikalischen Stoffwerte für eine gegebene
        Temperatur T [Kelvin].
        """
        fluid_map = {"Water": "Water", "Methanol": "Methanol", "Ammonia": "Ammonia"}
        fluid_string = fluid_map.get(self.name, "Water")
        
        # Begrenzung der Temperatur, um numerische Divergenz nahe dem kritischen Punkt zu verhindern
        T_crit = PropsSI('T_CRITICAL', fluid_string)
        T = np.clip(T_k, 200.0, T_crit - 5.0)

        try:
            # Abfrage der Dichte (Density) für Flüssigkeit (Q=0) und Dampf (Q=1)
            rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid_string)
            rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid_string)
            
            # Abfrage der Viskosität
            mu_l = PropsSI('V', 'T', T, 'Q', 0, fluid_string)
            mu_v = PropsSI('V', 'T', T, 'Q', 1, fluid_string)
            
            # Weitere thermodynamische Eigenschaften
            sigma = PropsSI('I', 'T', T, 'Q', 0, fluid_string) # Oberflächenspannung
            k_l = PropsSI('L', 'T', T, 'Q', 0, fluid_string)   # Wärmeleitfähigkeit Flüssigkeit
            p_sat = PropsSI('P', 'T', T, 'Q', 0, fluid_string) # Sättigungsdampfdruck
            
            # Enthalpie zur Berechnung der Verdampfungsenthalpie
            h_gas = PropsSI('H', 'T', T, 'Q', 1, fluid_string)
            h_liq = PropsSI('H', 'T', T, 'Q', 0, fluid_string)
            h_fg = h_gas - h_liq
            
            # Spezifische Wärmekapazität der Flüssigkeit
            cp_l = PropsSI('C', 'T', T, 'Q', 0, fluid_string)
            
            # Berechnung des Isentropenexponenten (Gamma) für Schallgeschwindigkeits-Limits
            cp_v = PropsSI('C', 'T', T, 'Q', 1, fluid_string)
            cv_v = PropsSI('O', 'T', T, 'Q', 1, fluid_string) # 'O' entspricht Cv (mass basis)
            gamma = cp_v / cv_v
            
            # Gasdaten
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
            raise ValueError(f"CoolProp Fehler für {self.name} bei {T:.1f} K: {str(e)}")

# ==============================================================================
# TEIL 3: MECHANISCHES MODELL (DRUCKBILANZ)
# ==============================================================================

class RotatingHeatPipeModel:
    """
    Modelliert die hydrodynamischen Vorgänge im rotierenden Wärmerohr,
    insbesondere die Druckbilanz zwischen treibenden und hemmenden Kräften.
    """
    def __init__(self, params: HeatPipeParameters):
        self.p = params
        self.gravity = 9.81

    def calc_omega(self, rpm):
        """Konvertiert Drehzahl [U/min] in Winkelgeschwindigkeit [rad/s]."""
        return rpm * (2 * np.pi / 60)

    def calc_film_state(self, mass_kg, rho_l, rpm):
        """
        Analysiert den Füllzustand des Wärmerohrs.
        Berechnet, ob das Fluid vollständig im Docht gebunden ist oder ob sich
        ein freier Film bzw. Pool bildet.
        """
        if rho_l <= 0: return 0, 0, "Error"
        vol_liq = mass_kg / rho_l
        r_wick_outer = self.p.d_in / 2 + self.p.wick_thickness
        vol_void_total = self.p.wick_area * self.p.length * self.p.porosity
        saturation = (vol_liq / vol_void_total) * 100.0
        
        vol_liq_effective = vol_liq / self.p.porosity
        term = vol_liq_effective / (np.pi * self.p.length)
        
        # Prüfung auf Überfüllung (Pool-Bildung im Innenraum)
        if (r_wick_outer**2 - term) < 0:
            delta = self.p.wick_thickness + 0.0005
            status = "ÜBERFÜLLT (Pool-Bildung)"
        else:
            r_inner_fluid = np.sqrt(r_wick_outer**2 - term)
            delta = r_wick_outer - r_inner_fluid
            if delta < self.p.wick_thickness * 0.95:
                status = "UNTERFÜLLT (Dry-Out Risiko)"
            elif delta > self.p.wick_thickness * 1.05:
                status = "ÜBERFÜLLT"
            else:
                status = "OPTIMAL"
        return delta, saturation, status

    def force_centrifugal(self, rpm, rho_liq):
        """
        Berechnet den treibenden Druckaufbau durch die Zentrifugalkraft.
        Dies ist der Hauptantriebsmechanismus in rotierenden Wärmerohren mit Konus.
        """
        omega = self.calc_omega(rpm)
        alpha = self.p.get_tilt_rad()
        
        # Der Radius des Fluids berechnet sich aus Achsversatz + halbem Rohrdurchmesser
        r_cond = self.p.r_min + (self.p.d_in / 2.0)
        
        # Projektion der axialen Länge auf den Radiuszuwachs durch den Konus
        dr = self.p.length * np.sin(alpha)
        r_evap = r_cond + dr
        
        # Integration der Fliehkraft über den Radius: 0.5 * rho * w^2 * (r_max^2 - r_min^2)
        dp = 0.5 * rho_liq * (omega**2) * (r_evap**2 - r_cond**2)
        return dp

    def force_capillary_max(self, sigma):
        """Berechnet den maximalen Kapillardruck basierend auf dem Porenradius."""
        return 2 * sigma / self.p.pore_radius

    def loss_vapor_advanced(self, m_dot, rho_vap, mu_vap, rpm):
        """
        Berechnet den Druckverlust in der Dampfströmung.
        Berücksichtigt Reibungseffekte und Coriolis-Kräfte (via Rossby-Zahl)
        bei hohen Drehzahlen.
        """
        if m_dot <= 1e-9: return 0.0
        r_v = self.p.d_in / 2.0
        v = m_dot / (rho_vap * (np.pi * r_v**2))
        omega = self.calc_omega(rpm)
        
        # Korrekturfaktor für rotierende Strömungen basierend auf der Rossby-Zahl
        if omega > 1.0 and v > 0:
            Ro = v / (omega * self.p.d_in)
            coriolis_factor = 1.0 + ROSSBY_FACTOR_VAPOR * (Ro + 0.05)**(-0.5)
        else:
            coriolis_factor = 1.0
            
        mu_eff = mu_vap * coriolis_factor
        
        # Bestimmung des Strömungsregimes (laminar/turbulent)
        Re = (rho_vap * v * self.p.d_in) / mu_eff
        if Re < 2300:
            dp = (8 * mu_eff * self.p.l_eff * m_dot) / (np.pi * rho_vap * r_v**4)
        else:
            f = 0.3164 * (Re**(-0.25)) # Blasius-Gleichung für turbulente Strömung
            dp = f * (self.p.l_eff / self.p.d_in) * 0.5 * rho_vap * v**2
        return dp

    def loss_liquid_hybrid(self, m_dot, rho_liq, mu_liq, rpm):
        """
        Berechnet den Druckverlust im Flüssigkeitsrücklauf.
        Verwendet ein paralleles Modell aus Darcy-Strömung (durch den Docht) und
        Filmströmung (über dem Docht), falls Fluidüberschuss besteht.
        """
        if m_dot <= 1e-9: return 0.0, "Static"

        omega = self.calc_omega(rpm)
        r_avg = self.p.d_in / 2.0
        
        # Rossby-Korrektur für die Viskosität der Flüssigkeit
        v_approx = m_dot / (rho_liq * self.p.wick_area)
        if omega > 1.0 and v_approx > 0:
            Ro_l = v_approx / (omega * (self.p.d_in/2))
            visc_factor_l = 1.0 + ROSSBY_FACTOR_LIQUID * (Ro_l + 0.1)**(-0.5)
        else:
            visc_factor_l = 1.0
            
        mu_eff_liq = mu_liq * visc_factor_l
        
        # Effektive Beschleunigung (Zentrifugal oder Gravitation)
        if omega < 1.0:
            g_eff = 9.81
        else:
            g_eff = omega**2 * r_avg
        
        # 1. Leitwert für Strömung durch den Docht (Darcy-Gesetz)
        G_wick = (self.p.permeability * self.p.wick_area * rho_liq) / (mu_eff_liq * self.p.l_eff)
        
        # 2. Berechnung der theoretischen Filmdicke bei totalem Massenstrom (Nusselt-Ansatz)
        term = (3 * mu_eff_liq * m_dot) / ((rho_liq**2) * g_eff * 2 * np.pi * r_avg)
        delta_theor = term**(1/3)
        
        if delta_theor > self.p.wick_thickness:
            # Fluid tritt aus dem Docht aus -> Parallele Filmströmung
            delta_excess = delta_theor - self.p.wick_thickness
            
            # Leitwert für die Filmströmung
            G_film = (rho_liq * 2 * np.pi * (self.p.d_in/2.0) * delta_excess**3) / (3 * mu_eff_liq * self.p.l_eff)
            
            G_total = G_wick + G_film
            mode = "Parallel (Wick+Film)"
        else:
            # Fluid fließt nur im Docht
            G_total = G_wick
            mode = "Darcy (Wick)"
            
        dp = m_dot / G_total
        return dp, mode

    def loss_coriolis(self, m_dot, rpm, rho_liq):
        """
        Platzhalter für explizite Coriolis-Verluste.
        (Aktuell bereits implizit über Rossby-Faktoren in der Viskosität berücksichtigt).
        """
        return 0.0

    def force_gravity(self, rho_liq):
        """Berechnet den Einfluss der Schwerkraft basierend auf der globalen Neigung."""
        beta = self.p.get_global_inclination_rad()
        h = self.p.length * np.sin(beta)
        return rho_liq * self.gravity * h

    def check_balance(self, dp_k, dp_omega, dp_v, dp_l, dp_grav, dp_cor):
        """
        Prüft die Druckbilanz des Systems.
        Treibende Kräfte: Kapillardruck + Zentrifugalkraft
        Hemmende Kräfte: Druckverluste (Dampf/Flüssigkeit) + Schwerkraft + Coriolis
        """
        drive = dp_k + dp_omega
        resist = dp_v + dp_l + dp_grav + dp_cor
        return drive - resist

# ==============================================================================
# TEIL 3b: THERMISCHES WIDERSTANDSMODELL
# ==============================================================================

class ThermalNetwork:
    """
    Berechnet die thermischen Widerstände und Wärmeübergangskoeffizienten (HTC)
    im System.
    """
    def __init__(self, model: RotatingHeatPipeModel, fluid: WorkingFluid):
        self.model = model
        self.fluid = fluid

    def get_accel(self, rpm, diameter):
        """Berechnet die radiale Zentrifugalbeschleunigung."""
        omega = self.model.calc_omega(rpm)
        r = diameter / 2.0
        return omega**2 * r

    def calc_h_natural_convection(self, props, rpm, d_in, dT, L_char):
        """
        Berechnet den HTC für natürliche Konvektion im starken Zentrifugalfeld.
        Verwendet Korrelationen für rotierende Systeme (Rayleigh-Zahl).
        """
        a_c = self.get_accel(rpm, d_in)
        
        rho = props['rho_l']
        mu = props['mu_l']
        k = props['k_l']
        cp = props['cp_l']
        
        # Volumetrischer Ausdehnungskoeffizient (Näherung)
        beta = 0.001 

        if L_char <= 0: L_char = 1e-5
        
        # Berechnung der Kennzahlen
        Pr = (mu * cp) / k
        Gr = (rho**2 * a_c * beta * dT * L_char**3) / (mu**2)
        Ra = Gr * Pr
        
        if Ra < 1e-9: return 100.0 # Minimalwert zur Vermeidung von Singularitäten
        
        # Korrelation nach Song/Marto für rotierende Systeme
        Nu_n = 0.133 * (Ra**0.375)
        h_nat = (Nu_n * k) / L_char
        return h_nat

    def calc_h_cond_film_exact(self, props, rpm, m_dot):
        """
        Berechnet den Wärmeübergangskoeffizienten (HTC) bei der Kondensation.
        Berücksichtigt die Verdünnung des Films durch Zentrifugalkräfte (Drainage).
        """
        omega = self.model.calc_omega(rpm)
        rho_l = props['rho_l']
        mu_l = props['mu_l']
        k_l = props['k_l']
        
        r_avg = self.model.p.d_in / 2.0
        alpha = self.model.p.get_tilt_rad() # Konuswinkel
        sin_alpha = np.sin(alpha)
        
        # Grenzwinkel, ab dem die Pumpwirkung dominant wird
        TRANSITION_ANGLE = 0.0008 
        
        if sin_alpha < 1e-6: sin_alpha = 1e-6 

        # Festlegung des Basis-HTC im Stillstand
        if alpha < TRANSITION_ANGLE:
            # 0 Grad: Sehr guter Kontakt im Stillstand (Sättigung)
            h_static = 25000.0
        else:
            # > 0 Grad: Fluid sammelt sich durch Schwerkraft unten, geringerer HTC oben
            h_static = 10600.0

        if m_dot <= 1e-9: return h_static

        omega_calc = max(0.1, omega) 
        
        # Berechnung der Filmdicke nach Nusselt unter Fliehkraft
        numerator = 3 * mu_l * m_dot
        denominator = 2 * np.pi * r_avg * (rho_l**2) * (omega_calc**2) * r_avg * sin_alpha
        delta_flow = (numerator / denominator)**(1.0/3.0) 
        
        # Limitierung für Pool-Bildung bei 0 Grad
        vol_void = self.model.p.wick_area * self.model.p.length * self.model.p.porosity
        vol_fluid_cond = vol_void * 0.5 
        A_inner = np.pi * self.model.p.d_in * self.model.p.length
        delta_pool_limit = vol_fluid_cond / A_inner
        
        # HTC für vollständig gefluteten Zustand
        h_pool_saturated = max(k_l / delta_pool_limit, 10000.0)

        # Fallunterscheidung für Verhalten bei Rotation
        if alpha < TRANSITION_ANGLE:
            # Fall 0 Grad: Verschlechterung durch "Flooding" (Fliehkraft drückt Fluid in Struktur)
            r_rotation = self.model.p.r_min + (self.model.p.d_in / 2.0)
            g_force = (omega**2 * r_rotation) / 9.81
            decay = 1.0 - np.exp(-g_force / 150.0) 
            h_combined = h_static * (1.0 - decay) + h_pool_saturated * decay
            
        else:
            # Fall > 0 Grad: Verbesserung durch "Drainage" (Film wird dünner)
            delta = delta_flow
            delta = max(delta, 1e-7) # Physikalisch minimale Filmdicke
            h_rot = (k_l / delta) * 0.7 # Korrektur für Scherungseffekte
            
            # Vektorielle Addition
            h_combined = (h_static**2 + h_rot**2)**0.5
        
        return h_combined

    def calc_resistances(self, T_op, rpm, q_in_watts=100.0):
        """
        Hauptfunktion zur Berechnung des thermischen Netzwerks.
        Ermittelt den Gesamtwiderstand und die effektive Wärmeleitfähigkeit.
        """
        props = self.fluid.get_properties(T_op)
        p = self.model.p
        omega = self.model.calc_omega(rpm)
        
        # 1. Berechnung des Massenstroms aus der thermischen Last
        m_dot = q_in_watts / props['h_fg']
        
        # Berücksichtigung von "Pool Blocking" im Kondensator (Geometrische Überfüllung)
        vol_void = p.wick_area * p.length * p.porosity
        mass_internal = vol_void * props['rho_l']
        delta_calc, saturation_pct, _ = self.model.calc_film_state(mass_internal, props['rho_l'], rpm)
        
        # Ermittlung der aktiven Kondensatorlänge (Reduktion durch Sumpfbildung)
        l_cond_active = p.l_cond
        l_pool = 0.0
        
        if saturation_pct > 100.0:
            vol_excess = max(0.0, (saturation_pct - 100.0) / 100.0) * vol_void
            cross_section_inner = np.pi * (p.d_in / 2.0)**2
            l_pool = vol_excess / cross_section_inner
            l_pool = min(l_pool, p.l_cond * 0.99)
            l_cond_active = p.l_cond - l_pool

        # 2. Kondensator HTC (Verwendung des exakten Filmmodells)
        h_cond = self.calc_h_cond_film_exact(props, rpm, m_dot)
        
        # 3. Verdampfer HTC (Mischkonvektion)
        dT_guess = 5.0 # Startwert für Iteration
        h_evap = 500.0
        
        k_s = p.k_solid
        k_l = props['k_l']
        eps = p.porosity
        
        # Effektive Leitfähigkeit des Dochtmaterials
        num = k_l + 2*k_s - 2*eps*(k_s - k_l)
        den = k_l + 2*k_s + eps*(k_s - k_l)
        k_wick_eff = k_s * (num / den)
        
        # Bestimmung der charakteristischen Länge für Kennzahlen
        if delta_calc > p.wick_thickness:
             L_char = delta_calc
        else:
             L_char = p.wick_thickness
             
        g_acc = omega**2 * (p.d_in / 2.0)
        
        # Benetzungsmodell und Sinter-Einfluss (Physikalischer Ansatz)
        # Berechnung der Benetzung in Abhängigkeit von Hydrostatik und Kapillarkraft
        r_rot_center = self.model.p.r_min + (self.model.p.d_in / 2.0)
        a_centrifugal = omega**2 * r_rot_center
        
        # Hydrostatischer Druckunterschied quer zum Rohr
        dp_hydro_trans = props['rho_l'] * a_centrifugal * self.model.p.d_in
        dp_cap_max = 2.0 * props['sigma'] / self.model.p.pore_radius

        # Benetzungsfaktor (Transversaler Dry-Out)
        wetting_factor = 1.0 

        if dp_hydro_trans > dp_cap_max:
            # Kapillarkraft reicht nicht aus, um Fluid gegen die Fliehkraft um den Umfang zu verteilen
            base_pool_fraction = 0.35 
            climb_ratio = dp_cap_max / dp_hydro_trans
            wetting_factor = base_pool_fraction + (1.0 - base_pool_fraction) * climb_ratio
            wetting_factor = max(0.3, wetting_factor)
        
        # Einfluss der Axialkomponente (Neigung/Konus) auf die Benetzung
        boost_angle = self.model.p.get_tilt_rad()
        
        if boost_angle < 0.001 and abs(self.model.p.inclination_deg) < 89.0:
             boost_angle = np.radians(self.model.p.inclination_deg)

        if boost_angle > 0.001 and omega > 50.0:
            axial_drive = (omega**2 * r_rot_center * np.sin(boost_angle))
            reflood_factor = 1.0 + (axial_drive / 500.0) 
            wetting_factor = wetting_factor * reflood_factor
            
            # Berücksichtigung des "Durchströmungs-Effekts" bei porösen Sinterstrukturen
            if self.model.p.porosity < 0.9 and self.model.p.porosity > 0.1:
                sinter_boost = 1.0 + (axial_drive / 200.0) * 0.8 
                wetting_factor = wetting_factor * sinter_boost
            else:
                wetting_factor = min(1.0, wetting_factor)

        else:
            wetting_factor = min(1.0, wetting_factor)

        # Berechnung der effektiven Verdampferfläche
        fill_ratio = min(1.0, max(0.0, saturation_pct / 100.0))
        A_evap_effective = (np.pi * self.model.p.d_in * self.model.p.l_evap) * wetting_factor * fill_ratio
        
        # Iterative Berechnung des Wärmeübergangs im Verdampfer
        for _ in range(3):
            # A) Sieden (Korrelation nach Cooper)
            pr = props['p_sat'] / props['p_crit']
            pr = np.clip(pr, 0.001, 0.99)
            M_g_mol = props['molar_mass'] * 1000.0
            
            q_flux = max(q_in_watts, 0.1) / max(A_evap_effective, 1e-6)
            
            h_boiling = 55.0 * (pr**0.12) * ((-np.log10(pr))**(-0.55)) * (M_g_mol**(-0.5)) * (q_flux**0.67)
            
            # B) Natürliche Konvektion (Song et al.)
            h_nat = self.calc_h_natural_convection(props, rpm, p.d_in, dT_guess, L_char)
            
            # C) Erzwungene Konvektion (Filmströmung)
            h_forced = k_l / L_char 
            
            # Kombiniertes Modell (Mischkonvektion)
            h_mixed = (h_forced**3.5 + h_nat**3.5)**(1/3.5)
            
            # Dämpfung des Siedens durch hohe g-Kräfte (Siede-Unterdrückung)
            g_critical = 250.0 * 9.81 
            g_width = 50.0 * 9.81 
            
            if g_acc > 0:
                S_suppression = 1.0 / (1.0 + np.exp((g_acc - g_critical) / g_width))
            else:
                S_suppression = 1.0
            
            # Überblendung zwischen Sieden und Konvektion
            n_blend = 3.0
            h_evap = ( (h_boiling * S_suppression)**n_blend + h_mixed**n_blend )**(1.0 / n_blend)
                
            dT_new = q_flux / h_evap
            dT_guess = 0.5 * dT_guess + 0.5 * dT_new
            
        # 4. Zusammenstellung der Widerstände (Resistor Network)
        r_o = p.d_out / 2
        r_i_wick = p.d_in / 2
        r_if_wall = r_i_wick + p.wick_thickness
        
        # Wärmeleitung durch die Rohrwand
        R_wall_evap = np.log(r_o / r_if_wall) / (2 * np.pi * p.l_evap * p.k_solid)
        R_wall_cond = np.log(r_o / r_if_wall) / (2 * np.pi * p.l_cond * p.k_solid)
        
        # Wärmeleitung durch den Docht
        R_wick_evap = np.log(r_if_wall / r_i_wick) / (2 * np.pi * p.l_evap * k_wick_eff)
        R_wick_cond = np.log(r_if_wall / r_i_wick) / (2 * np.pi * p.l_cond * k_wick_eff)
        
        # Phasenübergangswiderstand Verdampfer
        R_film_evap = 1.0 / (h_evap * A_evap_effective)
        
        # Phasenübergangswiderstand Kondensator (Differenzierung Active vs. Pool)
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

        # Widerstand des Sumpf-Bereichs (Pool)
        if l_pool > 1e-6:
            h_pool_conduction = k_l / (p.d_in / 4.0)
            R_pool = 1.0 / (h_pool_conduction * (np.pi * p.d_in * l_pool))
        else:
            R_pool = 1e9

        # Parallel-Schaltung im Kondensator
        R_film_cond = 1.0 / ( (1.0/R_film_active) + (1.0/R_pool) )
        
        R_vapor = 0.0001
        
        # Summe der radialen Widerstände
        R_radial = (R_wall_evap + R_wick_evap + R_film_evap +
                    R_vapor +
                    R_film_cond + R_wick_cond + R_wall_cond)
                    
        # Berücksichtigung der axialen Leitung durch die Rohrwand (Parallelpfad)
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
# TEIL 4: LEISTUNGSGRENZEN (LIMITS)
# ==============================================================================

class PerformanceLimits:
    """
    Berechnet die physikalischen Betriebsgrenzen des Wärmerohrs.
    """
    def __init__(self, model: RotatingHeatPipeModel, fluid: WorkingFluid):
        self.model = model
        self.fluid = fluid

    def limit_capillary(self, T_op, rpm):
        """
        Berechnet das Kapillarlimit (hydrodynamische Grenze).
        Sucht iterativ die Wärmelast Q, bei der die treibenden Drücke den
        Widerständen entsprechen.
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
            
            # Prüfung auf transversalen Dry-Out (Fliehkraft vs. Kapillarkraft am Umfang)
            omega = self.model.calc_omega(rpm)
            r_pipe = self.model.p.d_in / 2.0
            r_rotation_center = self.model.p.r_min + r_pipe
            accel_field = omega**2 * r_rotation_center
            
            dp_hydro_transversal = props['rho_l'] * accel_field * (2 * r_pipe)
            dp_cap_max = 2 * props['sigma'] / self.model.p.pore_radius 
            
            # Reduktion des verfügbaren Förderdrucks, wenn Kapillare durch Querbeschleunigung belastet ist
            if dp_hydro_transversal > dp_cap_max:
                dp_k_available = max(0.0, dp_cap_max - dp_hydro_transversal)
                dp_k = dp_k_available 
            
            return self.model.check_balance(dp_k, dp_omega, dp_v, dp_l, dp_grav, dp_cor)

        # Bisektionsverfahren zur Nullstellensuche
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
        Berechnet das Siedelimit.
        Grenze, ab der Blasenbildung im Docht den Flüssigkeitsstrom blockiert.
        Berücksichtigt, dass hohe g-Kräfte das Sieden unterdrücken.
        """
        props = self.fluid.get_properties(T_op)
        
        r_rot = self.model.p.d_in / 2.0
        omega = self.model.calc_omega(rpm)
        accel = omega**2 * r_rot
        g_force = accel / 9.81
        
        # Ab ca. 250g dominiert Konvektion, das klassische Siedelimit ist nicht mehr relevant
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
        Berechnet das Schalllimit.
        Dampfgeschwindigkeit erreicht Mach 1 am Verdampferausgang.
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
        Berechnet das Mitrisslimit (Entrainment).
        Hohe Dampfgeschwindigkeiten reißen Flüssigkeitstropfen aus dem Docht.
        """
        props = self.fluid.get_properties(T_op)
        A_v = np.pi * (self.model.p.d_in/2)**2
        l_char = self.model.p.pore_radius * 2
        term = (props['sigma'] * props['rho_v']) / l_char
        Q_ent = A_v * props['h_fg'] * np.sqrt(term)
        return Q_ent

    def limit_viscous(self, T_op):
        """
        Berechnet das Viskositätslimit.
        Relevant bei sehr niedrigen Temperaturen (hohe Dampfviskosität).
        """
        props = self.fluid.get_properties(T_op)
        d_v = self.model.p.d_in
        A_v = np.pi * (d_v / 2)**2
        numerator = A_v * ((d_v/2)**2) * props['h_fg'] * props['rho_v'] * props['p_sat']
        denominator = 16 * props['mu_v'] * self.model.p.l_eff
        return numerator / denominator

    def get_all_limits(self, T_op, rpm):
        """Fasst alle Leistungsgrenzen in einem Dictionary zusammen."""
        return {
            "Capillary": self.limit_capillary(T_op, rpm),
            "Sonic": self.limit_sonic(T_op),
            "Entrainment": self.limit_entrainment(T_op),
            "Boiling": self.limit_boiling(T_op, rpm),
            "Viscous": self.limit_viscous(T_op)
        }

# ==============================================================================
# TEIL 5: GUI UND VISUALISIERUNG
# ==============================================================================

class HeatPipeGUI:
    """
    Grafische Benutzeroberfläche (Tkinter) zur Steuerung der Simulation.
    Beinhaltet Eingabemasken, Echtzeit-Validierung, Plotting und Export.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Rotierende Wärmerohre - Simulation Tool ")
        self.root.geometry("1200x900") 
        
        # Konfiguration für Speichern/Laden
        self.settings_file = "rhp_settings.json"
        self.saved_data = {}
        self.load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Design-Einstellungen
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
        
        # Header-Bereich
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(header_frame, text="Berechnungsmodell für rotierende Wärmerohre", style="Header.TLabel").pack(side="left")
        ttk.Button(header_frame, text="ℹ Annahmen", command=self.show_assumptions, width=12).pack(side="right")

        # Aktionsleiste
        action_bar = ttk.LabelFrame(main_container, text=" Aktionen ", padding="5")
        action_bar.pack(fill="x", pady=(0, 10))

        ttk.Button(action_bar, text="▶ Berechnen & Plot", command=self.run_analysis).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="⚙ Optimierung", command=self.run_optimization).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="📈 Druckverlauf", command=self.run_pressure_plot).pack(side="left", padx=5, pady=2)
        ttk.Button(action_bar, text="⏱ Transient (Anlauf)", command=self.run_transient).pack(side="left", padx=5, pady=2)
        
        ttk.Button(action_bar, text="💾 Export Excel", command=self.export_to_excel).pack(side="left", padx=(20, 5), pady=2)

        self.vars = {}
        self.widgets = {}
        
        self.current_fig = None

        # Scrollbare Hauptfläche
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
        
        # Mausrad-Unterstützung
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Zweispaltiges Grid-Layout
        left_col = ttk.Frame(content_frame)
        left_col.grid(row=0, column=0, sticky="nw", padx=(0, 20))
        
        right_col = ttk.Frame(content_frame)
        right_col.grid(row=0, column=1, sticky="nsew", rowspan=2)

        content_frame.columnconfigure(1, weight=1) 
        right_col.columnconfigure(0, weight=1)     
        right_col.rowconfigure(2, weight=1)        

        # --- LINKE SPALTE: EINGABEN ---

        # 1. Betriebsparameter
        group_ops = ttk.LabelFrame(left_col, text=" 1. Betriebsparameter ", padding="15 10")
        group_ops.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_ops, "Fluid", "Water", ["Water", "Methanol", "Ammonia"])
        self.create_input_row(group_ops, "Wärmelast [W]", "100.0", None) 
        self.create_input_row(group_ops, "Drehzahl [U/min]", "24000", None)
        self.create_input_row(group_ops, "Temperatur [°C]", "60.0", None, info_cmd=self.show_temp_info)

        # 2. Geometrie
        group_geo = ttk.LabelFrame(left_col, text=" 2. Geometrie (Rohr) ", padding="15 10")
        group_geo.pack(fill="x", pady=(0, 15))
        
        btn_geo_info = ttk.Button(group_geo, text="ℹ Geometrie-Hilfe", command=self.show_geo_info, width=15)
        btn_geo_info.pack(anchor="e", pady=(0, 10))

        self.create_input_row(group_geo, "Länge L (Gesamt) [mm]", "250.0", None)
        self.create_input_row(group_geo, "L_Verdampfer [mm]", "50.0", None)
        self.create_input_row(group_geo, "L_Adiabat [mm]", "150.0", None)
        self.create_input_row(group_geo, "L_Kondensator [mm]", "50.0", None)
        ttk.Separator(group_geo, orient="horizontal").pack(fill="x", pady=10)
        self.create_input_row(group_geo, "D_aussen [mm]", "12.0", None)
        self.create_input_row(group_geo, "D_innen [mm]", "8.0", None)
        
        self.create_input_row(group_geo, "Wandstärke (auto) [mm]", "1.0", None)

        # 3. Struktur & Material
        group_struct = ttk.LabelFrame(left_col, text=" 3. Docht & Material ", padding="15 10")
        group_struct.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_struct, "Docht-Dicke [mm]", "1.0", None)
        self.create_input_row(group_struct, "Porenradius [µm]", "50.0", None)
        self.create_input_row(group_struct, "Permeabilität [1e-10 m²]", "1.0", None)
        self.create_input_row(group_struct, "Porosität Docht [0-1]", "0.5", None)
        self.create_input_row(group_struct, "Wärmeleitf. Wand [W/mK]", "380.0", None)
        
        # Automatische Berechnung der Wandstärke
        if "Wandstärke (auto) [mm]" in self.widgets:
            w_widget = self.widgets["Wandstärke (auto) [mm]"]
            
            style.map("Grey.TEntry", 
                      fieldbackground=[("readonly", "#e0e0e0")],  
                      foreground=[("readonly", "#333333")])       
            
            w_widget.configure(style="Grey.TEntry", state="readonly", cursor="no")
        
        self.vars["D_aussen [mm]"].trace("w", self.auto_calc_wall)
        self.vars["D_innen [mm]"].trace("w", self.auto_calc_wall)
        self.vars["Docht-Dicke [mm]"].trace("w", self.auto_calc_wall)


        # 4. Rotations-Setup
        group_rot = ttk.LabelFrame(left_col, text=" 4. Rotations-Setup ", padding="15 10")
        group_rot.pack(fill="x", pady=(0, 15))

        self.create_input_row(group_rot, "Rotations-Modus", "Exzentrisch", ["Exzentrisch", "Zentriert"])
        self.create_input_row(group_rot, "Achsversatz (r_min) [mm]", "15.0", None)
        self.create_input_row(group_rot, "Konuswinkel [deg]", "3.0", None)
        self.create_input_row(group_rot, "Neigungswinkel (Global) [deg]", "90.0", None)
        
        self.vars["Rotations-Modus"].trace("w", self.toggle_mode)

        # --- RECHTE SPALTE: VISUALISIERUNG & ERGEBNISSE ---
        
        # Geometrie-Vorschau
        self.sketch_frame = ttk.LabelFrame(right_col, text=" Geometrie-Vorschau (dient ausschließlich der groben Visualisierung) ", padding="10")
        self.sketch_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.geo_canvas = tk.Canvas(self.sketch_frame, width=500, height=120, bg="white", highlightthickness=0)
        self.geo_canvas.pack(fill="both", expand=True)

        # Ergebnis-Anzeige (Tabelle)
        self.group_res = ttk.LabelFrame(right_col, text=" Ergebnisse & Analyse ", padding="15 10")
        self.group_res.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        res_container = ttk.Frame(self.group_res)
        res_container.pack(fill="both", expand=True)

        btn_copy = ttk.Button(res_container, text="📋", width=3, command=self.copy_tree_value)
        btn_copy.pack(side="right", fill="y", padx=(2,0)) 

        text_scroll = ttk.Scrollbar(res_container)
        text_scroll.pack(side="right", fill="y")

        columns = ("Parameter", "Wert", "Einheit")
        self.res_tree = ttk.Treeview(res_container, columns=columns, show="headings", height=15, 
                                     yscrollcommand=text_scroll.set)
        
        self.res_tree.heading("Parameter", text="Parameter")
        self.res_tree.heading("Wert", text="Wert")
        self.res_tree.heading("Einheit", text="Einheit")
        
        self.res_tree.column("Parameter", width=200, anchor="w")
        self.res_tree.column("Wert", width=100, anchor="center")
        self.res_tree.column("Einheit", width=80, anchor="center")
        
        self.res_tree.pack(side="left", fill="both", expand=True)
        
        text_scroll.config(command=self.res_tree.yview)

        # Plot-Bereich
        self.plot_frame = ttk.LabelFrame(right_col, text=" Diagramme ", padding="5")
        self.plot_frame.grid(row=2, column=0, sticky="nsew")
        
        ttk.Label(self.plot_frame, text="Hier erscheinen die Plots nach der Berechnung", foreground="grey").pack(pady=20)
        
        ttk.Frame(left_col, height=50).pack(fill="x")

        # Initialisierung der Traces für Echtzeit-Validierung und Skizzen-Update
        sketch_vars = ["Länge L (Gesamt) [mm]", "L_Verdampfer [mm]", "L_Adiabat [mm]", "L_Kondensator [mm]", 
                       "D_aussen [mm]", "D_innen [mm]", "Docht-Dicke [mm]",
                       "Achsversatz (r_min) [mm]", "Konuswinkel [deg]", "Neigungswinkel (Global) [deg]"]
        
        for sv in sketch_vars:
            if sv in self.vars:
                self.vars[sv].trace("w", self.update_sketch)
                self.vars[sv].trace("w", self.perform_visual_validation)

        if "Länge L (Gesamt) [mm]" in self.vars:
            self.vars["Länge L (Gesamt) [mm]"].trace("w", self.auto_adjust_adiabat)

        self.root.after(100, self.update_sketch)
        self.root.after(100, self.perform_visual_validation)


    def copy_tree_value(self):
        """Kopiert den Inhalt der Ergebnistabelle in die Zwischenablage."""
        try:
            lines = ["Parameter\tWert\tEinheit"]
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
            messagebox.showinfo("Kopiert", "Tabelle wurde in die Zwischenablage kopiert.")
        except Exception:
            pass

    def create_input_row(self, parent, lbl, val, opts, info_cmd=None):
        """Erstellt eine standardisierte Eingabezeile mit Label und Eingabefeld."""
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
        """Lädt Benutzereinstellungen aus einer JSON-Datei."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    self.saved_data = json.load(f)
            except:
                pass 

    def on_closing(self):
        """Speichert die aktuellen Eingaben beim Schließen des Programms."""
        try:
            data = {k: v.get() for k, v in self.vars.items()}
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=4)
        except:
            pass
        self.root.destroy()
    
    def perform_visual_validation(self, *args):
        """Führt eine visuelle Validierung der Eingaben durch (Farbkodierung bei Fehlern)."""
        COLOR_OK = "black"
        COLOR_ERR = "red"

        # Längen-Validierung
        try:
            L = float(self.vars["Länge L (Gesamt) [mm]"].get())
            l_e = float(self.vars["L_Verdampfer [mm]"].get())
            l_a = float(self.vars["L_Adiabat [mm]"].get())
            l_c = float(self.vars["L_Kondensator [mm]"].get())

            if abs(L - (l_e + l_a + l_c)) > 0.1 or l_e < 0 or l_a < 0 or l_c < 0 or L <= 0:
                c_len = COLOR_ERR
            else:
                c_len = COLOR_OK
            
            self.widgets["Länge L (Gesamt) [mm]"].configure(foreground=c_len)
            self.widgets["L_Verdampfer [mm]"].configure(foreground=c_len)
            self.widgets["L_Adiabat [mm]"].configure(foreground=c_len)
            self.widgets["L_Kondensator [mm]"].configure(foreground=c_len)

        except ValueError:
            pass 

        # Durchmesser-Validierung
        try:
            da = float(self.vars["D_aussen [mm]"].get())
            di = float(self.vars["D_innen [mm]"].get())
            wick = float(self.vars["Docht-Dicke [mm]"].get())

            if di >= da or (da - di - 2*wick) <= 0 or da <= 0 or di <= 0:
                c_dia = COLOR_ERR
            else:
                c_dia = COLOR_OK
            
            self.widgets["D_aussen [mm]"].configure(foreground=c_dia)
            self.widgets["D_innen [mm]"].configure(foreground=c_dia)
            self.widgets["Docht-Dicke [mm]"].configure(foreground=c_dia)

        except ValueError:
            pass
    
    def auto_adjust_adiabat(self, *args):
        """Passt die Länge der Adiabatenzone automatisch an, wenn die Gesamtlänge geändert wird."""
        try:
            l_total = float(self.vars["Länge L (Gesamt) [mm]"].get())
            l_evap = float(self.vars["L_Verdampfer [mm]"].get())
            l_cond = float(self.vars["L_Kondensator [mm]"].get())
            
            new_adiabat = l_total - l_evap - l_cond
            
            if new_adiabat >= 0:
                self.vars["L_Adiabat [mm]"].set(f"{new_adiabat:.1f}")
        except ValueError:
            pass

    def update_sketch(self, *args):
        """Aktualisiert die geometrische Vorschau basierend auf den aktuellen Parametern."""
        try:
            l_evap = float(self.vars["L_Verdampfer [mm]"].get())
            l_adiab = float(self.vars["L_Adiabat [mm]"].get())
            l_cond = float(self.vars["L_Kondensator [mm]"].get())
            
            da = float(self.vars["D_aussen [mm]"].get())
            di_start = float(self.vars["D_innen [mm]"].get()) 
            wick_thickness = float(self.vars["Docht-Dicke [mm]"].get())
            
            r_min_val = float(self.vars["Achsversatz (r_min) [mm]"].get())
            cone_angle = float(self.vars["Konuswinkel [deg]"].get())
            inc_val = float(self.vars["Neigungswinkel (Global) [deg]"].get())

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

            # Rotationsachse zeichnen
            self.geo_canvas.create_line(10, axis_y, c_w - 10, axis_y, fill="black", dash=(10, 4), width=1)
            
            current_x_px = margin_x
            current_x_mm = 0.0
            
            sections = [
                (l_cond, "Kondensator", "#e3f2fd", "cond"), 
                (l_adiab, "Adiabat", "#f5f5f5", "adiab"),   
                (l_evap, "Verdampfer", "#ffebee", "evap")   
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
                
                # Rohrwand (oben und unten)
                self.geo_canvas.create_rectangle(x1, cy - r_o_px, x2, cy - r_wick_start_px, 
                                                 fill="#90a4ae", outline="black", tags="wall")
                self.geo_canvas.create_rectangle(x1, cy + r_wick_start_px, x2, cy + r_o_px, 
                                                 fill="#90a4ae", outline="black", tags="wall")

                # Dochtstruktur
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

                # Dampfkern
                pts_vapor = [
                    x1, cy - r_i_start_px,
                    x2, cy - r_i_end_px,
                    x2, cy + r_i_end_px,
                    x1, cy + r_i_start_px
                ]
                self.geo_canvas.create_polygon(pts_vapor, fill=col, outline="")

                # Vertikale Trennlinien
                self.geo_canvas.create_line(x1, cy - r_o_px, x1, cy + r_o_px, fill="#555")
                self.geo_canvas.create_line(x2, cy - r_o_px, x2, cy + r_o_px, fill="#555")

                if w_px > 30:
                    self.geo_canvas.create_text(x1 + w_px/2, cy, text=name, font=("Segoe UI", 7), fill="#333")

                # Bemaßung
                dim_y = axis_y + 10 
                self.geo_canvas.create_line(x1, dim_y, x2, dim_y, arrow=tk.BOTH, fill="#444", width=1)
                self.geo_canvas.create_text(x1 + w_px/2, dim_y + 8, text=f"{l_sect:.0f}", font=("Segoe UI", 7))

                current_x_px += w_px
                current_x_mm += l_sect

            # Deckel (visuelle Abschlüsse)
            start_x = margin_x
            end_x = current_x_px
            cy = axis_y - r_min_val * scale_y
            ro = (da/2.0) * scale_y
            
            self.geo_canvas.create_oval(start_x - 5, cy - ro, start_x + 5, cy + ro, fill="#cfd8dc", outline="black")
            r_end_total = ro 
            self.geo_canvas.create_oval(end_x - 5, cy - r_end_total, end_x + 5, cy + r_end_total, fill="#cfd8dc", outline="black")

            # Visualisierung des Neigungswinkels
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
        """Berechnet automatisch die Wandstärke basierend auf Außen-, Innendurchmesser und Dochtdicke."""
        try:
            s_da = self.vars["D_aussen [mm]"].get()
            s_di = self.vars["D_innen [mm]"].get()
            s_wick = self.vars["Docht-Dicke [mm]"].get()
            
            if not s_da or not s_di or not s_wick: return

            da = float(s_da)
            di = float(s_di)
            wick = float(s_wick)
            
            # Formel: D_aussen = D_innen + 2*Docht + 2*Wand
            wall = (da - di - 2 * wick) / 2.0
            
            self.vars["Wandstärke (auto) [mm]"].set(f"{wall:.2f}")
            
            if wall <= 0:
                self.widgets["Wandstärke (auto) [mm]"].configure(foreground="red")
            else:
                self.widgets["Wandstärke (auto) [mm]"].configure(foreground="#333333")
                
        except ValueError:
            pass 

    def toggle_mode(self, *args):
        """Schaltet die Eingabe des Achsversatzes je nach Rotationsmodus frei oder sperrt sie."""
        mode = self.vars["Rotations-Modus"].get()
        target_field = "Achsversatz (r_min) [mm]"
        
        if target_field not in self.widgets: return
        
        widget = self.widgets[target_field]
        var = self.vars[target_field]
        
        if mode == "Zentriert":
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
        """Zeigt Informationen zu kritischen Temperaturen der Arbeitsmedien."""
        msg = ("Temperatur-Grenzwert (Kritischer Punkt):\n\n"
               "Ammoniak: max. 131 °C\n"
               "Methanol: max. 238 °C\n"
               "Wasser:   max. 326 °C (Code-Limit)\n\n"
               "Über diesen Werten ist das Fluid überkritisch\n"
               "und keine Verdampfung mehr möglich.")
        messagebox.showinfo("Temperatur-Limits", msg)

    def show_assumptions(self):
        """Zeigt die dem Modell zugrundeliegenden Annahmen und Vereinfachungen."""
        info_text = (
            "Wesentliche Modell-Annahmen für die Validierung:\n\n"
            "1. Thermo-Mechanik:\n"
            "   Entkoppelte Berechnung von Druckbilanz (Limits) und thermischem Widerstand.\n\n"
            "2. Docht-Struktur (Hybrid-Ansatz):\n"
            "   Unterscheidung zwischen Darcy-Flow (im Sinter) und Annular-Flow (Film) bei hohen Drehzahlen.\n\n"
            "3. Phasenübergang:\n"
            "   Kondensations-HTC skaliert physikalisch korrekt mit Omega^2 (Zentrifugalkraft).\n"
            "   Siedegrenze berücksichtigt rotierenden hydrostatischen Druck.\n"
            "   Zusatz: Vernachlässigung der Grenzflächenschubspannung (Dampf-Gegenstrom) bei der Kondensations-Berechnung (Nusselt-Filmtheorie).\n\n"
            "4. Rotation:\n"
            "   Zentrifugalkraft pumpt Fluid vom Kondensator (r_min) zum Verdampfer (r_max).\n\n"
            "5. Dampfströmung (Coriolis nach Ansatz 2):\n"
            "   Integration der Coriolis-Effekte über die Rossby-Zahl direkt in den Reibungsfaktor."
        )
        messagebox.showinfo("Modell-Grundlagen & Annahmen", info_text)

    def show_geo_info(self):
        """Erklärt die geometrischen Zusammenhänge der Rohrabmessungen."""
        msg = (
            "Geometrie-Plausibilität:\n\n"
            "Die Wandstärke wird nun automatisch berechnet:\n"
            "  Wand = (D_aussen - D_innen - 2 * Docht) / 2\n\n"
            "Bitte stellen Sie sicher, dass D_aussen groß genug ist!"
        )
        messagebox.showinfo("Geometrie-Formel", msg)

    def check_geometry(self, da, di, t_wick, t_wall):
        """Überprüft die geometrische Konsistenz der Eingaben."""
        if t_wall <= 0:
            messagebox.showerror("Geometrie-Fehler", 
                                 f"Die berechnete Wandstärke ist {t_wall:.2f} mm (<= 0)!\n"
                                 "Bitte erhöhen Sie D_aussen oder verringern Sie D_innen/Docht.")
            return False
        return True

    def get_model_objects(self):
        """
        Extrahiert alle Eingabewerte, konvertiert Einheiten und initialisiert
        die Modell-Objekte für die Simulation.
        """
        v = {k: val.get() for k, val in self.vars.items()}
        
        da_check = float(v["D_aussen [mm]"])
        di_check = float(v["D_innen [mm]"])
        wick_check = float(v["Docht-Dicke [mm]"])
        wall_check = float(v["Wandstärke (auto) [mm]"])
        
        if not self.check_geometry(da_check, di_check, wick_check, wall_check):
            raise ValueError("Geometrie ungültig. Abbruch.")

        L_mm = float(v["Länge L (Gesamt) [mm]"])
        l_evap_mm = float(v["L_Verdampfer [mm]"])
        l_adiab_mm = float(v["L_Adiabat [mm]"])
        l_cond_mm = float(v["L_Kondensator [mm]"])

        if abs((l_evap_mm + l_adiab_mm + l_cond_mm) - L_mm) > 0.1:
            msg = (f"Summen-Fehler der Längen!\n"
                   f"Verdampfer ({l_evap_mm}) + Adiabat ({l_adiab_mm}) + Kondensator ({l_cond_mm})\n"
                   f"= {l_evap_mm + l_adiab_mm + l_cond_mm:.2f} mm\n"
                   f"!= Gesamtlänge {L_mm} mm\n\n"
                   "Bitte passen Sie die Teillängen an.")
            messagebox.showerror("Eingabefehler Länge", msg)
            raise ValueError("Längensumme ungültig. Abbruch.")

        # Umrechnung in SI-Einheiten
        L = L_mm / 1000.0
        l_evap_m = l_evap_mm / 1000.0
        l_adiab_m = l_adiab_mm / 1000.0
        l_cond_m = l_cond_mm / 1000.0

        Da = float(v["D_aussen [mm]"]) / 1000.0
        Di = float(v["D_innen [mm]"]) / 1000.0
        r_pore = float(v["Porenradius [µm]"]) * 1e-6
        perm = float(v["Permeabilität [1e-10 m²]"]) * 1e-10
        wick_thick = float(v["Docht-Dicke [mm]"]) / 1000.0
        porosity = float(v["Porosität Docht [0-1]"])
        k_solid = float(v["Wärmeleitf. Wand [W/mK]"])
        inc_deg = float(v["Neigungswinkel (Global) [deg]"])
        cone_deg = float(v["Konuswinkel [deg]"])
        
        r_min_val = float(v["Achsversatz (r_min) [mm]"]) / 1000.0
        
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
        """Exportiert die Simulationsergebnisse und Parameter in eine Excel-Datei."""
        if pd is None:
            messagebox.showerror("Bibliothek fehlt", "Pandas fehlt.")
            return
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperatur [°C]"]) + 273.15
            rpm_pt = float(v["Drehzahl [U/min]"])
            
            Q_input = float(v["Wärmelast [W]"])

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
            
            input_list = [{"Parameter": k, "Wert": val} for k, val in v.items()]
            df_inputs = pd.DataFrame(input_list)
            
            res_list = [
                {"Ergebnis": "Limitierender Faktor", "Wert": limiting_factor},
                {"Ergebnis": "Max. Leistung [W]", "Wert": max_Q},
                {"Ergebnis": "R_th [K/W] (bei Last)", "Wert": R_th},
                {"Ergebnis": "k_eff [W/mK]", "Wert": k_eff},
                {"Ergebnis": "Füllmenge (100%) [g]", "Wert": mass_charge_g},
                {"Ergebnis": "Calc: Filmdicke [mm]", "Wert": delta_film * 1000.0},
                {"Ergebnis": "Calc: Füllzustand", "Wert": status_film},
            ]
            
            for lname, lval in res_pt.items():
                if lval >= 1e8:
                    if lname == "Boiling":
                        val_to_write = "Nicht relevant (> 250g)"
                    elif lname == "Viscous":
                        val_to_write = "Nicht relevant (Viskos)"
                    else:
                        val_to_write = "Nicht relevant"
                else:
                    val_to_write = lval
                    
                res_list.append({"Ergebnis": f"Limit {lname} [W]", "Wert": val_to_write})
            
            df_results = pd.DataFrame(res_list)
            df_curves = pd.DataFrame(data_curves)

            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")], title="Ergebnisse exportieren")

            if not file_path: return

            with pd.ExcelWriter(file_path) as writer:
                df_inputs.to_excel(writer, sheet_name="Parameter", index=False)
                df_results.to_excel(writer, sheet_name="Ergebnisse_Punkt", index=False)
                df_curves.to_excel(writer, sheet_name="RPM_Kurven", index=False)
            
            messagebox.showinfo("Export Erfolgreich", f"Daten wurden gespeichert unter:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Fehler", str(e))

    def run_analysis(self):
        """Führt die stationäre Analyse durch und visualisiert die Leistungsgrenzen."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperatur [°C]"]) + 273.15
            rpm_pt = float(v["Drehzahl [U/min]"])
            
            Q_input = float(v["Wärmelast [W]"])
            
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
            
            # Ergebnisse anzeigen
            for item in self.res_tree.get_children():
                self.res_tree.delete(item)
            
            self.res_tree.insert("", "end", values=(f"Ergebnisse für {rpm_pt:.0f} U/min:", "", ""))
            self.res_tree.insert("", "end", values=("Limitierender Faktor", limiting_factor, ""))
            self.res_tree.insert("", "end", values=("Max. Leistung Q_max", f"{max_Q:.2f}", "W"))
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Füllmenge (100% Sättigung)", f"{mass_charge_g:.2f}", "g"))
            self.res_tree.insert("", "end", values=("Eff. Filmdicke", f"{delta_film*1e6:.1f}", "µm"))
            self.res_tree.insert("", "end", values=("Füllzustand", status_film, ""))
            
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Therm. Widerstand", f"{R_th:.4f}", "K/W"))
            self.res_tree.insert("", "end", values=("Eff. Wärmeleitfähigkeit", f"{k_eff:.1f}", "W/mK"))
            self.res_tree.insert("", "end", values=("Temp.-Diferenz bei Q_max", f"{dT_at_max:.1f}", "K"))
            
            self.res_tree.insert("", "end", values=("", "", ""))
            self.res_tree.insert("", "end", values=("Details Leistungsgrenzen:", "", ""))
            
            for k, val in res_pt.items():
                if val >= 1e8:
                    if k == "Boiling":
                        disp_val = "Nicht relevant (> 250g)"
                    elif k == "Viscous":
                        disp_val = "Nicht relevant (Viskos)"
                    else:
                        disp_val = "Nicht relevant"
                    disp_unit = "-"
                else:
                    disp_val = f"{val:.1f}"
                    disp_unit = "W"
                self.res_tree.insert("", "end", values=(f" - {k}", disp_val, disp_unit))

            # Berechnung der Kurvenverläufe
            rpms = np.linspace(0, 30000, 50)
            res_map = {"Capillary": [], "Boiling": [], "Sonic": [], "Entrainment": [], "Viscous": []}
            for r in rpms:
                res = limits.get_all_limits(T_k, r)
                for k in res_map:
                    res_map[k].append(res[k])
            
            # Plot-Vorbereitung
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
            
            ax.plot(rpms, res_map["Capillary"], label="Kapillarlimit", linewidth=2)
            ax.plot(rpms, res_map["Boiling"], label="Siedegrenze", linestyle='-.')
            ax.plot(rpms, res_map["Entrainment"], label="Mitrissgrenze", linestyle=':')
            ax.plot(rpms, res_map["Sonic"], label="Schallgrenze", linestyle='--')
            
            safe_q = np.minimum.reduce([res_map[k] for k in res_map])
            ax.fill_between(rpms, 0, safe_q, color='green', alpha=0.1, label="Sicherer Bereich")
            ax.plot(rpm_pt, max_Q, 'ro', markersize=8, label="Arbeitspunkt")
            
            ax.set_yscale("log")
            ax.set_ylim(1, 100000)
            ax.set_xlabel("Drehzahl [U/min]")
            ax.set_ylabel("Leistung [W]")
            ax.set_title("Betriebsgrenzen des rotierenden Wärmerohrs")
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
            messagebox.showerror("Fehler", str(e))

    def run_optimization(self):
        """Führt eine Parameteroptimierung für den Konuswinkel durch."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperatur [°C]"]) + 273.15
            rpm = float(v["Drehzahl [U/min]"])

            # Berechnung des maximal geometrisch möglichen Winkels
            r_out = params.d_out / 2.0
            r_in_start = params.d_in / 2.0
            min_wall_thickness = 0.0005 
            
            max_radial_growth = r_out - r_in_start - params.wick_thickness - min_wall_thickness
            
            if max_radial_growth <= 0:
                max_angle_deg = 0.1
                messagebox.showwarning("Geometrie-Warnung", 
                    "D_aussen ist zu klein für eine Konus-Optimierung.\n"
                    "Es ist kein Platz für eine Aufweitung vorhanden!")
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
            
            ax.set_xlabel("Konuswinkel [Grad]")
            ax.set_ylabel("Max Q [W]")
            ax.set_title(f"Optimierung (Geom. Limit: {max_angle_deg:.2f}°)")
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
        """Visualisiert den axialen Druckverlauf von Dampf und Flüssigkeit."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            T_k = float(v["Temperatur [°C]"]) + 273.15
            rpm = float(v["Drehzahl [U/min]"])
            
            res_pt = limits.get_all_limits(T_k, rpm)
            
            Q_limit = min(res_pt.values())
            Q_input = float(v["Wärmelast [W]"])
            
            Q_plot = min(Q_input, Q_limit) if Q_limit > 1.0 else Q_input
            
            m_dot = Q_plot / fluid.get_properties(T_k)['h_fg']
            
            z = np.linspace(0, params.length, 100)
            props = fluid.get_properties(T_k)
            
            # Dampfdruckverlauf (inkl. Reibungsverluste)
            dp_v = model.loss_vapor_advanced(m_dot, props['rho_v'], props['mu_v'], rpm)
            pv = props['p_sat'] - dp_v * (z/params.length) 
            
            omega = model.calc_omega(rpm)
            alpha = params.get_tilt_rad()
            
            r_z = params.r_min + (params.d_in / 2.0) + z * np.sin(alpha)
            
            r_start = params.r_min + (params.d_in / 2.0)
            
            # Flüssigkeitsdruckverlauf (Sättigung + Zentrifugaldruck - Reibung)
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

            ax.plot(z*1000, pv/1000, 'r--', label='Dampf')
            ax.plot(z*1000, pl/1000, 'b-', label='Flüssigkeit')
            
            ax.set_xlabel("z [mm] (Cond -> Evap)")
            ax.set_ylabel("Druck [kPa]")
            ax.set_title(f"Druckverlauf bei {Q_plot:.1f} W ({rpm:.0f} rpm)")
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
        """Simuliert das Anlaufverhalten über die Zeit."""
        try:
            params, fluid, model, limits, thermal, v = self.get_model_objects()
            
            t_max = 120.0  # Simulationsdauer [s]
            t_ramp = 5.0   # Zeit bis Solldrehzahl [s]
            
            target_rpm = float(v["Drehzahl [U/min]"])
            T_start = 20.0 + 273.15 
            
            Q_load = float(v["Wärmelast [W]"]) 
            
            R_ext_cooling = 0.05 
            T_amb = 20.0 + 273.15
            
            # Berechnung der thermischen Massen
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
            
            # Zeitschrittverfahren
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
            ax1.set_title(f'Transientes Anlaufverhalten (bei Wärmelast={Q_load} W)')
            ax1.grid(True)
            ax1.legend(fontsize='small')
            
            ax2.plot(t_hist, rpm_hist, 'k--')
            ax2.set_ylabel('RPM')
            ax2.set_xlabel('Zeit [s]')
            ax2.grid(True)
            
            fig.tight_layout()

            self.current_fig = fig

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()

        except Exception as e:
            messagebox.showerror("Fehler Transient", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = HeatPipeGUI(root)
    root.mainloop()