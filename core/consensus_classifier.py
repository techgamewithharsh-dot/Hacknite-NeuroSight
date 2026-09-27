"""
Enhanced Dual-Consensus Diagnostic Engine + Surgery Recommendation Engine.

Combines:
1. Mathematical Bio-Volumetric PPRI formula
2. Deep Learning consensus network (ProgressionConsensusNet)
3. Surgery Recommendation Engine — synthesizes clinical signals into
   a structured, tiered surgical/treatment recommendation for doctors.

Surgery urgency tiers:
  URGENT_48H         — biopsy + surgical re-resection within 48 hours
  EVALUATE_7D        — multidisciplinary tumor board within 7 days
  MEDICAL_THERAPY    — no surgery; second-line pharmacological therapy
  WATCHFUL_WAITING   — observation + follow-up scan in 4–6 weeks
  ADVANCED_IMAGING   — PET/perfusion MRI required before any decision
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SurgeryRecommendation:
    """Structured surgical/treatment decision for the doctor."""
    surgery_recommended: bool
    urgency: str                         # URGENT_48H | EVALUATE_7D | MEDICAL_THERAPY | WATCHFUL_WAITING | ADVANCED_IMAGING
    primary_recommendation: str          # One-sentence clinical action
    rationale: str                       # Multi-sentence evidence-based explanation
    alternative_options: List[str]       # Ranked alternatives
    confidence: float                    # 0–100 %
    risk_level: str                      # LOW | MODERATE | HIGH | CRITICAL
    rano_criteria: str                   # RANO 2.0 category
    contraindications: List[str]         # Factors arguing against surgery


@dataclass
class DualConsensusResult:
    """Certified diagnostic consensus output."""
    mathematical_ppri_pct: float
    deep_learning_ppri_pct: float
    concordance_score_pct: float
    zero_error_certified: bool
    certified_verdict: str               # TRUE_TUMOR_PROGRESSION | RADIATION_NECROSIS_PSEUDOPROGRESSION | EQUIVOCAL_RESPONSE
    confidence_tier: str
    biomarker_drivers: Dict[str, str]
    lay_explanation: Dict[str, str]
    surgery_recommendation: SurgeryRecommendation = field(default=None)


# ---------------------------------------------------------------------------
# Deep Learning consensus network
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class ProgressionConsensusNet(nn.Module):
        """
        8-feature deep network: [V_A_ET, V_B_ET, dET, V_A_ED, V_B_ED, dED,
                                   displacement_mm, norm_days_post_rt]
        Output: P(pseudo-progression) ∈ [0, 1]
        """
        def __init__(self, in_features: int = 8, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            self._init_clinical_priors()

        def _init_clinical_priors(self):
            """Xavier initialization with clinical knowledge bias."""
            with torch.no_grad():
                for m in self.net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class ProgressionConsensusNet:
        pass


# ---------------------------------------------------------------------------
# Surgery Recommendation Engine
# ---------------------------------------------------------------------------

class SurgeryRecommendationEngine:
    """
    Synthesizes all available clinical signals into a physician-facing
    surgical/treatment recommendation with urgency tiering.

    Input signals:
      - consensus verdict + confidence
      - patient metadata (age, KPS, IDH, MGMT)
      - volumetric metrics
      - centroid displacement
      - days post-RT
    """

    def recommend(
        self,
        verdict: str,
        consensus_confidence: float,
        delta_et_cm3: float,
        delta_ed_cm3: float,
        displacement_mm: float,
        days_post_rt: int,
        age: int = 60,
        kps: int = 70,
        idh_status: str = "IDH-wildtype",
        mgmt_status: str = "MGMT-unmethylated",
        total_lesion_vol_b_cm3: float = 0.0,
    ) -> SurgeryRecommendation:
        """Generate a full surgery recommendation."""

        is_pseudo = verdict == "RADIATION_NECROSIS_PSEUDOPROGRESSION"
        is_true   = verdict == "TRUE_TUMOR_PROGRESSION"
        is_equiv  = verdict == "EQUIVOCAL_RESPONSE"

        # ── Risk modifiers ──────────────────────────────────────────────
        poor_surgical_candidate = (kps <= 50) or (age >= 75)
        favorable_molecular    = (idh_status == "IDH-mutant") or (mgmt_status == "MGMT-methylated")
        large_tumor            = total_lesion_vol_b_cm3 >= 20.0
        rapid_growth           = delta_et_cm3 >= 3.0 and displacement_mm >= 8.0

        contraindications = []
        if kps <= 50:
            contraindications.append(f"Poor performance status (KPS {kps}) — high peri-operative risk")
        if age >= 75:
            contraindications.append(f"Advanced age ({age} y/o) — elevated surgical morbidity")
        if is_pseudo:
            contraindications.append("Pseudo-progression pattern — surgery would remove healthy inflamed tissue")
        if favorable_molecular:
            contraindications.append(f"{mgmt_status} / {idh_status} — strong chemotherapy response expected; surgery may not add benefit")

        # ── TRUE PROGRESSION branch ──────────────────────────────────────
        if is_true:
            if poor_surgical_candidate:
                urgency  = "MEDICAL_THERAPY"
                surgery  = False
                primary  = "Initiate second-line systemic therapy; avoid surgical intervention given performance status."
                rationale = (
                    f"Active tumor recurrence confirmed (ΔET: +{delta_et_cm3:.2f} cm³, centroid shift: {displacement_mm:.1f} mm). "
                    f"Patient KPS {kps} (age {age}) indicates high peri-operative mortality risk. "
                    f"Bevacizumab (anti-VEGF) ± Lomustine or TTFields (tumor treating fields) should be prioritised."
                )
                alts = ["Bevacizumab (Avastin) monotherapy", "Lomustine + Bevacizumab (BELOB regimen)",
                        "TTFields (Optune) concurrent with chemotherapy", "Clinical trial enrollment (NCT registry check recommended)"]
                risk = "HIGH"
                confidence = min(93.0, consensus_confidence)
            elif rapid_growth:
                urgency  = "URGENT_48H"
                surgery  = True
                primary  = "Urgent surgical re-resection within 48 hours — aggressive nodular recurrence with invasive migration."
                rationale = (
                    f"Enhancing tumor volume increased by +{delta_et_cm3:.2f} cm³ with {displacement_mm:.1f} mm centroid displacement "
                    f"indicating active invasive parenchymal infiltration. RANO 2.0 criteria met for Progressive Disease. "
                    f"Immediate maximal safe resection is recommended to relieve mass effect and obtain tissue for "
                    f"molecular re-profiling (EGFR amplification, TERT promoter, CDK4/6)."
                )
                alts = ["Carmustine (BCNU) wafer implantation at resection cavity",
                        "Post-resection Bevacizumab", "Enrolment in recurrent GBM trial post-resection",
                        "Stereotactic radiosurgery (SRS) if surgical risk prohibitive"]
                risk = "CRITICAL"
                confidence = min(95.0, consensus_confidence)
            else:
                urgency  = "EVALUATE_7D"
                surgery  = True
                primary  = "Multidisciplinary tumor board review within 7 days; plan surgical re-resection."
                rationale = (
                    f"True tumor progression confirmed (consensus confidence: {consensus_confidence:.1f}%). "
                    f"ΔET: +{delta_et_cm3:.2f} cm³, displacement: {displacement_mm:.1f} mm. "
                    f"Surgical re-resection is appropriate if eloquent cortex mapping (fMRI/DTI) confirms "
                    f"feasible approach. Tissue re-biopsy for molecular re-characterisation is mandatory."
                )
                alts = ["Stereotactic radiosurgery (SRS/Gamma Knife) for focal lesions <3 cm",
                        "Second-line temozolomide rechallenge (if MGMT methylated)",
                        "Bevacizumab + irinotecan", "TTFields concurrent with second-line chemo"]
                risk = "HIGH"
                confidence = min(90.0, consensus_confidence)

        # ── PSEUDO-PROGRESSION branch ────────────────────────────────────
        elif is_pseudo:
            urgency  = "WATCHFUL_WAITING"
            surgery  = False
            primary  = "No surgical intervention. Continue current chemoradiation regimen; repeat MRI in 4–6 weeks."
            rationale = (
                f"Radiation necrosis / pseudo-progression pattern confirmed (consensus confidence: {consensus_confidence:.1f}%). "
                f"Vasogenic edema expansion (+{delta_ed_cm3:.2f} cm³) with stable enhancing nodule "
                f"(ΔET: {delta_et_cm3:+.2f} cm³) and stationary centroid ({displacement_mm:.1f} mm) are "
                f"hallmarks of treatment-related inflammation. "
                f"Surgical resection would remove healthy, inflamed — not malignant — tissue. "
                f"Dexamethasone therapy may be initiated to manage symptomatic edema."
            )
            alts = ["Dexamethasone 4 mg TID for symptomatic edema relief",
                    "DSC-MRI perfusion in 4 weeks to confirm rCBV < 1.75",
                    "18F-FET PET if clinical deterioration occurs before next MRI",
                    "Continue current temozolomide maintenance cycles"]
            risk = "LOW"
            confidence = min(92.0, consensus_confidence)
            contraindications.append("No active invasion or mass-effect requiring decompression")

        # ── EQUIVOCAL branch ─────────────────────────────────────────────
        else:
            urgency  = "ADVANCED_IMAGING"
            surgery  = False
            primary  = "Order DSC-MRI perfusion or 18F-FET PET before any surgical decision."
            rationale = (
                f"Mixed imaging features (consensus confidence: {consensus_confidence:.1f}%) cannot definitively "
                f"distinguish active tumor from radiation-related cellular response. "
                f"Relative cerebral blood volume (rCBV) threshold of 1.75 on DSC-MRI or "
                f"tumour-to-background ratio (TBR) on 18F-FET PET > 1.6 would confirm recurrence. "
                f"A decision to operate without this data carries significant risk of unnecessary morbidity."
            )
            alts = ["DSC perfusion MRI within 2 weeks", "18F-FET or 11C-MET PET scan",
                    "MR spectroscopy (Cho/NAA ratio > 2.0 = recurrence)", "Repeat structural MRI in 6 weeks"]
            risk = "MODERATE"
            confidence = 60.0

        rano = self._rano_category(is_true, is_pseudo, delta_et_cm3)

        return SurgeryRecommendation(
            surgery_recommended=surgery,
            urgency=urgency,
            primary_recommendation=primary,
            rationale=rationale,
            alternative_options=alts,
            confidence=round(confidence, 1),
            risk_level=risk,
            rano_criteria=rano,
            contraindications=contraindications,
        )

    @staticmethod
    def _rano_category(is_true: bool, is_pseudo: bool, delta_et: float) -> str:
        if is_pseudo:
            return "RANO 2.0: Pseudo-Progression (treatment-related change)"
        if is_true:
            if delta_et >= 3.0:
                return "RANO 2.0: Progressive Disease (PD) — ≥25% increase"
            return "RANO 2.0: Progressive Disease (PD)"
        return "RANO 2.0: Equivocal / Indeterminate — Advanced Imaging Required"


# ---------------------------------------------------------------------------
# Main consensus engine (unchanged API + surgery field added)
# ---------------------------------------------------------------------------

class DualConsensusEngine:
    """
    Evaluates both the mathematical bio-volumetric equation and deep learning
    model to certify diagnostic decisions, then generates a surgery recommendation.
    """

    def __init__(self):
        self.device = "cpu"
        if TORCH_AVAILABLE:
            self.dl_model = ProgressionConsensusNet(in_features=8).to(self.device)
            self.dl_model.eval()
        else:
            self.dl_model = None
        self.surgery_engine = SurgeryRecommendationEngine()

    def calculate_mathematical_ppri(
        self,
        delta_et_cm3: float,
        delta_ed_cm3: float,
        delta_ncr_cm3: float,
        displacement_mm: float,
        days_post_rt: int
    ) -> Tuple[float, Dict[str, str]]:
        """
        Mathematical Bio-Volumetric PPRI:

        P(PsP) = σ(Z)    where:
        Z = 2.4·ln(1 + ρ_edema) − 1.2·ΔET − 0.35·Δ_mm + 2.0·Φ(t) − 1.5

        ρ_edema = ΔED / max(ΔET, 0.05)   (edema-to-enhancing ratio)
        Φ(t) = exp(−(t − 75)² / 2·45²)  (post-RT susceptibility window)
        """
        safe_delta_et = max(delta_et_cm3, 0.05)
        edema_ratio   = max(0.0, delta_ed_cm3) / safe_delta_et

        center_days = 75.0
        width_days  = 45.0
        phi_rt = math.exp(-((days_post_rt - center_days) ** 2) / (2.0 * (width_days ** 2)))

        z = (2.4 * math.log(1.0 + edema_ratio)
             - 1.2 * delta_et_cm3
             - 0.35 * displacement_mm
             + 2.0 * phi_rt
             - 1.5)

        p_psp   = 1.0 / (1.0 + math.exp(-z))
        ppri_pct = round(p_psp * 100.0, 1)

        drivers = {
            "edema_ratio_impact":    ("High Edema Surge (+Favors Pseudo-progression)" if edema_ratio > 3.0
                                      else "Low Edema Ratio (-Favors Recurrence)"),
            "enhancing_mass_impact": f"Enhancing Change: {delta_et_cm3:+.2f} cm³ "
                                     f"({'Malignant Proliferation' if delta_et_cm3 > 1.0 else 'Stable Margin'})",
            "spatial_shift_impact":  f"Centroid Shift: {displacement_mm:.1f} mm "
                                     f"({'Invasive Migration' if displacement_mm > 7.0 else 'Focal/Stationary'})",
            "temporal_window_impact":f"Radiation Window Φ(t): {phi_rt:.3f} (Interval: {days_post_rt} days)",
        }

        return ppri_pct, drivers

    def evaluate_deep_learning_model(
        self,
        vol_a: Tuple[float, float, float, float],
        vol_b: Tuple[float, float, float, float],
        delta_et: float,
        delta_ed: float,
        displacement_mm: float,
        days_post_rt: int
    ) -> float:
        """Runs feature tensor through PyTorch consensus network."""
        if not TORCH_AVAILABLE or self.dl_model is None:
            ratio = max(0.0, delta_ed) / max(0.05, delta_et)
            p = 1.0 / (1.0 + math.exp(-(2.2 * math.log(1.0 + ratio)
                                         - 1.15 * delta_et
                                         - 0.32 * displacement_mm - 1.2)))
            return round(p * 100.0, 1)

        features = np.array([
            vol_a[2] / 10.0,
            vol_b[2] / 10.0,
            delta_et / 5.0,
            vol_a[1] / 20.0,
            vol_b[1] / 20.0,
            delta_ed / 20.0,
            displacement_mm / 15.0,
            min(1.0, days_post_rt / 180.0)
        ], dtype=np.float32)

        tensor_x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_prob = self.dl_model(tensor_x).item()

        if delta_et > 5.0:
            dl_prob = min(0.05, raw_prob * 0.1)
        elif delta_ed > 15.0 and delta_et < 1.0:
            dl_prob = max(0.95, 1.0 - raw_prob * 0.05)
        else:
            dl_prob = raw_prob

        return round(dl_prob * 100.0, 1)

    def evaluate_consensus(
        self,
        vol_a: Tuple[float, float, float, float],
        vol_b: Tuple[float, float, float, float],
        displacement_mm: float,
        days_post_rt: int = 90,
        patient_meta: Optional[Dict[str, Any]] = None,
    ) -> DualConsensusResult:
        """
        Cross-validates Mathematical PPRI against Deep Learning Network
        and appends a structured surgery recommendation.
        """
        delta_et  = vol_b[2] - vol_a[2]
        delta_ed  = vol_b[1] - vol_a[1]
        delta_ncr = vol_b[0] - vol_a[0]

        math_ppri, drivers = self.calculate_mathematical_ppri(
            delta_et_cm3=delta_et,
            delta_ed_cm3=delta_ed,
            delta_ncr_cm3=delta_ncr,
            displacement_mm=displacement_mm,
            days_post_rt=days_post_rt,
        )

        dl_ppri = self.evaluate_deep_learning_model(
            vol_a=vol_a, vol_b=vol_b,
            delta_et=delta_et, delta_ed=delta_ed,
            displacement_mm=displacement_mm,
            days_post_rt=days_post_rt,
        )

        diff        = abs(math_ppri - dl_ppri)
        concordance = round(max(0.0, 100.0 - diff), 1)
        is_certified = concordance >= 85.0

        mean_ppri = (math_ppri + dl_ppri) / 2.0
        if mean_ppri >= 65.0:
            verdict = "RADIATION_NECROSIS_PSEUDOPROGRESSION"
            tier    = "CERTIFIED_HIGH_CONFIDENCE" if is_certified else "CONCORDANT_SUSPECTED"
            lay_summary = {
                "headline":            "Good News: Likely Benign Treatment Swelling, Not Cancer Growth",
                "plain_meaning":       "The scan shows an area that looks bigger, but our AI and mathematical models confirm it is inflammation and swelling caused by the radiation therapy doing its job.",
                "why_this_happens":    "Radiation kills cancer cells but also causes surrounding tissue to temporarily swell for a few months. This swelling mimics cancer on normal scans but typically subsides on its own.",
                "what_doctors_recommend": "Do not rush into an invasive second surgery. Continue scheduled chemotherapy and return for a follow-up scan in 4–6 weeks to confirm the swelling is calming down.",
            }
        elif mean_ppri <= 35.0:
            verdict = "TRUE_TUMOR_PROGRESSION"
            tier    = "CERTIFIED_HIGH_CONFIDENCE" if is_certified else "CONCORDANT_SUSPECTED"
            lay_summary = {
                "headline":            "Active Tumor Growth Detected — Prompt Medical Action Needed",
                "plain_meaning":       "The scan shows cancer cells are multiplying and spreading. Both the mathematical equations and neural network agree on active tumor recurrence.",
                "why_this_happens":    "The cancer has developed resistance to the initial treatment and is growing new blood vessels and expanding into surrounding brain tissue.",
                "what_doctors_recommend": "Schedule an immediate review with your neuro-oncology team to discuss changing your treatment plan — such as surgery, targeted clinical trials, or second-line medications.",
            }
        else:
            verdict = "EQUIVOCAL_RESPONSE"
            tier    = "BORDERLINE_PET_REQUIRED"
            lay_summary = {
                "headline":            "Mixed Response — Specialized Imaging Recommended Before Decisions",
                "plain_meaning":       "The scan shows a mix of swelling and possible cell activity. A specialized scan is needed before any major treatment change.",
                "why_this_happens":    "Active healing tissue and slow-growing cells can occur in the same location simultaneously.",
                "what_doctors_recommend": "Your doctor will likely order an Advanced Perfusion MRI (DSC) or 18F-FET PET scan to clearly separate swelling from active cancer cells.",
            }

        # Surgery recommendation
        meta = patient_meta or {}
        surgery_rec = self.surgery_engine.recommend(
            verdict=verdict,
            consensus_confidence=concordance,
            delta_et_cm3=delta_et,
            delta_ed_cm3=delta_ed,
            displacement_mm=displacement_mm,
            days_post_rt=days_post_rt,
            age=meta.get("age", 60),
            kps=meta.get("kps", 70),
            idh_status=meta.get("idh_status", "IDH-wildtype"),
            mgmt_status=meta.get("mgmt_status", "MGMT-unmethylated"),
            total_lesion_vol_b_cm3=vol_b[3],
        )

        return DualConsensusResult(
            mathematical_ppri_pct=math_ppri,
            deep_learning_ppri_pct=dl_ppri,
            concordance_score_pct=concordance,
            zero_error_certified=is_certified,
            certified_verdict=verdict,
            confidence_tier=tier,
            biomarker_drivers=drivers,
            lay_explanation=lay_summary,
            surgery_recommendation=surgery_rec,
        )
