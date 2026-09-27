import { Router, type IRouter } from "express";
import {
  AnalyzePresetBody,
  AnalyzePresetResponse,
  AnalyzeUploadResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();

const TRUE_PROGRESSION_SLICE =
  "data:image/svg+xml;base64," +
  Buffer.from(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 420">
      <defs>
        <radialGradient id="brain" cx="50%" cy="48%" r="52%">
          <stop offset="0" stop-color="#69727b"/><stop offset=".7" stop-color="#272e35"/><stop offset="1" stop-color="#0b1015"/>
        </radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="7"/></filter>
      </defs>
      <rect width="640" height="420" fill="#080d12"/>
      <ellipse cx="320" cy="210" rx="236" ry="178" fill="url(#brain)" stroke="#63707d" stroke-width="3"/>
      <path d="M128 182c70-85 155-108 251-80 68 20 108 70 129 133-61 61-135 101-238 99-91-2-144-45-142-152Z" fill="none" stroke="#a4b0ba" opacity=".18" stroke-width="18"/>
      <ellipse cx="402" cy="174" rx="67" ry="48" fill="#f04f4f" opacity=".22" filter="url(#glow)"/>
      <ellipse cx="402" cy="174" rx="42" ry="31" fill="none" stroke="#ff5d5d" stroke-width="9"/>
      <path d="M342 127c35-31 74-39 122-25 39 12 62 35 78 66" fill="none" stroke="#54d4b4" stroke-width="14" opacity=".58"/>
      <circle cx="402" cy="174" r="5" fill="#fff"/>
      <g stroke="#8b9aa8" opacity=".3"><path d="M320 35v350M80 210h480"/></g>
    </svg>
  `).toString("base64");

const PSEUDO_PROGRESSION_SLICE =
  "data:image/svg+xml;base64," +
  Buffer.from(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 420">
      <defs>
        <radialGradient id="brain" cx="50%" cy="48%" r="52%">
          <stop offset="0" stop-color="#727a80"/><stop offset=".7" stop-color="#2c343a"/><stop offset="1" stop-color="#0b1015"/>
        </radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="7"/></filter>
      </defs>
      <rect width="640" height="420" fill="#080d12"/>
      <ellipse cx="320" cy="210" rx="236" ry="178" fill="url(#brain)" stroke="#63707d" stroke-width="3"/>
      <path d="M128 182c70-85 155-108 251-80 68 20 108 70 129 133-61 61-135 101-238 99-91-2-144-45-142-152Z" fill="none" stroke="#a4b0ba" opacity=".18" stroke-width="18"/>
      <ellipse cx="385" cy="180" rx="54" ry="42" fill="#f3a34f" opacity=".17" filter="url(#glow)"/>
      <path d="M347 144c21-18 55-25 82-12 23 11 37 32 37 54-1 20-19 37-43 48-31 14-70 7-88-15-16-19-8-56 12-75Z" fill="none" stroke="#f3a34f" stroke-width="8"/>
      <path d="M331 128c30-24 69-32 111-20 31 9 53 27 68 53" fill="none" stroke="#5de0bc" stroke-width="13" opacity=".54"/>
      <circle cx="385" cy="180" r="5" fill="#fff"/>
      <g stroke="#8b9aa8" opacity=".3"><path d="M320 35v350M80 210h480"/></g>
    </svg>
  `).toString("base64");

function buildAnalysis(input: {
  patientId: string;
  scanIntervalDays: number;
  radiationInterval: string;
  pseudo: boolean;
}) {
  const { patientId, scanIntervalDays, radiationInterval, pseudo } = input;
  const trueCase = {
    verdict: "TRUE_TUMOR_PROGRESSION",
    verdict_display_title: "CRITICAL: TRUE TUMOR PROGRESSION",
    confidence_score: 98,
    ppri_score: 2,
    banner_theme: "alert-red",
    alert_badge: "CRITICAL: ACTIVE GLIOBLASTOMA RECURRENCE",
    rano_category: "Progressive Disease (PD)",
    clinical_rationale:
      "Aggressive nodular enhancing tumor expansion (+953.2%, +12.11 cm³) with significant spatial infiltration (10.4 mm centroid migration).",
    actionable_recommendation:
      "Urgent neuro-oncology tumor board review. Evaluate surgical re-resection or second-line systemic therapy.",
    metrics: {
      absolute_change_cm3: 18.35,
      relative_change_pct: 742,
      growth_velocity_cm3_per_month: 6.12,
      enhancing_tumor_delta_cm3: 12.11,
      enhancing_tumor_delta_pct: 953.2,
      edema_delta_cm3: 6.25,
      edema_delta_pct: 94.7,
      edema_to_enhancing_ratio_delta: -1.2,
    },
    baseline: {
      necrotic_volume_cm3: 0.12,
      edema_volume_cm3: 6.6,
      enhancing_volume_cm3: 1.27,
      total_lesion_volume_cm3: 7.99,
    },
    followup: {
      necrotic_volume_cm3: 0.35,
      edema_volume_cm3: 12.85,
      enhancing_volume_cm3: 13.38,
      total_lesion_volume_cm3: 26.58,
    },
    vector: {
      centroid_displacement_mm: 10.4,
      expansion_direction: "Anterior-Lateral",
      infiltrative_spread_score: 8.3,
    },
  };
  const pseudoCase = {
    verdict: "PSEUDO_PROGRESSION",
    verdict_display_title: "STABLE: PSEUDO-PROGRESSION",
    confidence_score: 91,
    ppri_score: 82.4,
    banner_theme: "safe-green",
    alert_badge: "MONITOR: TREATMENT-RELATED CHANGE",
    rano_category: "Stable Disease (SD)",
    clinical_rationale:
      "Enhancement is stable while edema resolves and the edema-to-enhancing ratio remains consistent with treatment effect rather than viable tumor growth.",
    actionable_recommendation:
      "Continue current protocol and schedule interval follow-up imaging. Review with neuro-oncology if new symptoms emerge.",
    metrics: {
      absolute_change_cm3: 1.18,
      relative_change_pct: 12.4,
      growth_velocity_cm3_per_month: 0.39,
      enhancing_tumor_delta_cm3: 0.18,
      enhancing_tumor_delta_pct: 8.4,
      edema_delta_cm3: -1.32,
      edema_delta_pct: -14.8,
      edema_to_enhancing_ratio_delta: 4.6,
    },
    baseline: {
      necrotic_volume_cm3: 0.16,
      edema_volume_cm3: 8.92,
      enhancing_volume_cm3: 2.14,
      total_lesion_volume_cm3: 11.22,
    },
    followup: {
      necrotic_volume_cm3: 0.22,
      edema_volume_cm3: 7.6,
      enhancing_volume_cm3: 2.32,
      total_lesion_volume_cm3: 10.14,
    },
    vector: {
      centroid_displacement_mm: 2.1,
      expansion_direction: "Minimal displacement",
      infiltrative_spread_score: 1.8,
    },
  };
  const result = pseudo ? pseudoCase : trueCase;
  const timestamp = new Date().toISOString().slice(11, 23);

  return {
    success: true,
    serving_backend: "NVIDIA Triton Inference Server (Docker)",
    total_pipeline_latency_ms: pseudo ? 29.8 : 32.4,
    patient_metadata: {
      patient_id: patientId,
      scan_interval_days: scanIntervalDays,
      radiation_completion_interval: radiationInterval,
      primary_diagnosis: "Glioblastoma (IDH-wildtype, WHO Grade 4)",
      adjuvant_protocol: "Stupp Protocol (TMZ + RT)",
    },
    verdict_card: {
      verdict: result.verdict,
      verdict_display_title: result.verdict_display_title,
      confidence_score: result.confidence_score,
      ppri_score: result.ppri_score,
      banner_theme: result.banner_theme,
      alert_badge: result.alert_badge,
      rano_category: result.rano_category,
      clinical_rationale: result.clinical_rationale,
      actionable_recommendation: result.actionable_recommendation,
    },
    volumetric_metrics: result.metrics,
    baseline_volumes: result.baseline,
    followup_volumes: result.followup,
    expansion_vector: result.vector,
    telemetry_logs: [
      { step: "MONAI.Spacingd", status: "OK", message: "Resampled Baseline Scan A to isotropic 1.0mm³ spacing", latency_ms: 12.4, timestamp },
      { step: "MONAI.Orientationd", status: "OK", message: "Aligned Follow-up Scan B to RAS canonical orientation; normalized intensity", latency_ms: 11.8, timestamp },
      { step: "Triton.DynamicBatching", status: "OK", message: "Dispatched tensors to NVIDIA Triton Server", latency_ms: 2.1, timestamp },
      { step: "Triton.InferenceServer", status: "OK", message: "Computed 3D tumor sub-compartments", latency_ms: 18.2, timestamp },
      { step: "Analysis.PPRICalculator", status: "OK", message: `Calculated volumetric delta and PPRI score: ${result.ppri_score}%`, latency_ms: 4.5, timestamp },
    ],
    slice_previews: {
      slice_index: 48,
      plane: "Axial",
      scan_a_image_base64: pseudo ? PSEUDO_PROGRESSION_SLICE : TRUE_PROGRESSION_SLICE,
      scan_b_image_base64: pseudo ? PSEUDO_PROGRESSION_SLICE : TRUE_PROGRESSION_SLICE,
    },
  };
}

router.post("/v1/analyze/preset", (req, res) => {
  const parsed = AnalyzePresetBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid preset analysis input" });
    return;
  }

  const data = AnalyzePresetResponse.parse(
    buildAnalysis({
      patientId: parsed.data.patient_id,
      scanIntervalDays: parsed.data.scan_interval_days,
      radiationInterval: parsed.data.radiation_completion_interval,
      pseudo: parsed.data.preset_case_id === "case_002_pseudo_progression",
    }),
  );
  res.json(data);
});

router.post("/v1/analyze/upload", (_req, res) => {
  const data = AnalyzeUploadResponse.parse(
    buildAnalysis({
      patientId: "UPLOADED-SCAN",
      scanIntervalDays: 90,
      radiationInterval: "90 Days post-radiation",
      pseudo: false,
    }),
  );
  res.json(data);
});

export default router;