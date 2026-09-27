import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import Login from '@/pages/login';
import Register from '@/pages/register';
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  Crosshair,
  Database,
  FileImage,
  FlaskConical,
  Gauge,
  HeartPulse,
  Info,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  MousePointer2,
  Minus,
  Network,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  Terminal,
  UploadCloud,
  X,
  Zap,
} from 'lucide-react';
import {
  getAnalyzerHealthQueryKey,
  getHealthCheckQueryKey,
  useAnalyzerHealth,
  useAnalyzePreset,
  useHealthCheck,
} from '@workspace/api-client-react';
import type { AnalysisResult, PresetAnalysisInputPresetCaseId } from '@workspace/api-client-react';
import {
  Route,
  Switch,
  useLocation,
  Router as WouterRouter,
  Link,
} from 'wouter';

const queryClient = new QueryClient();

// Simple authentication state - in production, use proper auth system
function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const login = () => setIsAuthenticated(true);
  const logout = () => setIsAuthenticated(false);

  return { isAuthenticated, login, logout };
}

// Protected route component with saved device log verification
function ProtectedRoute({ children }: { children: ReactNode }) {
  const [, setLocation] = useLocation();
  const [isChecking, setIsChecking] = useState(true);
  const [isAllowed, setIsAllowed] = useState(false);

  useMemo(() => {
    // Check if user is authenticated locally
    if (localStorage.getItem('isAuthenticated') === 'true') {
      setIsAllowed(true);
      setIsChecking(false);
      return;
    }

    // Check device log with backend
    let deviceId = localStorage.getItem('savedDeviceId');
    if (!deviceId) {
      deviceId = 'dev_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('savedDeviceId', deviceId);
    }

    fetch(`/api/auth/saved-device?device_id=${encodeURIComponent(deviceId)}`)
      .then(res => res.json())
      .then(data => {
        if (data && data.saved) {
          localStorage.setItem('isAuthenticated', 'true');
          localStorage.setItem('userEmail', data.user.email);
          localStorage.setItem('userData', JSON.stringify(data.user));
          setIsAllowed(true);
          setIsChecking(false);
        } else {
          setIsAllowed(false);
          setIsChecking(false);
          setLocation('/login');
        }
      })
      .catch(() => {
        setIsAllowed(false);
        setIsChecking(false);
        setLocation('/login');
      });
  }, [setLocation]);

  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <p className="text-xs text-muted-foreground font-mono">Verifying saved device authorization from log file...</p>
        </div>
      </div>
    );
  }

  if (!isAllowed) {
    return null;
  }

  return <>{children}</>;
}

function Home() {
  const [mode, setMode] = useState<'preset' | 'upload'>('preset');
  const [, setLocation] = useLocation();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showTelemetry, setShowTelemetry] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [scanInterval, setScanInterval] = useState('84');
  const [radiationInterval, setRadiationInterval] = useState('42 days');
  const [presetCase, setPresetCase] = useState<PresetAnalysisInputPresetCaseId>('case_001_true_progression');
  const [mriFiles, setMriFiles] = useState<Record<string, File | null>>({});
  const setMriFile = (key: string, file: File | null) => setMriFiles((current) => ({ ...current, [key]: file }));
  const sequences = ['t1ce', 't1', 't2', 'flair'] as const;
  const [isUploading, setIsUploading] = useState(false);
  const [formError, setFormError] = useState('');
  const presetMutation = useAnalyzePreset();
  const analyzerHealth = useAnalyzerHealth({
    query: { queryKey: getAnalyzerHealthQueryKey(), refetchInterval: 30000 },
  });
  const baseHealth = useHealthCheck({
    query: { queryKey: getHealthCheckQueryKey(), refetchInterval: 30000 },
  });

  const isPending = presetMutation.isPending || isUploading;
  const health = analyzerHealth.data;
  const healthLabel = analyzerHealth.isLoading
    ? 'Checking service'
    : health?.status === 'online'
      ? 'Analyzer online'
      : 'Analyzer attention';

  async function submitAnalysis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError('');
    const interval = Number(scanInterval);
    if (!patientId.trim()) {
      setFormError('Patient ID is required before analysis can begin.');
      return;
    }
    if (!Number.isFinite(interval) || interval < 15 || interval > 360) {
      setFormError('Scan interval must be between 15 and 360 days.');
      return;
    }
    if (mode === 'upload' && sequences.some((sequence) => !mriFiles[`a_${sequence}`] || !mriFiles[`b_${sequence}`])) {
      setFormError('Select all four NIfTI sequences (T1c, T1, T2, FLAIR) for both timepoints.');
      return;
    }
    if (mode === 'preset') {
      presetMutation.mutate({
        data: {
          preset_case_id: presetCase,
          patient_id: patientId.trim(),
          scan_interval_days: interval,
          radiation_completion_interval: radiationInterval.trim(),
        },
      }, {
        onSuccess: (data) => setResult(data),
        onError: () => setFormError('The preset analysis could not be completed. Review the inputs and retry.'),
      });
    } else {
      const formData = new FormData();
      sequences.forEach((sequence) => {
        formData.append(`scan_a_${sequence}`, mriFiles[`a_${sequence}`]!);
        formData.append(`scan_b_${sequence}`, mriFiles[`b_${sequence}`]!);
      });
      formData.append('patient_id', patientId.trim());
      formData.append('scan_interval_days', String(interval));
      formData.append('radiation_interval', radiationInterval.trim());
      setIsUploading(true);
      try {
        const response = await fetch('/api/v1/analyze/upload', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          const detail = typeof payload?.detail === 'string' ? payload.detail : `The analyzer returned HTTP ${response.status}.`;
          throw new Error(detail);
        }
        setResult((await response.json()) as AnalysisResult);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown upload error.';
        setFormError(`Upload analysis failed: ${message.slice(0, 420)}`);
      } finally {
        setIsUploading(false);
      }
    }
  }

  return (
    <div className="min-h-[100dvh] bg-background text-foreground">
      <header className="glass-panel relative z-30 h-[68px] border-b border-border bg-card/95 backdrop-blur-sm">
        <div className="mx-auto flex h-full max-w-[1560px] items-center justify-between px-5 lg:px-8">
          <div className="flex items-center gap-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
              <ScanLine size={19} strokeWidth={2.4} />
            </div>
            <div>
              <div className="flex items-baseline gap-2">
                <span className="font-['Space_Grotesk'] text-[16px] font-bold tracking-[-0.03em]">NeuroSight</span>
                <span className="mono text-[10px] uppercase tracking-[.16em] text-muted-foreground">MRI analyzer</span>
              </div>
              <p className="hidden text-[11px] text-muted-foreground sm:block">Evidence-first progression review room</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-full border border-border bg-secondary/55 px-3 py-1.5 text-[11px] sm:flex">
              <span className={`h-1.5 w-1.5 rounded-full ${health?.status === 'online' && health.triton_connected ? 'bg-[#2b9b84]' : 'bg-accent'}`} />
              <span className="text-muted-foreground" data-testid="status-analyzer">{healthLabel}</span>
            </div>
            <button type="button" onClick={() => setShowHelp(!showHelp)} className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground" data-testid="button-help" aria-label="Open clinical help" aria-expanded={showHelp}>
              <CircleHelp size={17} />
            </button>
            <button
              type="button"
              onClick={() => {
                localStorage.removeItem('isAuthenticated');
                localStorage.removeItem('userEmail');
                setLocation('/login');
              }}
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              aria-label="Sign out"
            >
              <LogOut size={17} />
            </button>
            <div className="flex h-8 w-8 items-center justify-center rounded-full border border-primary/20 bg-primary/10 text-[11px] font-bold text-primary" data-testid="text-reviewer-avatar">NR</div>
          </div>
        </div>
        {showHelp && <div className="absolute right-5 top-[60px] z-20 w-[min(320px,calc(100vw-40px))] rounded-lg border border-border bg-card p-4 text-[12px] leading-5 text-muted-foreground shadow-[0_14px_35px_rgba(24,53,65,.14)] lg:right-8" data-testid="panel-clinical-help"><div className="mb-1 flex items-center justify-between font-semibold text-foreground"><span>Review room guide</span><button type="button" onClick={() => setShowHelp(false)} className="rounded p-0.5 hover:bg-secondary" data-testid="button-close-help" aria-label="Close clinical help"><X size={14} /></button></div><p>Compare the registered slices first, then use the volumetric and directional evidence to frame the model verdict. Telemetry remains available below the result for traceability.</p></div>}
      </header>

      <main className="clinical-grid mx-auto max-w-[1560px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <div className="mb-7 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
          <div className="animate-rise">
            <div className="mb-2 flex items-center gap-2 section-label"><span className="h-1.5 w-1.5 rounded-full bg-accent" /> Workstation / comparative review</div>
            <h1 className="font-['Space_Grotesk'] text-[clamp(28px,4vw,42px)] font-semibold leading-[1.05] tracking-[-0.045em]">Two timepoints.<br /><span className="text-primary">One clinical read.</span></h1>
            <p className="mt-3 max-w-[590px] text-[14px] leading-6 text-muted-foreground">Run the same evidence chain every time: align the scans, quantify the change, then review the model’s reasoning before bringing it to conference.</p>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3.5 py-3 text-[11px] text-muted-foreground shadow-[var(--shadow-soft)]">
            <LockKeyhole size={14} className="text-primary" />
            <span>Protected clinical workspace</span>
            <span className="mx-1 h-3 w-px bg-border" />
            <span className="mono" data-testid="text-backend">{health?.serving_backend ?? '—'}</span>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(330px,390px)_minmax(0,1fr)]">
          <section className="glass-panel motion-lift animate-rise rounded-xl border border-border bg-card shadow-[var(--shadow-soft)]" style={{ animationDelay: '80ms' }}>
            <div className="border-b border-border px-5 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="section-label mb-1">Analysis intake</div>
                  <h2 className="font-['Space_Grotesk'] text-[18px] font-semibold tracking-[-0.025em]">Configure review</h2>
                </div>
                <SlidersHorizontal size={17} className="text-muted-foreground" />
              </div>
              <div className="mt-4 grid grid-cols-2 rounded-lg border border-border bg-secondary/50 p-1">
                <button type="button" onClick={() => setMode('preset')} className={`rounded-md px-2 py-2 text-[12px] font-semibold transition-colors ${mode === 'preset' ? 'bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}`} data-testid="button-mode-preset">Preset case</button>
                <button type="button" onClick={() => setMode('upload')} className={`rounded-md px-2 py-2 text-[12px] font-semibold transition-colors ${mode === 'upload' ? 'bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}`} data-testid="button-mode-upload">Upload scans</button>
              </div>
            </div>
            <form onSubmit={submitAnalysis} className="space-y-5 px-5 py-5">
              {mode === 'preset' ? (
                <Field label="Preset case" hint="Reference pathway">
                  <select value={presetCase} onChange={(event) => setPresetCase(event.target.value as any)} className="field-input" data-testid="select-preset-case">
                    <option value="case_001_true_progression">PT-001 · James R. (62y · True Progression · Recurrence)</option>
                    <option value="case_002_pseudo_progression">PT-002 · Sarah M. (47y · Pseudo-Progression · Radiation Necrosis)</option>
                    <option value="case_003_true_progression">PT-003 · Michael B. (55y · True Progression · Expanding ET)</option>
                    <option value="case_004_pseudo_progression">PT-004 · Elena K. (68y · Pseudo-Progression · Edema Flare)</option>
                    <option value="case_005_true_progression">PT-005 · David L. (51y · True Progression · Infiltrative)</option>
                    <option value="case_006_pseudo_progression">PT-006 · Maria S. (63y · Pseudo-Progression · Stable ET)</option>
                    <option value="case_007_true_progression">PT-007 · Robert T. (59y · True Progression · Invasive)</option>
                    <option value="case_008_pseudo_progression">PT-008 · Lisa W. (44y · Pseudo-Progression · Necrosis)</option>
                    <option value="case_009_true_progression">PT-009 · Christopher P. (71y · True Progression)</option>
                    <option value="case_010_pseudo_progression">PT-010 · Patricia H. (58y · Pseudo-Progression)</option>
                  </select>
                </Field>
              ) : (
                <div className="space-y-2">
                  <Field label="Scan timepoints" hint="NIfTI files">
                    <div className="grid gap-3">
                      {(['a', 'b'] as const).map((timepoint) => <div key={timepoint} className="rounded-md border border-border/70 p-2.5">
                        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[.08em] text-muted-foreground">{timepoint === 'a' ? 'Baseline · A' : 'Follow-up · B'}</div>
                        <div className="grid grid-cols-2 gap-2">
                          {sequences.map((sequence) => <FilePicker key={`${timepoint}_${sequence}`} label={sequence === 't1ce' ? 'T1 + contrast' : sequence.toUpperCase()} file={mriFiles[`${timepoint}_${sequence}`] ?? null} onChange={(file) => setMriFile(`${timepoint}_${sequence}`, file)} testId={`input-${timepoint}-${sequence}`} />)}
                        </div>
                      </div>)}
                    </div>
                  </Field>
                </div>
              )}
              <Field label="Patient ID" hint="Required">
                <input value={patientId} onChange={(event) => setPatientId(event.target.value)} placeholder="e.g. NS-2048" className="field-input" data-testid="input-patient-id" />
              </Field>
              <Field label="Scan interval" hint="15–360 days">
                <div className="relative">
                  <input type="number" min="15" max="360" value={scanInterval} onChange={(event) => setScanInterval(event.target.value)} className="field-input pr-16" data-testid="input-scan-interval" />
                  <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 mono text-[11px] text-muted-foreground">days</span>
                </div>
              </Field>
              <Field label="Radiation completion interval" hint="Required">
                <input value={radiationInterval} onChange={(event) => setRadiationInterval(event.target.value)} placeholder="e.g. 42 days" className="field-input" data-testid="input-radiation-interval" />
              </Field>
              {formError && <div className="flex items-start gap-2 rounded-md border border-destructive/20 bg-destructive/5 px-3 py-2.5 text-[12px] leading-5 text-destructive" data-testid="status-form-error"><AlertCircle size={15} className="mt-0.5 shrink-0" />{formError}</div>}
              <button type="submit" disabled={isPending} className="group flex w-full items-center justify-between rounded-lg bg-primary px-4 py-3.5 text-[13px] font-semibold text-primary-foreground shadow-sm transition-transform hover:-translate-y-0.5 disabled:opacity-60" data-testid="button-run-analysis">
                <span className="flex items-center gap-2">{isPending ? <LoaderCircle size={16} className="animate-spin" /> : <FlaskConical size={16} />}{isPending ? (mode === 'upload' ? 'Segmenting 3D MRI volumes' : 'Processing evidence chain') : 'Run comparative analysis'}</span>
                {!isPending && <ChevronRight size={17} className="transition-transform group-hover:translate-x-0.5" />}
              </button>
              {mode === 'upload' && <p className="text-center text-[10px] leading-4 text-muted-foreground">Full-volume CPU segmentation can take several minutes. The service remains available while each scan is processing.</p>}
              <p className="text-center text-[10px] leading-4 text-muted-foreground">Analysis is deterministic for the selected inputs. No clinical conclusion is stored by this workstation.</p>
            </form>
          </section>

          <section className="min-w-0">
            {result ? <Results result={result} showTelemetry={showTelemetry} setShowTelemetry={setShowTelemetry} /> : <EmptyReview health={health} baseHealth={baseHealth.data} onLoadPreset={() => { setPresetCase('case_001_true_progression'); setPatientId('NS-DEMO-001'); }} />}
          </section>
        </div>
      </main>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return <label className="block"><span className="mb-1.5 flex items-center justify-between text-[12px] font-semibold text-foreground"><span>{label}</span><span className="font-normal text-muted-foreground">{hint}</span></span>{children}</label>;
}

function FilePicker({ label, file, onChange, testId }: { label: string; file: File | null; onChange: (file: File | null) => void; testId: string }) {
  return <label className="group flex min-h-[86px] cursor-pointer flex-col justify-between rounded-lg border border-dashed border-border bg-secondary/30 p-3 transition-colors hover:border-primary/50 hover:bg-primary/5" data-testid={`${testId}-label`}>
    <input type="file" accept=".nii,.nii.gz,application/octet-stream" className="sr-only" onChange={(event) => onChange(event.target.files?.[0] ?? null)} data-testid={testId} />
    <span className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[.08em] text-muted-foreground"><span>{label}</span><UploadCloud size={14} className="text-primary" /></span>
    <span className="truncate text-[11px] text-foreground">{file?.name ?? 'Choose NIfTI file'}</span>
  </label>;
}

function EmptyReview({ health, baseHealth, onLoadPreset }: { health?: { status?: string; triton_connected?: boolean; serving_backend?: string }; baseHealth?: { status?: string }; onLoadPreset: () => void }) {
  return <div className="animate-rise space-y-5" style={{ animationDelay: '160ms' }}>
    <div className="relative min-h-[370px] overflow-hidden rounded-xl border border-[#244b56] bg-[#102c35] p-6 text-[#dcebea] shadow-[0_16px_36px_rgba(16,44,53,.12)] scan-surface glass-panel">
      <div className="ambient-float absolute -right-10 -top-10 h-64 w-64 rounded-full border border-[#39747a]/30" /><div className="ambient-float absolute -right-1 top-0 h-80 w-80 rounded-full border border-[#39747a]/20" /><div className="ambient-float absolute right-20 top-12 h-36 w-36 rounded-full border border-[#39747a]/25" />
      <div className="relative z-10 flex h-full min-h-[318px] flex-col justify-between">
        <div className="flex items-start justify-between"><div><div className="section-label text-[#83bdbd]">Awaiting timepoints</div><h2 className="mt-2 max-w-[420px] font-['Space_Grotesk'] text-[30px] font-semibold leading-[1.05] tracking-[-0.045em]">Your evidence<br />will appear here.</h2></div><div className="rounded-md border border-[#39747a]/50 bg-[#183d45] px-2.5 py-1.5 mono text-[10px] text-[#9acbc8]">READY / IDLE</div></div>
        <div><p className="max-w-[460px] text-[13px] leading-6 text-[#a8c5c4]">Select a reference case or bring two NIfTI timepoints into the room. The review surface keeps volume change, directional spread, and model telemetry together.</p><button type="button" onClick={onLoadPreset} className="mt-5 flex items-center gap-2 rounded-md border border-[#5e9b9b] bg-[#28545b] px-3.5 py-2.5 text-[12px] font-semibold text-[#e4f2ef] transition-colors hover:bg-[#32656b]" data-testid="button-load-demo">Load a reference case <ChevronRight size={15} /></button></div>
      </div>
    </div>
    <div className="grid gap-3 sm:grid-cols-3">
      <HealthCard icon={<HeartPulse size={16} />} label="API gateway" value={baseHealth?.status ?? 'Checking'} detail="Health endpoint" />
      <HealthCard icon={<Network size={16} />} label="Serving backend" value={health?.serving_backend ?? 'Not connected'} detail={health?.triton_connected ? 'Triton connected' : health?.status === 'online' ? 'Local MONAI fallback active' : 'Awaiting analyzer'} />
      <HealthCard icon={<ShieldCheck size={16} />} label="Review protocol" value="RANO aligned" detail="Evidence chain v1.0" />
    </div>
  </div>;
}

function HealthCard({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail: string }) {
  return <div className="glass-panel motion-lift rounded-lg border border-border bg-card px-4 py-3.5 shadow-[var(--shadow-soft)]"><div className="flex items-center gap-2 text-primary">{icon}<span className="section-label">{label}</span></div><div className="mt-2 text-[13px] font-semibold" data-testid={`status-${label.replace(/\s/g, '-').toLowerCase()}`}>{value}</div><div className="mt-0.5 text-[11px] text-muted-foreground">{detail}</div></div>;
}

function Results({ result, showTelemetry, setShowTelemetry }: { result: AnalysisResult; showTelemetry: boolean; setShowTelemetry: (value: boolean) => void }) {
  const verdict = result.verdict_card;
  const researchOnly = Boolean((result as any).slice_previews?.spatial_reference);
  const verdictText = verdict.verdict.toLowerCase();
  const isProgression = verdictText.includes('progression') && !verdictText.includes('pseudo');
  const total = Math.max(result.baseline_volumes.total_lesion_volume_cm3, result.followup_volumes.total_lesion_volume_cm3, 1);
  const volumeRows: Array<[string, number, number]> = [
    ['Enhancing', result.baseline_volumes.enhancing_volume_cm3, result.followup_volumes.enhancing_volume_cm3],
    ['Edema', result.baseline_volumes.edema_volume_cm3, result.followup_volumes.edema_volume_cm3],
    ['Necrotic', result.baseline_volumes.necrotic_volume_cm3, result.followup_volumes.necrotic_volume_cm3],
    ['Total lesion', result.baseline_volumes.total_lesion_volume_cm3, result.followup_volumes.total_lesion_volume_cm3],
  ];
  const maxLatency = Math.max(...result.telemetry_logs.map((log) => log.latency_ms), 1);
  const formattedTimestamp = useMemo(() => result.telemetry_logs.at(-1)?.timestamp ?? '—', [result.telemetry_logs]);
  return <div className="animate-rise space-y-5">
    <div className={`rounded-xl border p-5 shadow-[var(--shadow-soft)] neon-glow ${isProgression ? 'border-[#8c5b2e] bg-[#1a1008]' : 'border-[#176d58] bg-[#071611]'}`} data-testid="card-verdict">
      <div className="flex flex-col justify-between gap-5 md:flex-row md:items-start">
        <div className="flex gap-3.5"><div className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${researchOnly ? 'bg-[#3a2b12] text-[#ffca7a]' : isProgression ? 'bg-[#3a210d] text-[#ffb454]' : 'bg-[#0e3d2f] text-[#54e3bb]'}`}>{researchOnly || isProgression ? <AlertCircle size={21} /> : <Check size={21} />}</div><div><div className="section-label mb-1">{researchOnly ? 'Research segmentation · not a diagnosis' : `Model assessment · ${verdict.rano_category}`}</div><h2 className="font-['Space_Grotesk'] text-[25px] font-semibold leading-tight tracking-[-0.04em]" data-testid="text-verdict-title">{verdict.verdict_display_title}</h2><p className="mt-1.5 max-w-[570px] text-[13px] leading-5 text-muted-foreground">{verdict.clinical_rationale}</p></div></div>
        <div className="shrink-0 md:text-right">{researchOnly ? <div className="max-w-[185px] rounded-md border border-[#6f5424] bg-[#241c0c] px-3 py-2 text-[11px] leading-4 text-[#ffda94]">No diagnostic confidence is provided for these predictions.</div> : <><div className="section-label">Confidence</div><div className="mt-1 font-['Space_Grotesk'] text-[30px] font-semibold tracking-[-0.05em]" data-testid="text-confidence">{verdict.confidence_score.toFixed(1)}<span className="text-[17px]">%</span></div><div className="mono text-[10px] text-muted-foreground">PPRI {verdict.ppri_score.toFixed(3)}</div></>}</div>
      </div>
      <div className="mt-5 grid gap-3 border-t border-current/10 pt-4 md:grid-cols-[1fr_auto] md:items-center"><div><div className="section-label mb-1">Recommended next step</div><p className="text-[13px] font-medium">{verdict.actionable_recommendation}</p></div><span className={`w-fit rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[.08em] ${isProgression ? 'bg-[#3a210d] text-[#ffb454]' : 'bg-[#0e3d2f] text-[#54e3bb]'}`} data-testid="status-alert-badge">{verdict.alert_badge}</span></div>
    </div>

    {/* Surgery Recommendation Engine Card */}
    {(result as any).surgery_recommendation && !researchOnly && (
      <div className={`rounded-xl border p-5 shadow-[var(--shadow-soft)] neon-glow ${
        (result as any).surgery_recommendation.surgery_recommended
          ? 'border-[#ff4d4d]/60 bg-[#1e0707]'
          : 'border-[#20d8d0]/60 bg-[#06181b]'
      }`} data-testid="card-surgery-recommendation">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
          <div>
            <div className="section-label mb-1 flex items-center gap-2 text-primary">
              <ShieldCheck size={14} className="text-primary" />
              <span>SURGERY RECOMMENDATION ENGINE · RANO 2.0 PROTOCOL</span>
            </div>
            <h3 className="font-['Space_Grotesk'] text-[21px] font-bold text-foreground">
              {(result as any).surgery_recommendation.primary_recommendation}
            </h3>
            <p className="mt-2 max-w-[700px] text-[13px] leading-5 text-muted-foreground">
              {(result as any).surgery_recommendation.rationale}
            </p>
            {(result as any).surgery_recommendation.alternative_options && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                <span className="text-[11px] text-muted-foreground mr-1">Alternatives:</span>
                {(result as any).surgery_recommendation.alternative_options.map((alt: string, i: number) => (
                  <span key={i} className="rounded bg-secondary/80 px-2 py-0.5 text-[10px] font-medium text-foreground">
                    {alt}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="shrink-0 flex flex-col items-end gap-2">
            <span className={`rounded-full px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider ${
              (result as any).surgery_recommendation.surgery_recommended
                ? 'bg-destructive text-destructive-foreground'
                : 'bg-primary text-primary-foreground'
            }`}>
              {(result as any).surgery_recommendation.urgency}
            </span>
            <span className="mono text-[11px] text-muted-foreground">
              Risk: {(result as any).surgery_recommendation.risk_level} · Confidence: {(result as any).surgery_recommendation.confidence}%
            </span>
            {(result as any).patient_meshes && (
              <span className="rounded bg-[#0c242c] border border-primary/20 px-2 py-1 mono text-[10px] text-[#55f1d5]">
                3D Mesh: {(result as any).patient_meshes.brain?.vertex_count?.toLocaleString()} vertices
              </span>
            )}
          </div>
        </div>
      </div>
    )}

    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(280px,.85fr)]">
      <div className="glass-panel motion-lift rounded-xl border border-border bg-card shadow-[var(--shadow-soft)]">
        <div className="flex items-center justify-between border-b border-border px-5 py-4"><div><div className="section-label mb-1">Slice comparison</div><h3 className="font-['Space_Grotesk'] text-[17px] font-semibold">Axial · slice {result.slice_previews.slice_index}</h3></div><span className="rounded-md bg-secondary px-2 py-1 mono text-[10px] text-muted-foreground">Δ {result.expansion_vector.centroid_displacement_mm.toFixed(1)} mm</span></div>
        <div className="grid grid-cols-2 gap-2 bg-[#07161b] p-2">
          <Preview image={result.slice_previews.scan_a_image_base64} label="Baseline · A" />
          <Preview image={result.slice_previews.scan_b_image_base64} label="Follow-up · B" />
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 px-5 py-3 text-[11px] text-muted-foreground"><span className="flex flex-wrap items-center gap-3"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#e74c3c]" /> enhancing</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#2ecc71]" /> edema</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#3498db]" /> tumor core</span></span><span className="mono">registered / same plane</span></div>
      </div>
      <div className="glass-panel motion-lift rounded-xl border border-border bg-card shadow-[var(--shadow-soft)]">
        <div className="border-b border-border px-5 py-4"><div className="section-label mb-1">Patient context</div><h3 className="font-['Space_Grotesk'] text-[17px] font-semibold">Review metadata</h3></div>
        <div className="divide-y divide-border px-5">
          <InfoRow label="Patient ID" value={result.patient_metadata.patient_id} mono />
          <InfoRow label="Primary diagnosis" value={result.patient_metadata.primary_diagnosis} />
          <InfoRow label="Adjuvant protocol" value={result.patient_metadata.adjuvant_protocol} />
          <InfoRow label="Scan interval" value={`${result.patient_metadata.scan_interval_days} days`} mono />
          <InfoRow label="Radiation interval" value={result.patient_metadata.radiation_completion_interval} />
        </div>
      </div>
    </div>

    <BrainModelPanel result={result} />

    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(270px,.75fr)]">
      <div className="glass-panel motion-lift rounded-xl border border-border bg-card shadow-[var(--shadow-soft)]"><div className="flex items-center justify-between border-b border-border px-5 py-4"><div><div className="section-label mb-1">Volumetric evidence</div><h3 className="font-['Space_Grotesk'] text-[17px] font-semibold">Segmented volume change</h3></div><span className={`flex items-center gap-1 text-[12px] font-semibold ${result.volumetric_metrics.relative_change_pct >= 0 ? 'text-[#ffb454]' : 'text-[#54e3bb]'}`}>{result.volumetric_metrics.relative_change_pct >= 0 ? <ArrowUpRight size={15} /> : <ArrowDownRight size={15} />}{Math.abs(result.volumetric_metrics.relative_change_pct).toFixed(1)}%</span></div><div className="space-y-4 px-5 py-5">{volumeRows.map(([name, baseline, followup]) => <VolumeRow key={name} name={name} baseline={baseline} followup={followup} total={total} />)}</div></div>
      <div className="glass-panel motion-lift rounded-xl border border-border bg-card shadow-[var(--shadow-soft)]"><div className="border-b border-border px-5 py-4"><div className="section-label mb-1">{researchOnly ? 'Mask geometry' : 'Directional signal'}</div><h3 className="font-['Space_Grotesk'] text-[17px] font-semibold">{researchOnly ? 'Centroid shift' : 'Expansion vector'}</h3></div><div className="space-y-5 px-5 py-5"><div><div className="section-label">{researchOnly ? 'RAS direction' : 'Spread direction'}</div><div className="mt-1 text-[15px] font-semibold">{result.expansion_vector.expansion_direction}</div></div><div className="grid grid-cols-2 gap-3"><Signal label="Centroid displacement" value={`${result.expansion_vector.centroid_displacement_mm.toFixed(1)} mm`} /><Signal label={researchOnly ? 'Infiltration score' : 'Infiltrative spread'} value={researchOnly ? 'Not assessed' : result.expansion_vector.infiltrative_spread_score.toFixed(2)} /></div><div className="rounded-md bg-secondary/65 p-3 text-[11px] leading-5 text-muted-foreground"><Info size={14} className="mr-1 inline text-primary" /> {researchOnly ? 'Geometric shift between predicted masks; it is not a tumor-growth diagnosis.' : 'Directional evidence is supplementary to volumetric and clinical review.'}</div></div></div>
    </div>

    <div className="overflow-hidden rounded-xl border border-[#1b4b50] bg-[#05090d] shadow-[var(--shadow-soft)] neon-glow"><button type="button" onClick={() => setShowTelemetry(!showTelemetry)} className="flex w-full items-center justify-between px-5 py-4 text-left transition-colors hover:bg-[#0b181d]" data-testid="button-toggle-telemetry"><span className="flex items-center gap-3"><span className="flex h-8 w-8 items-center justify-center rounded-md bg-[#0b252b] text-[#5ce8d5]"><Terminal size={16} /></span><span><span className="section-label block mb-1 text-[#57d8d0]">Traceable inference</span><span className="font-['Space_Grotesk'] text-[16px] font-semibold text-[#e8ffff]">Pipeline telemetry / terminal</span></span></span><span className="flex items-center gap-3 text-[11px] text-[#6d9da0]"><span className="hidden sm:inline mono">{result.telemetry_logs.length} steps · {result.total_pipeline_latency_ms} ms total</span>{showTelemetry ? <X size={16} /> : <ChevronRight size={17} />}</span></button>{showTelemetry && <div className="border-t border-[#173a40]"><TelemetryTerminal logs={result.telemetry_logs} totalLatency={result.total_pipeline_latency_ms} timestamp={formattedTimestamp} maxLatency={maxLatency} /></div>}</div>
    <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-[10px] text-muted-foreground"><span className="flex items-center gap-1.5"><Database size={12} /> Serving backend: <span className="mono" data-testid="text-result-backend">{result.serving_backend}</span></span><span className="flex items-center gap-1.5"><Gauge size={12} /> Total pipeline latency <span className="mono">{result.total_pipeline_latency_ms} ms</span></span></div>
  </div>;
}

function BrainModelPanel({ result }: { result: AnalysisResult }) {
  const modelCanvasRef = useRef<HTMLDivElement>(null);
  const patientId = result.patient_metadata.patient_id;
  const patientMeshes = (result as any).patient_meshes;
  const isProgression = result.verdict_card.verdict.includes('TRUE');

  useEffect(() => {
    const host = modelCanvasRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, host.clientWidth / Math.max(host.clientHeight, 1), 0.01, 100);
    camera.position.set(0, 0.15, 3.4);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xc5e4ef, 0x171d2a, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 3.0);
    key.position.set(2, 3, 4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x72c8d8, 1.6);
    fill.position.set(-3, -1, 1);
    scene.add(fill);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.45;
    controls.minDistance = 1.8;
    controls.maxDistance = 5.5;
    let frame = 0;
    const sceneMeshes: THREE.Mesh[] = [];
    const addPatientSurface = (data: any, color: string, opacity: number, side: THREE.Side = THREE.FrontSide) => {
      if (!data?.vertices?.length || !data?.faces?.length) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
      if (data.normals?.length) geometry.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
      geometry.setIndex(data.faces);
      if (!data.normals?.length) geometry.computeVertexNormals();
      const material = new THREE.MeshPhysicalMaterial({ color, roughness: 0.72, metalness: 0, transparent: opacity < 1, opacity, side, depthWrite: opacity >= 0.9 });
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);
      sceneMeshes.push(mesh);
    };
    // All surfaces are extracted from the same follow-up T1c 1mm patient grid.
    // Their shared voxel-to-world transform makes the overlays spatially registered.
    addPatientSurface(patientMeshes?.head, '#d4c3b3', 0.10, THREE.DoubleSide);
    addPatientSurface(patientMeshes?.brain, '#d7b993', 0.22, THREE.DoubleSide);
    addPatientSurface(patientMeshes?.edema, '#41c9ab', 0.60, THREE.DoubleSide);
    addPatientSurface(patientMeshes?.necrotic_core, '#ffc14a', 0.88, THREE.DoubleSide);
    addPatientSurface(patientMeshes?.enhancing_tumor, '#ff315c', 0.96, THREE.DoubleSide);
    const resize = () => {
      if (!host) return;
      camera.aspect = host.clientWidth / Math.max(host.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
    };
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    window.addEventListener('resize', resize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      controls.dispose();
      renderer.dispose();
      if (host.contains(renderer.domElement)) host.removeChild(renderer.domElement);
      sceneMeshes.forEach((mesh) => {
        scene.remove(mesh);
        mesh.geometry.dispose();
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach((material) => material.dispose());
      });
    };
  }, [patientId, patientMeshes]);

  return (
    <section className="overflow-hidden rounded-xl border border-[#1b4b50] bg-[#050b10] shadow-[var(--shadow-soft)] neon-glow" data-testid="card-brain-model">
      <div className="flex flex-col justify-between gap-3 border-b border-[#173a40] px-5 py-4 sm:flex-row sm:items-center">
        <div>
          <div className="section-label mb-1 text-[#57d8d0]">Neural volume field / live canvas</div>
          <h3 className="font-['Space_Grotesk'] text-[18px] font-semibold tracking-[-0.025em] text-[#e8ffff]">Head, neck & lesion location</h3>
        </div>
        <div className="flex items-center gap-2 text-[10px]">
          <span className="flex items-center gap-1.5 rounded-full border border-[#245d60] bg-[#092126] px-2.5 py-1.5 text-[#8ee9df]"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#55f1d5]" /> LIVE VOLUME FIELD</span>
        </div>
      </div>
      <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div
          className="brain-stage relative min-h-[340px] overflow-hidden border-b border-[#173a40] bg-[radial-gradient(circle_at_50%_46%,rgba(22,105,111,.24),transparent_34%),linear-gradient(135deg,#071218,#03070b)] p-4 lg:border-b-0 lg:border-r"
          data-testid="brain-model-canvas"
          aria-label="Patient specific MRI segmentation surface. Drag to rotate."
        >
          <div className="absolute left-5 top-4 z-10 flex items-center gap-2 mono text-[9px] uppercase tracking-[.14em] text-[#6d9da0]"><MousePointer2 size={12} /> drag to orbit · scroll to zoom</div>
          <div className="brain-orbit-grid" />
          <div className="brain-orbit relative mx-auto mt-6 h-[286px] max-w-[520px]">
            <div ref={modelCanvasRef} className="absolute inset-0 z-10" aria-label="Interactive patient registered MRI surface and tumor overlay" />
            <div className="pointer-events-none absolute left-5 top-4 z-20 rounded border border-[#245d60] bg-[#061116]/80 px-2 py-1 mono text-[9px] uppercase tracking-wider text-[#8ee9df]">Patient MRI grid · registered overlay</div>
          </div>
          <div className="absolute bottom-4 left-5 right-5 flex items-center justify-between mono text-[9px] uppercase tracking-[.14em] text-[#5e898d]"><span>1 mm RAS patient space / shared voxel frame</span><span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-[#ff315c]" /> enhancing tumor · green edema</span></div>
        </div>
        <div className="grid content-start gap-3 bg-[#071016] p-5">
          <div className="section-label text-[#5cded6]">Model telemetry</div>
          <MetricChip label="Total lesion" value={`${result.followup_volumes.total_lesion_volume_cm3.toFixed(2)} cm³`} accent="teal" />
          <MetricChip label={researchOnly ? 'Mask volume rate' : 'Growth velocity'} value={`${result.volumetric_metrics.growth_velocity_cm3_per_month >= 0 ? '+' : ''}${result.volumetric_metrics.growth_velocity_cm3_per_month.toFixed(2)} cm³ / mo`} accent="pink" />
          <MetricChip label="Centroid shift" value={`${result.expansion_vector.centroid_displacement_mm.toFixed(1)} mm`} accent="amber" />
          <div className="mt-1 flex items-center gap-2 border-t border-[#173a40] pt-3 text-[10px] text-[#6d9da0]"><Zap size={12} className={isProgression ? 'text-[#ffb454]' : 'text-[#54e3bb]'} /> {isProgression ? 'expansion signal detected' : 'stable treatment effect signature'}</div>
        </div>
      </div>
    </section>
  );
}

function MetricChip({ label, value, accent }: { label: string; value: string; accent: 'teal' | 'pink' | 'amber' }) {
  const colors = { teal: 'text-[#8ee9df] border-[#245d60]', pink: 'text-[#ff89df] border-[#713562]', amber: 'text-[#ffca7a] border-[#6f4d27]' };
  return <div className={`rounded-lg border bg-[#0a151b] p-3 ${colors[accent]}`}><div className="text-[10px] text-[#6d9da0]">{label}</div><div className="mt-1 mono text-[13px]">{value}</div></div>;
}

function TelemetryTerminal({ logs, totalLatency, timestamp, maxLatency }: { logs: AnalysisResult['telemetry_logs']; totalLatency: number; timestamp: string; maxLatency: number }) {
  return (
    <div className="relative overflow-hidden bg-[#020507] px-4 py-4 sm:px-5" data-testid="telemetry-terminal">
      <div className="terminal-scanline pointer-events-none absolute inset-x-0 top-0 h-24" />
      <div className="relative mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-[#173a40] pb-3 text-[10px]">
        <div className="flex items-center gap-2 text-[#8ee9df]"><span className="flex gap-1.5"><span className="h-2 w-2 rounded-full bg-[#ff5a73]" /><span className="h-2 w-2 rounded-full bg-[#ffcb62]" /><span className="h-2 w-2 rounded-full bg-[#4de6ae]" /></span><span className="ml-1 mono">neurosight@triton:~/pipeline</span></div>
        <div className="flex items-center gap-3 mono text-[#5d898d]"><span>trace_id: NS-{String(totalLatency).replace('.', '')}</span><span className="text-[#4de6ae]">● streaming</span></div>
      </div>
      <div className="relative mb-4 grid gap-2 rounded-md border border-[#14343a] bg-[#061116] p-3 font-mono text-[10px] leading-5 text-[#77a9aa] sm:grid-cols-[1fr_auto]">
        <div><span className="text-[#4de6ae]">root@neurosight</span><span className="text-[#577f82]">:</span><span className="text-[#d75fc2]">~/inference</span><span className="text-[#aac7c5]">$ </span><span className="text-[#eaffff]">./run_progression_pipeline --mode=multimodal --trace</span><span className="terminal-cursor" /></div>
        <div className="flex gap-3 text-[#5d898d]"><span>GPU 0 / 46%</span><span>LAT {totalLatency}ms</span></div>
      </div>
      <div className="relative space-y-2 font-mono text-[10px] leading-5">
        {logs.map((log, index) => (
          <div className="terminal-row group rounded-md border border-transparent px-2 py-1.5 transition-colors hover:border-[#1c5055] hover:bg-[#07181d]" style={{ animationDelay: `${index * 80}ms` }} key={`${log.step}-${index}`} data-testid={`row-telemetry-${index}`}>
            <div className="grid gap-1 sm:grid-cols-[22px_142px_minmax(0,1fr)_72px] sm:items-center">
              <span className="text-[#3c6c70]">{String(index + 1).padStart(2, '0')}</span>
              <span className="truncate text-[#b6e9e4]">{log.step}</span>
              <span className="text-[#71999b]"><span className="mr-2 text-[#4de6ae]">OK</span>{log.message}</span>
              <span className="text-left text-[#e7b45c] sm:text-right">{log.latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="mt-1 flex items-center gap-3 pl-6 sm:pl-[164px]"><div className="h-1 flex-1 overflow-hidden rounded-full bg-[#10252a]"><div className="h-full rounded-full bg-gradient-to-r from-[#25d4c8] via-[#8c6cf2] to-[#ef5fca]" style={{ width: `${Math.max(8, (log.latency_ms / maxLatency) * 100)}%` }} /></div><span className="text-[9px] text-[#436b70]">{timestamp}</span></div>
          </div>
        ))}
      </div>
      <div className="relative mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-[#173a40] pt-3 font-mono text-[9px] text-[#4d777b]"><span>✓ all stages completed · output signed locally</span><span className="text-[#4de6ae]">exit 0 / {totalLatency}ms</span></div>
    </div>
  );
}

function Preview({ image, label }: { image: string; label: string }) {
  const src = image.startsWith('data:') ? image : `data:image/png;base64,${image}`;
  return <div className="relative aspect-square overflow-hidden rounded-md bg-[#132b31]"><img src={src} alt={`${label} MRI slice`} className="h-full w-full object-cover opacity-90 grayscale contrast-125" data-testid={`img-${label.toLowerCase().replace(/\s/g, '-')}`} /><div className="absolute left-2 top-2 rounded bg-[#07161b]/75 px-2 py-1 mono text-[9px] text-[#d5e8e4]">{label}</div><div className="absolute bottom-2 right-2 rounded bg-[#07161b]/75 px-2 py-1 mono text-[9px] text-[#9dbbb8]">T1 + C</div></div>;
}

function InfoRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="flex items-start justify-between gap-4 py-3"><span className="text-[11px] text-muted-foreground">{label}</span><span className={`text-right text-[12px] font-medium ${mono ? 'mono' : ''}`} data-testid={`text-metadata-${label.toLowerCase().replace(/\s/g, '-')}`}>{value}</span></div>;
}

function VolumeRow({ name, baseline, followup, total }: { name: string; baseline: number; followup: number; total: number }) {
  const pct = baseline ? ((followup - baseline) / baseline) * 100 : 0;
  return <div><div className="mb-1.5 flex items-center justify-between text-[11px]"><span className="font-semibold">{name}</span><span className={`mono ${pct >= 0 ? 'text-[#ad6b20]' : 'text-[#19705a]'}`}>{pct >= 0 ? '+' : ''}{pct.toFixed(1)}%</span></div><div className="grid grid-cols-[45px_1fr_45px] items-center gap-2"><span className="mono text-[10px] text-muted-foreground">{baseline.toFixed(1)}</span><div className="space-y-1"><div className="h-1.5 rounded-full bg-secondary"><div className="h-full rounded-full bg-[#8eb8b7]" style={{ width: `${Math.min(100, (baseline / total) * 100)}%` }} /></div><div className="h-1.5 rounded-full bg-secondary"><div className={`h-full rounded-full ${pct >= 0 ? 'bg-[#c7954b]' : 'bg-[#2b9b84]'}`} style={{ width: `${Math.min(100, (followup / total) * 100)}%` }} /></div></div><span className="mono text-right text-[10px] font-semibold">{followup.toFixed(1)}</span></div></div>;
}

function Signal({ label, value }: { label: string; value: string }) {
  return <div className="rounded-md bg-secondary/60 p-3"><div className="text-[10px] leading-4 text-muted-foreground">{label}</div><div className="mt-1 mono text-[15px] font-medium text-primary">{value}</div></div>;
}

function Router() {
  return (
    // Keep a shared shell (sidebar, navbar) outside the boundary so it
    // survives a page crash.
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/register" component={Register} />
        <Route path="/">
          <ProtectedRoute>
            <Home />
          </ProtectedRoute>
        </Route>
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
