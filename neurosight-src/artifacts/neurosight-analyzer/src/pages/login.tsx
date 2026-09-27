import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { ScanLine, Mail, Lock, ArrowRight, LoaderCircle, AlertCircle, ShieldCheck, CheckCircle2, Laptop } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import './login.css';

export default function Login() {
  const [email, setEmail] = useState('clinician@hospital.com');
  const [password, setPassword] = useState('clinician123');
  const [rememberDevice, setRememberDevice] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [savedDeviceDetected, setSavedDeviceDetected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  // Auto-check for saved device on mount
  useEffect(() => {
    let deviceId = localStorage.getItem('savedDeviceId');
    if (!deviceId) {
      deviceId = 'dev_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('savedDeviceId', deviceId);
    }

    // Check if this device is registered in the backend log file
    fetch(`/api/auth/saved-device?device_id=${encodeURIComponent(deviceId)}`)
      .then(res => res.json())
      .then(data => {
        if (data && data.saved) {
          setSavedDeviceDetected(true);
          setStatusMessage(`Recognized saved device in device log (${data.user.email}). Auto-entering workspace...`);
          localStorage.setItem('isAuthenticated', 'true');
          localStorage.setItem('userEmail', data.user.email);
          localStorage.setItem('userData', JSON.stringify(data.user));
          setTimeout(() => {
            window.location.href = '/';
          }, 800);
        }
      })
      .catch(err => {
        console.warn('Saved device check:', err);
      });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    let deviceId = localStorage.getItem('savedDeviceId');
    if (!deviceId) {
      deviceId = 'dev_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('savedDeviceId', deviceId);
    }

    try {
      // Call the API endpoint
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password,
          deviceId: deviceId,
          rememberDevice: rememberDevice,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Login failed');
      }

      // Set authentication state
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('userEmail', email);
      localStorage.setItem('userData', JSON.stringify(data.user));
      localStorage.setItem('savedDeviceId', deviceId);

      // Redirect to home
      window.location.href = '/';
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
      setIsLoading(false);
    }
  };

  const handleQuickBypass = () => {
    let deviceId = localStorage.getItem('savedDeviceId') || 'dev_workstation';
    localStorage.setItem('savedDeviceId', deviceId);
    localStorage.setItem('isAuthenticated', 'true');
    localStorage.setItem('userEmail', 'clinician@hospital.com');
    localStorage.setItem('userData', JSON.stringify({
      name: 'Dr. Dhyey (Lead Neuro-Oncologist)',
      email: 'clinician@hospital.com',
      role: 'Consulting Neuro-Oncologist'
    }));

    // Touch server log to register
    fetch('/api/auth/saved-device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device_id: deviceId, email: 'clinician@hospital.com' })
    }).finally(() => {
      window.location.href = '/';
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-7xl grid lg:grid-cols-2 gap-12 items-center">
        {/* Left side - 3D Brain Model */}
        <div className="brain-model-container order-2 lg:order-1">
          <div className="brain-glow"></div>
          <div className="sketchfab-embed-wrapper">
            <iframe
              title="Brain"
              frameBorder="0"
              allowFullScreen
              allow="autoplay; fullscreen; xr-spatial-tracking"
              src="https://sketchfab.com/models/3022543bb7a54b88b892ff4907e50929/embed?autostart=1&ui_controls=0&ui_infos=0&ui_watermark=0"
            ></iframe>
          </div>
          <div className="text-center mt-4">
            <p className="text-xs text-muted-foreground font-medium">
              Interactive 3D Brain Model
            </p>
            <p className="text-[10px] text-muted-foreground">
              Drag to rotate • Scroll to zoom • Interactive Cortical Surface
            </p>
          </div>
        </div>

        {/* Right side - Login Form */}
        <div className="w-full max-w-md order-1 lg:order-2">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-xl bg-primary text-primary-foreground mb-4 shadow-lg">
              <ScanLine size={32} strokeWidth={2.4} />
            </div>
            <h1 className="font-['Space_Grotesk'] text-3xl font-bold tracking-[-0.045em] mb-2">
              NeuroSight
            </h1>
            <p className="text-sm text-muted-foreground">
              Sign in to access the MRI progression analysis workstation
            </p>
          </div>

          {savedDeviceDetected && (
            <div className="mb-4 flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary animate-pulse">
              <CheckCircle2 size={18} className="shrink-0 text-primary" />
              <span>{statusMessage}</span>
            </div>
          )}

          <Card className="border-border shadow-[var(--shadow-soft)]">
            <CardHeader className="space-y-1">
              <CardTitle className="font-['Space_Grotesk'] text-xl font-semibold tracking-[-0.025em]">
                Welcome back
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Enter your credentials or enter via saved device session
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="clinician@hospital.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10"
                      disabled={isLoading}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password">Password</Label>
                    <Link
                      href="/forgot-password"
                      className="text-xs text-primary hover:underline"
                    >
                      Forgot password?
                    </Link>
                  </div>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="password"
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10"
                      disabled={isLoading}
                    />
                  </div>
                </div>

                {/* Remember device checkbox */}
                <div className="flex items-center justify-between rounded-md border border-border bg-secondary/30 p-2.5">
                  <label className="flex items-center gap-2 text-xs text-foreground cursor-pointer">
                    <input
                      type="checkbox"
                      checked={rememberDevice}
                      onChange={(e) => setRememberDevice(e.target.checked)}
                      className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="flex items-center gap-1.5 font-medium">
                      <Laptop size={14} className="text-primary" />
                      Remember this device in log file
                    </span>
                  </label>
                  <span className="text-[10px] text-muted-foreground">Skip login later</span>
                </div>

                {error && (
                  <div className="flex items-center gap-2 rounded-md border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                    <AlertCircle size={16} className="shrink-0" />
                    <span>{error}</span>
                  </div>
                )}

                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <LoaderCircle size={16} className="mr-2 animate-spin" />
                      Authenticating device...
                    </>
                  ) : (
                    <>
                      Sign in & Save Device
                      <ArrowRight size={16} className="ml-2" />
                    </>
                  )}
                </Button>

                <div className="pt-2">
                  <button
                    type="button"
                    onClick={handleQuickBypass}
                    className="w-full rounded-md border border-primary/40 bg-primary/10 py-2.5 text-xs font-semibold text-primary hover:bg-primary/20 transition-colors flex items-center justify-center gap-2"
                  >
                    <ShieldCheck size={15} />
                    One-Click Saved Device Access (No Password)
                  </button>
                </div>
              </form>

              <div className="mt-6 text-center text-sm text-muted-foreground">
                Don't have an account?{' '}
                <Link href="/register" className="text-primary hover:underline font-medium">
                  Request access
                </Link>
              </div>
            </CardContent>
          </Card>

          <div className="mt-6 text-center text-xs text-muted-foreground space-y-1">
            <p className="flex items-center justify-center gap-1.5">
              <ShieldCheck size={13} className="text-primary" />
              Saved Device Authorization File: <span className="mono text-[11px] text-foreground">logs/saved_devices.log</span>
            </p>
            <p>Protected clinical workspace • HIPAA compliant audit trail</p>
          </div>
        </div>
      </div>
    </div>
  );
}