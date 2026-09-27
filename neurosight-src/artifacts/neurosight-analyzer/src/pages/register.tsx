import { useState } from 'react';
import { Link } from 'wouter';
import { ScanLine, Mail, Lock, User, ArrowRight, LoaderCircle, AlertCircle, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import './login.css';

export default function Register() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!name.trim() || !email.trim() || !password.trim() || !confirmPassword.trim()) {
      setError('Please fill in all fields');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setIsLoading(true);

    try {
      // Call the real API endpoint
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim(),
          password: password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Registration failed');
      }

      setSuccess(true);
      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div className="w-full max-w-4xl grid lg:grid-cols-2 gap-12 items-center">
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
              <p className="text-xs text-muted-foreground">
                Interactive 3D Brain Model
              </p>
              <p className="text-[10px] text-muted-foreground">
                Drag to rotate • Scroll to zoom
              </p>
            </div>
          </div>

          {/* Right side - Success Message */}
          <div className="w-full max-w-md order-1 lg:order-2">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-xl bg-green-500 text-white mb-4 shadow-lg">
                <Check size={32} strokeWidth={2.4} />
              </div>
              <h1 className="font-['Space_Grotesk'] text-3xl font-bold tracking-[-0.045em] mb-2">
                Registration Submitted
              </h1>
              <p className="text-sm text-muted-foreground">
                Your access request has been received
              </p>
            </div>

            <Card className="border-border shadow-[var(--shadow-soft)]">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="flex items-start gap-3 p-4 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800">
                    <Check className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5 shrink-0" />
                    <div>
                      <p className="font-medium text-green-900 dark:text-green-100">
                        Request Received
                      </p>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        We'll review your application and send approval instructions to <strong>{email}</strong>
                      </p>
                    </div>
                  </div>

                  <div className="text-sm text-muted-foreground space-y-2">
                    <p>• Medical credential verification required</p>
                    <p>• Typical approval time: 1-2 business days</p>
                    <p>• You'll receive an email when your account is ready</p>
                  </div>

                  <Button
                    className="w-full"
                    onClick={() => (window.location.href = '/login')}
                  >
                    Return to Login
                    <ArrowRight size={16} className="ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            <div className="mt-6 text-center text-xs text-muted-foreground">
              <p>Protected clinical workspace • HIPAA compliant</p>
              <p className="mt-1">For technical support, contact your IT department</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
            <p className="text-xs text-muted-foreground">
              Interactive 3D Brain Model
            </p>
            <p className="text-[10px] text-muted-foreground">
              Drag to rotate • Scroll to zoom
            </p>
          </div>
        </div>

        {/* Right side - Registration Form */}
        <div className="w-full max-w-md order-1 lg:order-2">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-xl bg-primary text-primary-foreground mb-4 shadow-lg">
              <ScanLine size={32} strokeWidth={2.4} />
            </div>
            <h1 className="font-['Space_Grotesk'] text-3xl font-bold tracking-[-0.045em] mb-2">
              NeuroSight
            </h1>
            <p className="text-sm text-muted-foreground">
              Request access to the MRI analysis workstation
            </p>
          </div>

          <Card className="border-border shadow-[var(--shadow-soft)]">
            <CardHeader className="space-y-1">
              <CardTitle className="font-['Space_Grotesk'] text-xl font-semibold tracking-[-0.025em]">
                Create Account
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Enter your information to request clinical workspace access
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="name"
                      type="text"
                      placeholder="Dr. Jane Smith"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="pl-10"
                      disabled={isLoading}
                    />
                  </div>
                </div>

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
                  <Label htmlFor="password">Password</Label>
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
                  <p className="text-[10px] text-muted-foreground">
                    Must be at least 8 characters
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="confirmPassword"
                      type="password"
                      placeholder="••••••••"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="pl-10"
                      disabled={isLoading}
                    />
                  </div>
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
                      Submitting Request...
                    </>
                  ) : (
                    <>
                      Request Access
                      <ArrowRight size={16} className="ml-2" />
                    </>
                  )}
                </Button>
              </form>

              <div className="mt-6 text-center text-sm text-muted-foreground">
                Already have an account?{' '}
                <Link href="/login" className="text-primary hover:underline font-medium">
                  Sign in
                </Link>
              </div>
            </CardContent>
          </Card>

          <div className="mt-6 text-center text-xs text-muted-foreground">
            <p>Protected clinical workspace • HIPAA compliant</p>
            <p className="mt-1">For technical support, contact your IT department</p>
          </div>
        </div>
      </div>
    </div>
  );
}