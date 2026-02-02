import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, Network, BrainCircuit, Lock, Activity, ArrowRight, CheckCircle2, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LandingPage = () => {
  const navigate = useNavigate();
  const [showRegister, setShowRegister] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [loginMethod, setLoginMethod] = useState('email'); // 'email' or 'apikey'
  const [companyName, setCompanyName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginApiKey, setLoginApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [registeredData, setRegisteredData] = useState(null);

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!companyName || !email || !password) {
      toast.error('Please fill all fields');
      return;
    }

    if (password.length < 8) {
      toast.error('Password must be at least 8 characters long');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/auth/register`, {
        name: companyName,
        email: email,
        password: password
      });
      
      setRegisteredData(response.data);
      localStorage.setItem('finsecure_api_key', response.data.api_key);
      toast.success('Company registered successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    if (loginMethod === 'email') {
      if (!loginEmail || !loginPassword) {
        toast.error('Please enter email and password');
        return;
      }

      setLoading(true);
      try {
        const response = await axios.post(`${API}/auth/login`, {
          email: loginEmail,
          password: loginPassword
        });
        
        if (response.data.success) {
          localStorage.setItem('finsecure_api_key', response.data.api_key);
          toast.success(`Welcome back, ${response.data.name}!`);
          navigate('/dashboard');
        }
      } catch (error) {
        toast.error(error.response?.data?.detail || 'Invalid email or password');
      } finally {
        setLoading(false);
      }
    } else {
      // API Key login
      if (!loginApiKey) {
        toast.error('Please enter your API key');
        return;
      }

      setLoading(true);
      try {
        const response = await axios.get(`${API}/auth/verify`, {
          headers: { 'X-API-Key': loginApiKey }
        });
        
        if (response.data.valid) {
          localStorage.setItem('finsecure_api_key', loginApiKey);
          toast.success(`Welcome back, ${response.data.name}!`);
          navigate('/dashboard');
        }
      } catch (error) {
        toast.error('Invalid API key');
      } finally {
        setLoading(false);
      }
    }
  };

  const goToDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#EDEDED]">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1764336312138-14a5368a6cd3?crop=entropy&cs=srgb&fm=jpg&q=85')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center'
          }}
        />
        <div className="absolute inset-0" style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%)'
        }} />
        
        <div className="relative max-w-7xl mx-auto px-6 py-20">
          {/* Header */}
          <motion.div 
            className="flex justify-between items-center mb-20"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center gap-3">
              <ShieldCheck className="w-10 h-10 text-indigo-500" />
              <h1 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>
                FinSecure
              </h1>
            </div>
            <div className="flex gap-3">
              <Button 
                onClick={() => setShowLogin(true)}
                className="h-10 px-6 rounded-md bg-white/5 hover:bg-white/10 text-white border border-white/10"
                data-testid="login-btn"
              >
                Login
              </Button>
              <Button 
                onClick={() => setShowRegister(true)}
                className="h-10 px-6 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-all hover:shadow-[0_0_20px_rgba(99,102,241,0.4)]"
                data-testid="get-started-btn"
              >
                Get Started
              </Button>
            </div>
          </motion.div>

          {/* Hero Content */}
          <div className="grid lg:grid-cols-2 gap-12 items-center mb-32">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h2 
                className="text-4xl sm:text-5xl lg:text-6xl font-black mb-6"
                style={{ fontFamily: 'Chivo, sans-serif', letterSpacing: '-0.02em' }}
              >
                Collective Defense
                <br />
                <span className="text-indigo-500">Against Fraud</span>
              </h2>
              <p className="text-base sm:text-lg text-[#A1A1AA] leading-relaxed mb-8" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
                Join a privacy-preserving network where fintech companies collaborate to detect fraud without sharing sensitive transaction data. Powered by federated learning.
              </p>
              <div className="flex gap-4">
                <Button 
                  onClick={() => setShowRegister(true)}
                  className="h-12 px-8 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-all hover:shadow-[0_0_20px_rgba(99,102,241,0.4)]"
                  data-testid="hero-get-started-btn"
                >
                  Start Protecting <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </div>
            </motion.div>

            <motion.div
              className="relative"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="relative bg-black/40 backdrop-blur-xl border border-white/5 rounded-xl p-8">
                <img 
                  src="https://images.unsplash.com/photo-1696013910376-c56f76dd8178?crop=entropy&cs=srgb&fm=jpg&q=85"
                  alt="Security"
                  className="rounded-lg w-full"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#050505] via-transparent to-transparent rounded-xl" />
              </div>
            </motion.div>
          </div>

          {/* Features Bento Grid */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <h3 className="text-3xl font-bold mb-12 text-center" style={{ fontFamily: 'Chivo, sans-serif' }}>
              Why FinSecure?
            </h3>
            
            <div className="grid md:grid-cols-3 gap-8">
              <Card className="group relative overflow-hidden rounded-xl bg-[#0A0A0A] border border-white/5 p-8 hover:border-indigo-500/30 transition-all" data-testid="feature-privacy">
                <Lock className="w-12 h-12 text-indigo-500 mb-4" />
                <h4 className="text-xl font-bold mb-3" style={{ fontFamily: 'Chivo, sans-serif' }}>Privacy First</h4>
                <p className="text-[#A1A1AA]" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
                  Your transaction data never leaves your infrastructure. Only encrypted gradient updates are shared.
                </p>
              </Card>

              <Card className="group relative overflow-hidden rounded-xl bg-[#0A0A0A] border border-white/5 p-8 hover:border-indigo-500/30 transition-all" data-testid="feature-collaborative">
                <Network className="w-12 h-12 text-emerald-500 mb-4" />
                <h4 className="text-xl font-bold mb-3" style={{ fontFamily: 'Chivo, sans-serif' }}>Collaborative Learning</h4>
                <p className="text-[#A1A1AA]" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
                  Benefit from insights across the entire network. The more companies join, the smarter the model becomes.
                </p>
              </Card>

              <Card className="group relative overflow-hidden rounded-xl bg-[#0A0A0A] border border-white/5 p-8 hover:border-indigo-500/30 transition-all" data-testid="feature-accurate">
                <BrainCircuit className="w-12 h-12 text-rose-500 mb-4" />
                <h4 className="text-xl font-bold mb-3" style={{ fontFamily: 'Chivo, sans-serif' }}>Superior Accuracy</h4>
                <p className="text-[#A1A1AA]" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
                  Pre-trained TensorFlow models continuously improve through federated averaging, detecting fraud patterns faster.
                </p>
              </Card>
            </div>
          </motion.div>

          {/* How It Works */}
          <motion.div
            className="mt-32"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <h3 className="text-3xl font-bold mb-12 text-center" style={{ fontFamily: 'Chivo, sans-serif' }}>
              How It Works
            </h3>
            
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { step: '1', title: 'Register', desc: 'Sign up and receive your unique API key' },
                { step: '2', title: 'Download Script', desc: 'Install our lightweight Python client' },
                { step: '3', title: 'Train Locally', desc: 'Model trains on your private transaction data' },
                { step: '4', title: 'Share Gradients', desc: 'Only model updates are sent to central server' }
              ].map((item, idx) => (
                <div key={idx} className="text-center" data-testid={`how-step-${idx + 1}`}>
                  <div className="w-16 h-16 rounded-full bg-indigo-600/20 flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl font-bold text-indigo-500" style={{ fontFamily: 'JetBrains Mono, monospace' }}>{item.step}</span>
                  </div>
                  <h4 className="text-lg font-bold mb-2" style={{ fontFamily: 'Chivo, sans-serif' }}>{item.title}</h4>
                  <p className="text-sm text-[#A1A1AA]" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>{item.desc}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>

      {/* Registration Modal */}
      {showRegister && (
        <div 
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" 
          data-testid="registration-modal"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowRegister(false);
            }
          }}
        >
          <motion.div
            className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={(e) => e.stopPropagation()}
          >
            {!registeredData ? (
              <>
                <h3 className="text-2xl font-bold mb-6" style={{ fontFamily: 'Chivo, sans-serif' }}>Register Your Company</h3>
                <form onSubmit={handleRegister} className="space-y-4">
                  <div>
                    <label className="block text-sm mb-2 text-[#A1A1AA]">Company Name</label>
                    <Input
                      value={companyName}
                      onChange={(e) => setCompanyName(e.target.value)}
                      className="bg-[#050505] border-white/10 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-md h-10"
                      placeholder="Acme Fintech"
                      data-testid="company-name-input"
                    />
                  </div>
                  <div>
                    <label className="block text-sm mb-2 text-[#A1A1AA]">Email</label>
                    <Input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="bg-[#050505] border-white/10 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-md h-10"
                      placeholder="admin@acme.com"
                      data-testid="email-input"
                    />
                  </div>
                  <div className="flex gap-3 mt-6">
                    <Button
                      type="button"
                      onClick={() => setShowRegister(false)}
                      className="h-10 px-6 rounded-md bg-white/5 hover:bg-white/10 text-white border border-white/10"
                      data-testid="cancel-btn"
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      disabled={loading}
                      className="h-10 px-6 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white flex-1"
                      data-testid="register-submit-btn"
                    >
                      {loading ? 'Registering...' : 'Register'}
                    </Button>
                  </div>
                </form>
              </>
            ) : (
              <div data-testid="registration-success">
                <div className="flex items-center gap-3 mb-6">
                  <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                  <h3 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>Registration Complete!</h3>
                </div>
                <div className="bg-[#050505] border border-white/10 rounded-lg p-4 mb-4">
                  <p className="text-sm text-[#A1A1AA] mb-2">Your API Key:</p>
                  <code 
                    className="text-xs text-emerald-500 break-all block"
                    style={{ fontFamily: 'JetBrains Mono, monospace' }}
                    data-testid="api-key-display"
                  >
                    {registeredData.api_key}
                  </code>
                </div>
                <p className="text-sm text-[#A1A1AA] mb-6">Save this API key securely. You'll need it to download the client script.</p>
                <Button
                  onClick={goToDashboard}
                  className="w-full h-10 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white"
                  data-testid="go-to-dashboard-btn"
                >
                  Go to Dashboard <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </div>
            )}
          </motion.div>
        </div>
      )}

      {/* Login Modal */}
      {showLogin && (
        <div 
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" 
          data-testid="login-modal"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowLogin(false);
            }
          }}
        >
          <motion.div
            className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-2xl font-bold mb-6" style={{ fontFamily: 'Chivo, sans-serif' }}>Login to FinSecure</h3>
            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-sm mb-2 text-[#A1A1AA]">API Key</label>
                <Input
                  type="password"
                  value={loginApiKey}
                  onChange={(e) => setLoginApiKey(e.target.value)}
                  className="bg-[#050505] border-white/10 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-md h-10"
                  placeholder="fs_xxxxxxxxxxxxx"
                  data-testid="login-api-key-input"
                />
                <p className="text-xs text-[#A1A1AA] mt-2">Enter the API key you received during registration</p>
              </div>
              <div className="flex gap-3 mt-6">
                <Button
                  type="button"
                  onClick={() => setShowLogin(false)}
                  className="h-10 px-6 rounded-md bg-white/5 hover:bg-white/10 text-white border border-white/10"
                  data-testid="login-cancel-btn"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={loading}
                  className="h-10 px-6 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white flex-1"
                  data-testid="login-submit-btn"
                >
                  {loading ? 'Verifying...' : 'Login'}
                </Button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default LandingPage;
