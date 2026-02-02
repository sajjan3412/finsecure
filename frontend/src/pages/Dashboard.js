import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, BrainCircuit, Download, Activity, Network, Settings2, ShieldCheck, Eye, EyeOff, Bell, X } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('sentinel_api_key') || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [trainingRounds, setTrainingRounds] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [companyInfo, setCompanyInfo] = useState(null);

  useEffect(() => {
    if (apiKey) {
      verifyApiKey();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const verifyApiKey = async () => {
    try {
      const response = await axios.get(`${API}/auth/verify`, {
        headers: { 'X-API-Key': apiKey }
      });
      setAuthenticated(true);
      setCompanyInfo(response.data);
      loadDashboardData();
      toast.success('Connected successfully!');
    } catch (error) {
      toast.error('Invalid API key');
      setAuthenticated(false);
    }
  };

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [statsRes, roundsRes, companiesRes] = await Promise.all([
        axios.get(`${API}/analytics/dashboard`),
        axios.get(`${API}/analytics/rounds`),
        axios.get(`${API}/companies`)
      ]);
      
      setDashboardStats(statsRes.data);
      setTrainingRounds(roundsRes.data);
      setCompanies(companiesRes.data);
    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = () => {
    localStorage.setItem('sentinel_api_key', apiKey);
    verifyApiKey();
  };

  const downloadClientScript = async () => {
    try {
      const response = await axios.get(`${API}/client/script`, {
        headers: { 'X-API-Key': apiKey }
      });
      
      const blob = new Blob([response.data.content], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = response.data.filename;
      a.click();
      
      toast.success('Client script downloaded!');
    } catch (error) {
      toast.error('Failed to download script');
    }
  };

  // Login Screen
  if (!authenticated) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center p-4" data-testid="dashboard-login">
        <motion.div
          className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <ShieldCheck className="w-10 h-10 text-indigo-500" />
            <h1 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>Sentinel Dashboard</h1>
          </div>
          <p className="text-[#A1A1AA] mb-6" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
            Enter your API key to access the federated learning dashboard.
          </p>
          <div className="space-y-4">
            <div>
              <label className="block text-sm mb-2 text-[#A1A1AA]">API Key</label>
              <Input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-[#050505] border-white/10 focus:border-indigo-500 rounded-md h-10"
                placeholder="sf_xxxxxxxxxxxxx"
                data-testid="api-key-input"
              />
            </div>
            <Button
              onClick={handleConnect}
              className="w-full h-10 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white"
              data-testid="connect-btn"
            >
              Connect
            </Button>
          </div>
        </motion.div>
      </div>
    );
  }

  // Dashboard Screen
  return (
    <div className="min-h-screen bg-[#050505] text-[#EDEDED]" data-testid="dashboard-main">
      {/* Header */}
      <div className="border-b border-white/5 bg-[#0A0A0A]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-indigo-500" />
            <div>
              <h1 className="text-xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>Sentinel Federated</h1>
              {companyInfo && (
                <p className="text-xs text-[#A1A1AA]">{companyInfo.name}</p>
              )}
            </div>
          </div>
          <Button
            onClick={downloadClientScript}
            className="h-9 px-4 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm"
            data-testid="download-client-btn"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Client
          </Button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        {dashboardStats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card className="rounded-lg bg-[#121212] border border-white/5 p-4" data-testid="stat-companies">
              <div className="flex items-center justify-between mb-2">
                <Network className="w-5 h-5 text-indigo-500" />
                <span className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>
                  {dashboardStats.active_companies}
                </span>
              </div>
              <p className="text-sm text-[#A1A1AA]">Active Companies</p>
            </Card>

            <Card className="rounded-lg bg-[#121212] border border-white/5 p-4" data-testid="stat-rounds">
              <div className="flex items-center justify-between mb-2">
                <Activity className="w-5 h-5 text-emerald-500" />
                <span className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>
                  {dashboardStats.total_rounds}
                </span>
              </div>
              <p className="text-sm text-[#A1A1AA]">Training Rounds</p>
            </Card>

            <Card className="rounded-lg bg-[#121212] border border-white/5 p-4" data-testid="stat-accuracy">
              <div className="flex items-center justify-between mb-2">
                <BrainCircuit className="w-5 h-5 text-rose-500" />
                <span className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>
                  {(dashboardStats.current_accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-sm text-[#A1A1AA]">Model Accuracy</p>
            </Card>

            <Card className="rounded-lg bg-[#121212] border border-white/5 p-4" data-testid="stat-updates">
              <div className="flex items-center justify-between mb-2">
                <ShieldCheck className="w-5 h-5 text-indigo-500" />
                <span className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>
                  {dashboardStats.total_updates}
                </span>
              </div>
              <p className="text-sm text-[#A1A1AA]">Gradient Updates</p>
            </Card>
          </div>
        )}

        {/* Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="bg-[#0A0A0A] border border-white/5 mb-6">
            <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
            <TabsTrigger value="companies" data-testid="tab-companies">Companies</TabsTrigger>
            <TabsTrigger value="setup" data-testid="tab-setup">Client Setup</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" data-testid="overview-tab-content">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6" style={{ fontFamily: 'Chivo, sans-serif' }}>Model Performance Over Time</h3>
              {trainingRounds.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingRounds}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis 
                      dataKey="round_number" 
                      stroke="#A1A1AA"
                      label={{ value: 'Training Round', position: 'insideBottom', offset: -5, fill: '#A1A1AA' }}
                    />
                    <YAxis 
                      stroke="#A1A1AA"
                      label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', fill: '#A1A1AA' }}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#121212', border: '1px solid #333' }}
                      labelStyle={{ color: '#EDEDED' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="avg_accuracy" 
                      stroke="#6366F1" 
                      strokeWidth={2}
                      dot={{ fill: '#6366F1' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-20 text-[#A1A1AA]">
                  <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No training rounds yet. Upload your first gradient updates to see progress.</p>
                </div>
              )}
            </Card>
          </TabsContent>

          <TabsContent value="companies" data-testid="companies-tab-content">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6" style={{ fontFamily: 'Chivo, sans-serif' }}>Registered Companies</h3>
              <div className="space-y-3">
                {companies.map((company, idx) => (
                  <div 
                    key={idx}
                    className="bg-[#121212] border border-white/5 rounded-lg p-4 flex justify-between items-center"
                    data-testid={`company-item-${idx}`}
                  >
                    <div>
                      <p className="font-medium" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>{company.name}</p>
                      <p className="text-sm text-[#A1A1AA]">{company.email}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-3 py-1 rounded-full ${
                        company.status === 'active' ? 'bg-emerald-500/20 text-emerald-500' : 'bg-gray-500/20 text-gray-500'
                      }`}>
                        {company.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="setup" data-testid="setup-tab-content">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6" style={{ fontFamily: 'Chivo, sans-serif' }}>Client Setup Guide</h3>
              
              <div className="space-y-6">
                <div>
                  <h4 className="font-bold mb-3 text-emerald-500">Step 1: Download Client Script</h4>
                  <Button
                    onClick={downloadClientScript}
                    className="h-10 px-6 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white"
                    data-testid="download-script-btn"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Python Script
                  </Button>
                </div>

                <div>
                  <h4 className="font-bold mb-3 text-emerald-500">Step 2: Install Dependencies</h4>
                  <div className="bg-[#050505] border border-white/10 rounded-lg p-4">
                    <code className="text-sm" style={{ fontFamily: 'JetBrains Mono, monospace', color: '#10B981' }}>
                      pip install tensorflow numpy requests
                    </code>
                  </div>
                </div>

                <div>
                  <h4 className="font-bold mb-3 text-emerald-500">Step 3: Run the Script</h4>
                  <div className="bg-[#050505] border border-white/10 rounded-lg p-4">
                    <code className="text-sm" style={{ fontFamily: 'JetBrains Mono, monospace', color: '#10B981' }}>
                      python sentinel_client_xxxxx.py
                    </code>
                  </div>
                </div>

                <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-4">
                  <p className="text-sm text-rose-400">
                    <strong>Important:</strong> The client script trains locally on your transaction data and only shares gradient updates. Your raw data never leaves your infrastructure.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;
