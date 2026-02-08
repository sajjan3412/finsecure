import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, BrainCircuit, Download, Activity, Network, Settings2, ShieldCheck, Eye, EyeOff, Bell, X, Terminal, Copy, Check } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { toast } from 'sonner';
import axios from 'axios';

// Ensure no double slashes in URL
const BACKEND_URL = (process.env.REACT_APP_BACKEND_URL || 'https://finsecure-ochi.onrender.com').replace(/\/$/, '');
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('finsecure_api_key') || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  
  // Dashboard Data State
  const [dashboardStats, setDashboardStats] = useState(null);
  const [trainingRounds, setTrainingRounds] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [companyInfo, setCompanyInfo] = useState(null);
  const [copied, setCopied] = useState(false);
  
  // Notification State
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);

  // Initial Check
  useEffect(() => {
    if (apiKey) {
      verifyApiKey();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Polling Logic
  useEffect(() => {
    if (authenticated) {
      loadDashboardData(); 
      const interval = setInterval(() => {
        loadNotifications();
        loadDashboardData(true);
      }, 5000); 
      return () => clearInterval(interval);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authenticated]);

  const verifyApiKey = async () => {
    try {
      const response = await axios.get(`${API}/auth/verify`, {
        headers: { 'X-API-Key': apiKey }
      });
      setAuthenticated(true);
      setCompanyInfo(response.data);
      toast.success('Connected successfully!');
    } catch (error) {
      console.error(error);
      if (!authenticated) toast.error('Invalid API key');
      setAuthenticated(false);
    }
  };

  const loadDashboardData = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const [statsRes, roundsRes, companiesRes] = await Promise.all([
        axios.get(`${API}/analytics/dashboard`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/analytics/rounds`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/companies`, { headers: { 'X-API-Key': apiKey } })
      ]);
      
      setDashboardStats(statsRes.data);
      setTrainingRounds(roundsRes.data);
      setCompanies(companiesRes.data);
    } catch (error) {
      console.error('Failed to load dashboard data', error);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const loadNotifications = async () => {
    try {
      const [notifRes, countRes] = await Promise.all([
        axios.get(`${API}/notifications`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/notifications/unread/count`, { headers: { 'X-API-Key': apiKey } })
      ]);
      setNotifications(notifRes.data);
      setUnreadCount(countRes.data.count || 0);
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
  };

  const markAsRead = async (notificationId) => {
    setNotifications(prev => prev.map(n => n.notification_id === notificationId ? { ...n, read: true } : n));
    setUnreadCount(Math.max(0, unreadCount - 1));
  };

  const handleConnect = () => {
    localStorage.setItem('finsecure_api_key', apiKey);
    verifyApiKey();
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success('Command copied to clipboard!');
    setTimeout(() => setCopied(false), 2000);
  };

  // --- RENDER: LOGIN ---
  if (!authenticated) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center p-4">
        <motion.div
          className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <ShieldCheck className="w-10 h-10 text-indigo-500" />
            <h1 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>FinSecure Dashboard</h1>
          </div>
          <p className="text-[#A1A1AA] mb-6">Enter your API key to access the federated learning network.</p>
          <div className="space-y-4">
            <div>
              <label className="block text-sm mb-2 text-[#A1A1AA]">API Key</label>
              <Input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-[#050505] border-white/10 focus:border-indigo-500"
                placeholder="fs_xxxxxxxxxxxxx"
              />
            </div>
            <Button onClick={handleConnect} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white">
              Connect Securely
            </Button>
          </div>
        </motion.div>
      </div>
    );
  }

  // --- RENDER: DASHBOARD ---
  return (
    <div className="min-h-screen bg-[#050505] text-[#EDEDED]">
      {/* Header */}
      <div className="border-b border-white/5 bg-[#0A0A0A] sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-indigo-500" />
            <div>
              <h1 className="text-xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>FinSecure</h1>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                <p className="text-xs text-[#A1A1AA]">Network Online • {companyInfo?.name}</p>
              </div>
            </div>
          </div>
          
          <div className="relative">
            <Button
              onClick={() => setShowNotifications(!showNotifications)}
              className="h-9 w-9 p-0 rounded-md bg-white/5 hover:bg-white/10 border border-white/10 relative"
            >
              <Bell className="w-4 h-4" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 bg-rose-500 text-white text-[10px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                  {unreadCount}
                </span>
              )}
            </Button>

            <AnimatePresence>
              {showNotifications && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute right-0 top-12 w-96 bg-[#0A0A0A] border border-white/10 rounded-lg shadow-2xl z-50"
                >
                  <div className="p-3 border-b border-white/10 flex justify-between items-center">
                    <h3 className="font-bold text-sm">Notifications</h3>
                    <Button onClick={() => setShowNotifications(false)} variant="ghost" size="icon" className="h-6 w-6">
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="max-h-80 overflow-y-auto">
                    {notifications.length === 0 ? (
                      <div className="p-6 text-center text-sm text-[#A1A1AA]">No new alerts</div>
                    ) : (
                      notifications.map((notif) => (
                        <div
                          key={notif.notification_id}
                          className={`p-3 border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors ${!notif.read ? 'bg-indigo-500/5' : ''}`}
                          onClick={() => markAsRead(notif.notification_id)}
                        >
                          <div className="flex justify-between items-start mb-1">
                            <span className="font-semibold text-sm">{notif.title}</span>
                            <span className="text-[10px] text-[#A1A1AA]">{new Date(notif.created_at).toLocaleTimeString()}</span>
                          </div>
                          <p className="text-xs text-[#A1A1AA] line-clamp-2">{notif.message}</p>
                        </div>
                      ))
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Statistics Cards */}
        {dashboardStats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card className="bg-[#121212] border-white/5 p-4 hover:border-indigo-500/50 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="text-sm text-[#A1A1AA]">Active Nodes</p>
                  <h3 className="text-2xl font-bold font-mono">{dashboardStats.active_companies}</h3>
                </div>
                <Network className="w-5 h-5 text-indigo-500" />
              </div>
            </Card>
            
            <Card className="bg-[#121212] border-white/5 p-4 hover:border-emerald-500/50 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="text-sm text-[#A1A1AA]">Global Rounds</p>
                  <h3 className="text-2xl font-bold font-mono">{dashboardStats.total_rounds}</h3>
                </div>
                <Activity className="w-5 h-5 text-emerald-500" />
              </div>
            </Card>

            <Card className="bg-[#121212] border-white/5 p-4 hover:border-rose-500/50 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="text-sm text-[#A1A1AA]">Network Accuracy</p>
                  <h3 className="text-2xl font-bold font-mono">{(dashboardStats.current_accuracy * 100).toFixed(1)}%</h3>
                </div>
                <BrainCircuit className="w-5 h-5 text-rose-500" />
              </div>
            </Card>

            <Card className="bg-[#121212] border-white/5 p-4 hover:border-amber-500/50 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="text-sm text-[#A1A1AA]">Total Updates</p>
                  <h3 className="text-2xl font-bold font-mono">{dashboardStats.total_updates}</h3>
                </div>
                <ShieldCheck className="w-5 h-5 text-amber-500" />
              </div>
            </Card>
          </div>
        )}

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="bg-[#0A0A0A] border border-white/5 mb-6">
            <TabsTrigger value="overview">Live Analytics</TabsTrigger>
            <TabsTrigger value="companies">Network Nodes</TabsTrigger>
            <TabsTrigger value="setup">Client Setup</TabsTrigger>
          </TabsList>

          {/* TAB 1: OVERVIEW (GRAPH) */}
          <TabsContent value="overview">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold">Global Model Accuracy</h3>
                <Badge variant="outline" className="border-indigo-500 text-indigo-400">Live Feed</Badge>
              </div>
              
              {trainingRounds.length > 0 ? (
                <div className="h-[350px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart 
                      data={trainingRounds.slice(-20)}
                      margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                      <XAxis 
                        dataKey="round" 
                        stroke="#525252" 
                        tick={{fill: '#A1A1AA', fontSize: 12}}
                        tickLine={false}
                        axisLine={false}
                        label={{ value: 'Round', position: 'insideBottom', offset: -5, fill: '#525252' }}
                      />
                      <YAxis 
                        stroke="#525252" 
                        tick={{fill: '#A1A1AA', fontSize: 12}}
                        tickLine={false}
                        axisLine={false}
                        domain={[0, 1]}
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#171717', border: '1px solid #262626', borderRadius: '8px' }}
                        itemStyle={{ color: '#E5E5E5' }}
                        labelStyle={{ color: '#A1A1AA', marginBottom: '4px' }}
                        formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        stroke="#6366F1" 
                        strokeWidth={3} 
                        dot={{ r: 4, fill: '#1E1B4B', strokeWidth: 2, stroke: '#6366F1' }} 
                        activeDot={{ r: 6, fill: '#818CF8' }}
                        animationDuration={1000}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[300px] text-[#525252]">
                  <Activity className="w-12 h-12 mb-4 opacity-20" />
                  <p>Waiting for first training round...</p>
                </div>
              )}
            </Card>
          </TabsContent>

          {/* TAB 2: COMPANIES */}
          <TabsContent value="companies">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6">Connected Banks</h3>
              <div className="grid gap-3">
                {companies.map((company, idx) => (
                  <div key={idx} className="bg-[#121212] border border-white/5 rounded-lg p-4 flex justify-between items-center hover:bg-[#171717] transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-500 font-bold">
                        {company.name.charAt(0)}
                      </div>
                      <div>
                        <p className="font-medium text-[#EDEDED]">{company.name}</p>
                        <p className="text-xs text-[#737373]">{company.email}</p>
                      </div>
                    </div>
                    <Badge className="bg-emerald-500/10 text-emerald-500 border-0">Active Node</Badge>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          {/* TAB 3: SETUP - NEW CLI VERSION */}
          <TabsContent value="setup">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6">Client Configuration (CLI)</h3>
              <div className="space-y-8">
                
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 font-bold text-sm">1</div>
                  <div className="flex-1">
                    <h4 className="font-bold mb-2">Install Gateway</h4>
                    <p className="text-sm text-[#A1A1AA] mb-3">Copy and run this command in your terminal to install the gateway script:</p>
                    
                    <div className="bg-[#050505] border border-white/10 rounded-lg p-4 relative group">
                      <code className="text-sm font-mono text-emerald-400 block break-all pr-10">
                        curl -H "X-API-Key: {apiKey}" -o finsecure_gateway.py {API}/client/script
                      </code>
                      <Button 
                        onClick={() => copyToClipboard(`curl -H "X-API-Key: ${apiKey}" -o finsecure_gateway.py ${API}/client/script`)}
                        className="absolute top-2 right-2 h-8 w-8 p-0 bg-white/5 hover:bg-white/10"
                        variant="ghost"
                      >
                        {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4 text-[#A1A1AA]" />}
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 font-bold text-sm">2</div>
                  <div className="flex-1">
                    <h4 className="font-bold mb-2">Install Libraries</h4>
                    <div className="bg-[#050505] border border-white/10 rounded-lg p-3 font-mono text-sm text-[#A1A1AA]">
                      pip install tensorflow requests numpy pandas
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 font-bold text-sm">3</div>
                  <div className="flex-1">
                    <h4 className="font-bold mb-2">Run the Node</h4>
                    <div className="bg-[#050505] border border-white/10 rounded-lg p-3 font-mono text-sm text-[#A1A1AA]">
                      python universal_bank_node.py
                    </div>
                  </div>
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
