import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, BrainCircuit, Download, Activity, Network, Settings2, ShieldCheck, Eye, EyeOff, Bell, X, Terminal, Copy, Check, History, Package, LogOut } from 'lucide-react';
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
  const [authenticated, setAuthenticated] = useState(false);
  
  // Dashboard Data State
  const [dashboardStats, setDashboardStats] = useState(null);
  const [trainingRounds, setTrainingRounds] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [companyInfo, setCompanyInfo] = useState(null);
  
  // New State for History
  const [myUpdates, setMyUpdates] = useState([]);

  // Notification State
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);

  // Copy States
  const [copiedSdk, setCopiedSdk] = useState(false);
  const [copiedNode, setCopiedNode] = useState(false);

  // --- BRANDING REMOVER (HACK FOR DEMO) ---
  useEffect(() => {
    const cleaner = setInterval(() => {
      // Find any link pointing to the editor platform and hide it
      const branding = Array.from(document.querySelectorAll('a')).find(el => el.href.includes('emergent') || el.innerText.includes('Emergent'));
      if (branding) {
        branding.style.display = 'none';
        branding.style.visibility = 'hidden';
        branding.style.opacity = '0';
        branding.style.pointerEvents = 'none';
      }
    }, 500); // Check every 0.5s
    return () => clearInterval(cleaner);
  }, []);
  // ----------------------------------------

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
    try {
      const [statsRes, roundsRes, companiesRes, historyRes] = await Promise.all([
        axios.get(`${API}/analytics/dashboard`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/analytics/rounds`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/companies`, { headers: { 'X-API-Key': apiKey } }),
        axios.get(`${API}/analytics/my-updates`, { headers: { 'X-API-Key': apiKey } })
      ]);
      
      setDashboardStats(statsRes.data);
      setTrainingRounds(roundsRes.data);
      setCompanies(companiesRes.data);
      setMyUpdates(historyRes.data);
    } catch (error) {
      console.error('Failed to load dashboard data', error);
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

  const logout = () => {
    localStorage.removeItem('finsecure_api_key');
    setApiKey('');
    setAuthenticated(false);
    setCompanyInfo(null);
  };

  const copyCommand = (text, type) => {
    navigator.clipboard.writeText(text);
    if (type === 'sdk') {
      setCopiedSdk(true);
      setTimeout(() => setCopiedSdk(false), 2000);
    } else if (type === 'node') {
      setCopiedNode(true);
      setTimeout(() => setCopiedNode(false), 2000);
    }
    toast.success('Command copied!');
  };

  // --- RENDER: LOGIN ---
  if (!authenticated) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center p-4 relative z-50">
        <motion.div
          className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <ShieldCheck className="w-10 h-10 text-indigo-500" />
            <h1 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>FinSecure Access</h1>
          </div>
          <p className="text-[#A1A1AA] text-sm mb-6">Please authenticate your banking node to access the federated network.</p>
          <div className="space-y-4">
            <div>
              <label className="block text-sm mb-2 text-[#A1A1AA]">API Key</label>
              <Input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-[#050505] border-white/10 focus:border-indigo-500"
                placeholder="fs_xxxxxxxxxxxxx"
              />
            </div>
            <Button onClick={() => { localStorage.setItem('finsecure_api_key', apiKey); verifyApiKey(); }} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white">
              Connect Node
            </Button>
          </div>
        </motion.div>
      </div>
    );
  }

  // --- RENDER: DASHBOARD ---
  return (
    <div className="min-h-screen bg-[#050505] text-[#EDEDED] relative z-50">
      
      {/* --- NAVIGATION BAR --- */}
      <div className="border-b border-white/5 bg-[#0A0A0A] sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-indigo-500" />
            <div>
              <h1 className="text-xl font-bold" style={{fontFamily: 'Chivo, sans-serif'}}>FinSecure</h1>
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${authenticated ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                <p className="text-xs text-[#A1A1AA]">
                  {authenticated ? `Network Online • ${companyInfo?.name || 'Bank Node'}` : 'Disconnected'}
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Button onClick={() => setShowNotifications(!showNotifications)} className="h-9 w-9 p-0 bg-white/5 hover:bg-white/10 border border-white/10 relative">
              <Bell className="w-4 h-4" />
              {unreadCount > 0 && <span className="absolute -top-1 -right-1 bg-rose-500 text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center">{unreadCount}</span>}
            </Button>
            
            <Button onClick={logout} className="h-9 w-9 p-0 bg-white/5 hover:bg-rose-500/10 hover:text-rose-500 border border-white/10">
              <LogOut className="w-4 h-4" />
            </Button>

            <AnimatePresence>
              {showNotifications && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute right-0 top-12 w-96 bg-[#0A0A0A] border border-white/10 rounded-lg shadow-2xl z-50">
                  <div className="p-3 border-b border-white/10 flex justify-between"><h3 className="font-bold text-sm">Notifications</h3><X onClick={() => setShowNotifications(false)} className="w-4 h-4 cursor-pointer" /></div>
                  <div className="max-h-80 overflow-y-auto">
                    {notifications.length === 0 ? <p className="p-4 text-center text-xs text-gray-500">No new notifications</p> : notifications.map(n => (
                      <div key={n.notification_id} onClick={() => markAsRead(n.notification_id)} className={`p-3 border-b border-white/5 cursor-pointer ${!n.read ? 'bg-indigo-500/5' : ''}`}>
                        <p className="font-semibold text-sm">{n.title}</p>
                        <p className="text-xs text-[#A1A1AA]">{n.message}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* --- MAIN CONTENT --- */}
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

        {/* Dashboard Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="bg-[#0A0A0A] border border-white/5 mb-6">
            <TabsTrigger value="overview">Live Analytics</TabsTrigger>
            <TabsTrigger value="companies">Network Nodes</TabsTrigger>
            <TabsTrigger value="history">My History</TabsTrigger>
            <TabsTrigger value="setup">Client Setup</TabsTrigger>
          </TabsList>

          {/* TAB 1: OVERVIEW */}
          <TabsContent value="overview">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6 h-[400px]">
              <h3 className="text-xl font-bold mb-4">Global Accuracy</h3>
              {trainingRounds.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingRounds.slice(-20)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="round" stroke="#525252" />
                    <YAxis domain={[0, 1]} stroke="#525252" />
                    <Tooltip contentStyle={{ backgroundColor: '#171717', border: '1px solid #333' }} />
                    <Line type="monotone" dataKey="accuracy" stroke="#6366F1" strokeWidth={3} dot={{r: 4}} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-gray-500">
                  <Activity className="w-12 h-12 mb-2 opacity-20"/>
                  <p>Waiting for first training round...</p>
                </div>
              )}
            </Card>
          </TabsContent>

          {/* TAB 2: COMPANIES */}
          <TabsContent value="companies">
            <div className="grid gap-3">
              {companies.map((c, i) => (
                <Card key={i} className="bg-[#121212] border-white/5 p-4 flex justify-between">
                  <div className="flex items-center gap-3"><div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-500 font-bold">{c.name[0]}</div><div><p>{c.name}</p><p className="text-xs text-[#737373]">{c.email}</p></div></div>
                  <Badge className="bg-emerald-500/10 text-emerald-500">Active</Badge>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* TAB 3: HISTORY */}
          <TabsContent value="history">
            <Card className="bg-[#0A0A0A] border-white/5 p-6">
              <h3 className="text-xl font-bold mb-6">Contribution History</h3>
              {myUpdates.length === 0 ? (
                <p className="text-gray-500 text-center py-10">No updates found</p>
              ) : (
                <table className="w-full text-sm text-left text-[#A1A1AA]">
                  <thead className="bg-[#171717] text-xs uppercase"><tr><th className="px-6 py-3">Round</th><th className="px-6 py-3">Accuracy</th><th className="px-6 py-3">Time</th></tr></thead>
                  <tbody>
                    {myUpdates.map((u, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-[#121212]">
                        <td className="px-6 py-4 font-mono text-indigo-400">{u.round_id}</td>
                        <td className="px-6 py-4 text-emerald-400">{(u.metrics.accuracy * 100).toFixed(2)}%</td>
                        <td className="px-6 py-4">{new Date(u.timestamp).toLocaleTimeString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </TabsContent>

          {/* TAB 4: SETUP (UPDATED WITH SDK) */}
          <TabsContent value="setup">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6">Client SDK Setup</h3>
              
              <div className="space-y-6">
                
                {/* STEP 1: SDK */}
                <div className="p-4 border border-white/10 rounded-lg bg-[#050505]">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-bold flex items-center gap-2"><Package className="w-4 h-4 text-indigo-500"/> Step 1: Download SDK</h4>
                    <Button variant="ghost" size="sm" onClick={() => copyCommand(`curl -o finsecure_sdk.py ${API}/client/sdk`, 'sdk')}>
                      {copiedSdk ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                    </Button>
                  </div>
                  <code className="text-sm font-mono text-emerald-400">curl -o finsecure_sdk.py {API}/client/sdk</code>
                </div>

                {/* STEP 2: NODE SCRIPT */}
                <div className="p-4 border border-white/10 rounded-lg bg-[#050505]">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-bold flex items-center gap-2"><Terminal className="w-4 h-4 text-indigo-500"/> Step 2: Download Node Script</h4>
                    <Button variant="ghost" size="sm" onClick={() => copyCommand(`curl -H "X-API-Key: ${apiKey}" -o bank_node.py ${API}/client/script`, 'node')}>
                      {copiedNode ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                    </Button>
                  </div>
                  <code className="text-sm font-mono text-emerald-400">curl -H "X-API-Key: {apiKey}" -o bank_node.py {API}/client/script</code>
                </div>

                {/* STEP 3: RUN */}
                <div className="p-4 border border-white/10 rounded-lg bg-[#050505]">
                  <h4 className="font-bold mb-2">Step 3: Install & Run</h4>
                  <div className="space-y-2 font-mono text-sm text-[#A1A1AA]">
                    <p>pip install tensorflow requests numpy</p>
                    <p className="text-emerald-400">python bank_node.py</p>
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
