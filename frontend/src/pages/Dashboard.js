import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, BrainCircuit, Download, Activity, Network, Settings2, ShieldCheck, Eye, EyeOff, Bell, X, Terminal, Copy, Check, History, Package } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = (process.env.REACT_APP_BACKEND_URL || 'https://finsecure-ochi.onrender.com').replace(/\/$/, '');
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('finsecure_api_key') || '');
  const [authenticated, setAuthenticated] = useState(false);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [trainingRounds, setTrainingRounds] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [companyInfo, setCompanyInfo] = useState(null);
  const [myUpdates, setMyUpdates] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);
  const [copiedSdk, setCopiedSdk] = useState(false);
  const [copiedNode, setCopiedNode] = useState(false);

  useEffect(() => {
    if (apiKey) verifyApiKey();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      const response = await axios.get(`${API}/auth/verify`, { headers: { 'X-API-Key': apiKey } });
      setAuthenticated(true);
      setCompanyInfo(response.data);
      toast.success('Connected successfully!');
    } catch (error) {
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

  const markAsRead = (id) => {
    setNotifications(prev => prev.map(n => n.notification_id === id ? { ...n, read: true } : n));
    setUnreadCount(Math.max(0, unreadCount - 1));
  };

  const copyCommand = (text, type) => {
    navigator.clipboard.writeText(text);
    if (type === 'sdk') { setCopiedSdk(true); setTimeout(() => setCopiedSdk(false), 2000); }
    if (type === 'node') { setCopiedNode(true); setTimeout(() => setCopiedNode(false), 2000); }
    toast.success('Command copied!');
  };

  if (!authenticated) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center p-4">
        <motion.div className="bg-[#0A0A0A] border border-white/10 rounded-xl p-8 max-w-md w-full" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-3 mb-6">
            <ShieldCheck className="w-10 h-10 text-indigo-500" />
            <h1 className="text-2xl font-bold">FinSecure Dashboard</h1>
          </div>
          <div className="space-y-4">
            <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="bg-[#050505] border-white/10" placeholder="API Key" />
            <Button onClick={() => { localStorage.setItem('finsecure_api_key', apiKey); verifyApiKey(); }} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white">Connect</Button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505] text-[#EDEDED]">
      <div className="border-b border-white/5 bg-[#0A0A0A] sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-indigo-500" />
            <div>
              <h1 className="text-xl font-bold">FinSecure</h1>
              <p className="text-xs text-[#A1A1AA]">Network Online • {companyInfo?.name}</p>
            </div>
          </div>
          <Button onClick={() => setShowNotifications(!showNotifications)} className="h-9 w-9 p-0 bg-white/5 hover:bg-white/10 border border-white/10 relative">
            <Bell className="w-4 h-4" />
            {unreadCount > 0 && <span className="absolute -top-1 -right-1 bg-rose-500 text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center">{unreadCount}</span>}
          </Button>
          <AnimatePresence>
            {showNotifications && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute right-0 top-12 w-96 bg-[#0A0A0A] border border-white/10 rounded-lg shadow-2xl z-50">
                <div className="p-3 border-b border-white/10 flex justify-between"><h3 className="font-bold text-sm">Notifications</h3><X onClick={() => setShowNotifications(false)} className="w-4 h-4 cursor-pointer" /></div>
                <div className="max-h-80 overflow-y-auto">
                  {notifications.map(n => (
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

      <div className="max-w-7xl mx-auto px-6 py-8">
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="bg-[#0A0A0A] border border-white/5 mb-6">
            <TabsTrigger value="overview">Live Analytics</TabsTrigger>
            <TabsTrigger value="companies">Network Nodes</TabsTrigger>
            <TabsTrigger value="history">My History</TabsTrigger>
            <TabsTrigger value="setup">Client Setup</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6 h-[400px]">
              <h3 className="text-xl font-bold mb-4">Global Accuracy</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingRounds.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                  <XAxis dataKey="round" stroke="#525252" />
                  <YAxis domain={[0, 1]} stroke="#525252" />
                  <Tooltip contentStyle={{ backgroundColor: '#171717', border: '1px solid #333' }} />
                  <Line type="monotone" dataKey="accuracy" stroke="#6366F1" strokeWidth={3} dot={{r: 4}} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </TabsContent>

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

          <TabsContent value="history">
            <Card className="bg-[#0A0A0A] border-white/5 p-6">
              <h3 className="text-xl font-bold mb-6">Contribution History</h3>
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
            </Card>
          </TabsContent>

          <TabsContent value="setup">
            <Card className="bg-[#0A0A0A] border border-white/5 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-6">Client SDK Setup</h3>
              
              <div className="space-y-6">
                <div className="p-4 border border-white/10 rounded-lg bg-[#050505]">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-bold flex items-center gap-2"><Package className="w-4 h-4 text-indigo-500"/> Step 1: Download SDK</h4>
                    <Button variant="ghost" size="sm" onClick={() => copyCommand(`curl -o finsecure_sdk.py ${API}/client/sdk`, 'sdk')}>
                      {copiedSdk ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                    </Button>
                  </div>
                  <code className="text-sm font-mono text-emerald-400">curl -o finsecure_sdk.py {API}/client/sdk</code>
                </div>

                <div className="p-4 border border-white/10 rounded-lg bg-[#050505]">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-bold flex items-center gap-2"><Terminal className="w-4 h-4 text-indigo-500"/> Step 2: Download Node Script</h4>
                    <Button variant="ghost" size="sm" onClick={() => copyCommand(`curl -H "X-API-Key: ${apiKey}" -o bank_node.py ${API}/client/script`, 'node')}>
                      {copiedNode ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                    </Button>
                  </div>
                  <code className="text-sm font-mono text-emerald-400">curl -H "X-API-Key: {apiKey}" -o bank_node.py {API}/client/script</code>
                </div>

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
