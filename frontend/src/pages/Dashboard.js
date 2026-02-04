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
const API = `${https://finsecure-ochi.onrender.com}/api`;

const Dashboard = () => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('finsecure_api_key') || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [trainingRounds, setTrainingRounds] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [companyInfo, setCompanyInfo] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    if (apiKey) {
      verifyApiKey();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (authenticated) {
      // Poll for notifications every 30 seconds
      const interval = setInterval(() => {
        loadNotifications();
      }, 30000);
      return () => clearInterval(interval);
    }
  }, [authenticated]);

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
      
      // Load notifications
      await loadNotifications();
    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const loadNotifications = async () => {
    try {
      const [notifRes, countRes] = await Promise.all([
        axios.get(`${API}/notifications`, {
          headers: { 'X-API-Key': apiKey }
        }),
        axios.get(`${API}/notifications/unread/count`, {
          headers: { 'X-API-Key': apiKey }
        })
      ]);
      
      setNotifications(notifRes.data);
      setUnreadCount(countRes.data.unread_count);
      
      // Show toast for new unread notifications
      const newUnread = notifRes.data.filter(n => !n.read);
      if (newUnread.length > 0 && newUnread[0].type === 'success') {
        toast.success(newUnread[0].title, {
          description: newUnread[0].message
        });
      }
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
  };

  const markAsRead = async (notificationId) => {
    try {
      await axios.post(
        `${API}/notifications/${notificationId}/read`,
        {},
        { headers: { 'X-API-Key': apiKey } }
      );
      
      // Update local state
      setNotifications(notifications.map(n => 
        n.notification_id === notificationId ? { ...n, read: true } : n
      ));
      setUnreadCount(Math.max(0, unreadCount - 1));
    } catch (error) {
      console.error('Failed to mark as read:', error);
    }
  };

  const handleConnect = () => {
    localStorage.setItem('finsecure_api_key', apiKey);
    verifyApiKey();
  };

  const downloadClientScript = async () => {
    try {
      const response = await axios.get(`${API}/client/script`, {
        headers: { 'X-API-Key': apiKey }
      });
      
      if (!response.data || !response.data.content || !response.data.filename) {
        throw new Error('Invalid response from server');
      }
      
      // Create blob and download
      const blob = new Blob([response.data.content], { type: 'text/x-python' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = response.data.filename;
      document.body.appendChild(a);
      a.click();
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 100);
      
      toast.success(`Downloaded: ${response.data.filename}`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error(error.response?.data?.detail || 'Failed to download script. Please try again.');
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
            <h1 className="text-2xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>FinSecure Dashboard</h1>
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
                placeholder="fs_xxxxxxxxxxxxx"
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
              <h1 className="text-xl font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>FinSecure</h1>
              {companyInfo && (
                <p className="text-xs text-[#A1A1AA]">{companyInfo.name}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Button
                onClick={() => setShowNotifications(!showNotifications)}
                className="h-9 w-9 p-0 rounded-md bg-white/5 hover:bg-white/10 text-white border border-white/10 relative"
                data-testid="notifications-btn"
              >
                <Bell className="w-4 h-4" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-rose-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {unreadCount}
                  </span>
                )}
              </Button>
              
              {/* Notifications Dropdown */}
              <AnimatePresence>
                {showNotifications && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute right-0 top-12 w-96 bg-[#0A0A0A] border border-white/10 rounded-lg shadow-xl z-50"
                    data-testid="notifications-dropdown"
                  >
                    <div className="p-4 border-b border-white/10 flex justify-between items-center">
                      <h3 className="font-bold" style={{ fontFamily: 'Chivo, sans-serif' }}>Notifications</h3>
                      <Button
                        onClick={() => setShowNotifications(false)}
                        className="h-6 w-6 p-0 bg-transparent hover:bg-white/5"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.length === 0 ? (
                        <div className="p-8 text-center text-[#A1A1AA]">
                          <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
                          <p>No notifications yet</p>
                        </div>
                      ) : (
                        <div className="divide-y divide-white/5">
                          {notifications.map((notif) => (
                            <div
                              key={notif.notification_id}
                              className={`p-4 hover:bg-white/5 cursor-pointer transition-colors ${
                                !notif.read ? 'bg-indigo-500/5' : ''
                              }`}
                              onClick={() => markAsRead(notif.notification_id)}
                              data-testid={`notification-${notif.notification_id}`}
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                    <p className="font-medium text-sm">{notif.title}</p>
                                    {!notif.read && (
                                      <span className="w-2 h-2 bg-indigo-500 rounded-full"></span>
                                    )}
                                  </div>
                                  <p className="text-xs text-[#A1A1AA]">{notif.message}</p>
                                  <p className="text-xs text-[#A1A1AA] mt-1">
                                    {new Date(notif.created_at).toLocaleString()}
                                  </p>
                                </div>
                                <Badge
                                  className={`text-xs ${
                                    notif.type === 'success'
                                      ? 'bg-emerald-500/20 text-emerald-500'
                                      : notif.type === 'error'
                                      ? 'bg-rose-500/20 text-rose-500'
                                      : 'bg-indigo-500/20 text-indigo-500'
                                  }`}
                                >
                                  {notif.type}
                                </Badge>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
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
