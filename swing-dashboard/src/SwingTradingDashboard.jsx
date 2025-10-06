import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, Activity, BarChart2, Settings, Upload, Play, AlertCircle, CheckCircle, Clock, DollarSign, Target, Shield, Download, Save } from 'lucide-react';

const SwingTradingDashboard = () => {
  // Load initial data from storage or use defaults
  const loadFromStorage = () => {
    try {
      const stored = JSON.parse(window.localStorage?.getItem('swingTradingConfig') || '{}');
      return stored;
    } catch {
      return {};
    }
  };

  const initialConfig = loadFromStorage();

  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [apiKey, setApiKey] = useState('');
  
  const [modelStatus, setModelStatus] = useState(initialConfig.modelStatus || {
    trained: true,
    lastTraining: '2025-10-01',
    features: 127,
    accuracy: 0.8547
  });
  
  const [trainingConfig, setTrainingConfig] = useState(initialConfig.trainingConfig || {
    lookforwardPeriods: 10,
    swingThreshold: 0.15,
    estimators: 100,
    maxDepth: 8
  });
  
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  
  const [backtestResults, setBacktestResults] = useState(null);
  const [selectedSymbols, setSelectedSymbols] = useState(['AAPL']);
  const [dateRange, setDateRange] = useState({ start: '2025-07-01', end: '2025-10-01' });
  
  const [watchlist, setWatchlist] = useState(initialConfig.watchlist || [
    { symbol: 'AAPL', price: 178.45, change: 2.34, probability: 0.82, signal: 'BUY', stopLoss: 172.30, takeProfit: 186.20 },
    { symbol: 'MSFT', price: 412.89, change: -0.67, probability: 0.45, signal: 'HOLD', stopLoss: 405.50, takeProfit: 422.10 },
    { symbol: 'GOOGL', price: 142.33, change: 1.89, probability: 0.76, signal: 'BUY', stopLoss: 138.20, takeProfit: 148.50 },
    { symbol: 'TSLA', price: 267.82, change: -3.21, probability: 0.23, signal: 'HOLD', stopLoss: 260.00, takeProfit: 280.00 }
  ]);
  
  const [alerts, setAlerts] = useState(initialConfig.activeSignals || [
    { symbol: 'AAPL', message: 'Strong buy signal detected', confidence: 0.82, time: '10:34 AM' },
    { symbol: 'GOOGL', message: 'Swing opportunity confirmed', confidence: 0.76, time: '10:12 AM' }
  ]);
  
  const [riskSettings, setRiskSettings] = useState(initialConfig.riskSettings || {
    riskRewardRatio: 0.5,
    minStopLoss: 0.01,
    atrMultiplier: 2.0
  });
  
  const [notificationSettings, setNotificationSettings] = useState(initialConfig.notificationSettings || {
    emailAlerts: true,
    desktopAlerts: true
  });
  
  const [autoRetrainEnabled, setAutoRetrainEnabled] = useState(initialConfig.autoRetrainEnabled || false);
  
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showSaveNotification, setShowSaveNotification] = useState(false);

  // Save configuration to JSON structure
  const saveConfiguration = () => {
    const config = {
      modelStatus,
      trainingConfig,
      watchlist,
      activeSignals: alerts,
      riskSettings,
      notificationSettings,
      autoRetrainEnabled,
      lastSaved: new Date().toISOString()
    };
    
    // In a real app, this would save to a JSON file
    // For now, we'll use localStorage as a simulation
    try {
      window.localStorage?.setItem('swingTradingConfig', JSON.stringify(config));
      setShowSaveNotification(true);
      setTimeout(() => setShowSaveNotification(false), 3000);
    } catch (error) {
      console.error('Error saving configuration:', error);
    }
  };

  // Auto-save on critical changes
  useEffect(() => {
    const timer = setTimeout(() => {
      saveConfiguration();
    }, 1000);
    return () => clearTimeout(timer);
  }, [modelStatus, trainingConfig, watchlist, alerts, riskSettings, notificationSettings, autoRetrainEnabled]);

  // Download configuration as JSON file
  const downloadConfig = () => {
    const config = {
      modelStatus,
      trainingConfig,
      watchlist,
      activeSignals: alerts,
      riskSettings,
      notificationSettings,
      autoRetrainEnabled,
      exportDate: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'swing_trading_config.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  // Mock data for charts
  const equityCurveData = [
    { date: 'Jul 1', value: 10000 },
    { date: 'Jul 15', value: 10450 },
    { date: 'Aug 1', value: 10890 },
    { date: 'Aug 15', value: 10650 },
    { date: 'Sep 1', value: 11200 },
    { date: 'Sep 15', value: 11890 },
    { date: 'Oct 1', value: 12340 }
  ];

  const tradeDistribution = [
    { name: 'Stop-Loss', value: 8, winRate: 25 },
    { name: 'Take-Profit', value: 15, winRate: 87 },
    { name: 'Max Time', value: 7, winRate: 43 }
  ];

  const COLORS = ['#ef4444', '#10b981', '#f59e0b'];

  // Training simulation
  const handleTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          const newModelStatus = {
            ...modelStatus,
            trained: true,
            lastTraining: new Date().toISOString().split('T')[0],
            accuracy: 0.8547
          };
          setModelStatus(newModelStatus);
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };

  // File upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file.name);
    }
  };

  // Backtest simulation
  const runBacktest = () => {
    setBacktestResults({
      totalTrades: 30,
      winRate: 0.67,
      avgProfit: 0.0423,
      totalReturn: 0.234,
      bestTrade: 0.089,
      worstTrade: -0.032
    });
  };

  // Add symbol to watchlist
  const addToWatchlist = () => {
    const symbol = prompt('Enter symbol to add (e.g., AAPL):');
    if (symbol && symbol.trim()) {
      const newItem = {
        symbol: symbol.trim().toUpperCase(),
        price: 0,
        change: 0,
        probability: 0,
        signal: 'HOLD',
        stopLoss: 0,
        takeProfit: 0
      };
      setWatchlist([...watchlist, newItem]);
    }
  };

  // Remove symbol from watchlist
  const removeFromWatchlist = (symbol) => {
    setWatchlist(watchlist.filter(item => item.symbol !== symbol));
  };

  // Navigation menu
  const NavMenu = () => (
    <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-8 h-8 text-emerald-500" />
            <span className="text-xl font-bold text-white">SwingTrader ML</span>
          </div>
          <nav className="flex space-x-1">
            {['dashboard', 'train', 'backtest', 'monitor', 'settings'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  activeTab === tab
                    ? 'bg-emerald-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center space-x-4">
          {showSaveNotification && (
            <div className="flex items-center space-x-2 px-4 py-2 bg-emerald-600 rounded-lg animate-pulse">
              <CheckCircle className="w-4 h-4 text-white" />
              <span className="text-sm text-white">Saved</span>
            </div>
          )}
          <button
            onClick={downloadConfig}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <Download className="w-4 h-4 text-gray-300" />
            <span className="text-sm text-gray-300">Export Config</span>
          </button>
          <div className="flex items-center space-x-2 px-4 py-2 bg-gray-800 rounded-lg">
            <div className={`w-2 h-2 rounded-full ${modelStatus.trained ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-300">Model {modelStatus.trained ? 'Active' : 'Inactive'}</span>
          </div>
        </div>
      </div>
    </div>
  );

  // Dashboard Tab
  const DashboardView = () => (
    <div className="space-y-6">
      {/* Overview Panel */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Model Status</span>
            <CheckCircle className="w-5 h-5 text-emerald-500" />
          </div>
          <div className="text-2xl font-bold text-white">{modelStatus.trained ? 'Trained' : 'Untrained'}</div>
          <div className="text-xs text-gray-500 mt-1">Last: {modelStatus.lastTraining}</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Features</span>
            <Activity className="w-5 h-5 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-white">{modelStatus.features}</div>
          <div className="text-xs text-gray-500 mt-1">Technical indicators</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Accuracy</span>
            <Target className="w-5 h-5 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-white">{(modelStatus.accuracy * 100).toFixed(2)}%</div>
          <div className="text-xs text-gray-500 mt-1">Test set performance</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Active Signals</span>
            <AlertCircle className="w-5 h-5 text-yellow-500" />
          </div>
          <div className="text-2xl font-bold text-white">{alerts.length}</div>
          <div className="text-xs text-gray-500 mt-1">High confidence</div>
        </div>
      </div>

      {/* Market Summary */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Market Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Symbol</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Price</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Change</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Probability</th>
                <th className="text-center py-3 px-4 text-gray-400 font-medium">Signal</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map((item, idx) => (
                <tr key={idx} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                  <td className="py-4 px-4 text-white font-semibold">{item.symbol}</td>
                  <td className="py-4 px-4 text-right text-white">${item.price.toFixed(2)}</td>
                  <td className={`py-4 px-4 text-right font-medium ${item.change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                    {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
                  </td>
                  <td className="py-4 px-4 text-right text-white">{(item.probability * 100).toFixed(0)}%</td>
                  <td className="py-4 px-4 text-center">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      item.signal === 'BUY' ? 'bg-emerald-500/20 text-emerald-500' : 'bg-gray-700 text-gray-400'
                    }`}>
                      {item.signal}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Alerts</h3>
        <div className="space-y-3">
          {alerts.map((alert, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 bg-gray-900 rounded-lg border border-gray-700">
              <div className="flex items-center space-x-4">
                <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-emerald-500" />
                </div>
                <div>
                  <div className="text-white font-semibold">{alert.symbol}</div>
                  <div className="text-sm text-gray-400">{alert.message}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-emerald-500 font-semibold">{(alert.confidence * 100).toFixed(0)}%</div>
                <div className="text-xs text-gray-500">{alert.time}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // Training Tab
  const TrainingView = () => (
    <div className="space-y-6">
      {/* File Upload */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Upload className="w-5 h-5 mr-2" />
          Upload Historical Data
        </h3>
        <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-emerald-500 transition-colors cursor-pointer">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
            <p className="text-white mb-2">Drop your CSV or Excel files here</p>
            <p className="text-sm text-gray-500">or click to browse</p>
            {uploadedFile && (
              <div className="mt-4 text-emerald-500 font-medium">✓ {uploadedFile}</div>
            )}
          </label>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Training Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Lookforward Periods</label>
            <input
              type="number"
              value={trainingConfig.lookforwardPeriods}
              onChange={(e) => setTrainingConfig({...trainingConfig, lookforwardPeriods: parseInt(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Swing Threshold</label>
            <input
              type="number"
              step="0.01"
              value={trainingConfig.swingThreshold}
              onChange={(e) => setTrainingConfig({...trainingConfig, swingThreshold: parseFloat(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">N Estimators</label>
            <input
              type="number"
              value={trainingConfig.estimators}
              onChange={(e) => setTrainingConfig({...trainingConfig, estimators: parseInt(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Max Depth</label>
            <input
              type="number"
              value={trainingConfig.maxDepth}
              onChange={(e) => setTrainingConfig({...trainingConfig, maxDepth: parseInt(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
        <button
          onClick={handleTraining}
          disabled={isTraining}
          className="mt-6 w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center"
        >
          <Play className="w-5 h-5 mr-2" />
          {isTraining ? 'Training...' : 'Start Training'}
        </button>
      </div>

      {/* Progress Bar */}
      {isTraining && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Training Progress</h3>
          <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
            <div
              className="bg-emerald-500 h-full transition-all duration-300"
              style={{ width: `${trainingProgress}%` }}
            ></div>
          </div>
          <div className="text-center text-gray-400 mt-2">{trainingProgress}%</div>
        </div>
      )}

      {/* Results Card */}
      {modelStatus.trained && !isTraining && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Training Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Accuracy</div>
              <div className="text-2xl font-bold text-emerald-500">{(modelStatus.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Precision</div>
              <div className="text-2xl font-bold text-blue-500">83.21%</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Recall</div>
              <div className="text-2xl font-bold text-purple-500">78.95%</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">F1-Score</div>
              <div className="text-2xl font-bold text-yellow-500">81.02%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // Backtest Tab
  const BacktestView = () => (
    <div className="space-y-6">
      {/* Configuration */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Backtest Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Symbol(s)</label>
            <input
              type="text"
              value={selectedSymbols.join(', ')}
              onChange={(e) => setSelectedSymbols(e.target.value.split(',').map(s => s.trim()))}
              placeholder="AAPL, MSFT, GOOGL"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Start Date</label>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">End Date</label>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
        <button
          onClick={runBacktest}
          className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center"
        >
          <BarChart2 className="w-5 h-5 mr-2" />
          Run Backtest
        </button>
      </div>

      {backtestResults && (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Total Trades</div>
              <div className="text-3xl font-bold text-white">{backtestResults.totalTrades}</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Win Rate</div>
              <div className="text-3xl font-bold text-emerald-500">{(backtestResults.winRate * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Total Return</div>
              <div className="text-3xl font-bold text-blue-500">{(backtestResults.totalReturn * 100).toFixed(2)}%</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Avg Profit/Trade</div>
              <div className="text-3xl font-bold text-purple-500">{(backtestResults.avgProfit * 100).toFixed(2)}%</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Best Trade</div>
              <div className="text-3xl font-bold text-emerald-500">+{(backtestResults.bestTrade * 100).toFixed(2)}%</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-gray-400 text-sm mb-2">Worst Trade</div>
              <div className="text-3xl font-bold text-red-500">{(backtestResults.worstTrade * 100).toFixed(2)}%</div>
            </div>
          </div>

          {/* Equity Curve */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={equityCurveData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Trade Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Exit Reasons</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={tradeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {tradeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Win Rate by Exit</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={tradeDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                  <Bar dataKey="winRate" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );

  // Live Monitor Tab
  const MonitorView = () => (
    <div className="space-y-6">
      {/* Watchlist */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Watchlist</h3>
          <button 
            onClick={addToWatchlist}
            className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            + Add Symbol
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Symbol</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Price</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Change</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Probability</th>
                <th className="text-center py-3 px-4 text-gray-400 font-medium">Signal</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Stop Loss</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Take Profit</th>
                <th className="text-center py-3 px-4 text-gray-400 font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map((item, idx) => (
                <tr key={idx} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                  <td className="py-4 px-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-white font-semibold">{item.symbol}</span>
                    </div>
                  </td>
                  <td className="py-4 px-4 text-right text-white">${item.price.toFixed(2)}</td>
                  <td className={`py-4 px-4 text-right font-medium ${item.change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                    {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
                  </td>
                  <td className="py-4 px-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-emerald-500 h-2 rounded-full"
                          style={{ width: `${item.probability * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-white text-sm">{(item.probability * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="py-4 px-4 text-center">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      item.signal === 'BUY' ? 'bg-emerald-500/20 text-emerald-500' : 'bg-gray-700 text-gray-400'
                    }`}>
                      {item.signal}
                    </span>
                  </td>
                  <td className="py-4 px-4 text-right text-red-400">${item.stopLoss.toFixed(2)}</td>
                  <td className="py-4 px-4 text-right text-emerald-400">${item.takeProfit.toFixed(2)}</td>
                  <td className="py-4 px-4 text-center">
                    <button
                      onClick={() => removeFromWatchlist(item.symbol)}
                      className="text-red-500 hover:text-red-400 text-sm font-medium"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Active Alerts */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <AlertCircle className="w-5 h-5 mr-2 text-yellow-500" />
          Active Signals
        </h3>
        <div className="space-y-3">
          {alerts.map((alert, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 bg-gradient-to-r from-emerald-900/20 to-transparent rounded-lg border border-emerald-500/30">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-emerald-500" />
                </div>
                <div>
                  <div className="text-white font-bold text-lg">{alert.symbol}</div>
                  <div className="text-sm text-gray-300">{alert.message}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-emerald-500">{(alert.confidence * 100).toFixed(0)}%</div>
                <div className="text-xs text-gray-400 flex items-center justify-end mt-1">
                  <Clock className="w-3 h-3 mr-1" />
                  {alert.time}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chart Placeholder */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Price Chart with Signals</h3>
        <div className="bg-gray-900 rounded-lg p-8 text-center border border-gray-700">
          <Activity className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">Candlestick chart with buy/sell signals</p>
          <p className="text-sm text-gray-500 mt-2">Green arrows (Buy) • Red arrows (Sell) • Shaded zones (Stop/Target)</p>
        </div>
      </div>
    </div>
  );

  // Settings Tab
  const SettingsView = () => (
    <div className="space-y-6">
      {/* API Configuration */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">API Configuration</h3>
        <div>
          <label className="block text-sm text-gray-400 mb-2">Alpha Vantage API Key</label>
          <div className="flex items-center space-x-2">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
              placeholder="Enter your API key"
            />
            <button
              onClick={() => alert('API key saved to .env file')}
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
            >
              Save
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">⚠️ API key will be stored in .env file (not in JSON config)</p>
        </div>
      </div>

      {/* Risk Settings */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Shield className="w-5 h-5 mr-2" />
          Risk Management
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Risk-Reward Ratio</label>
            <input
              type="number"
              step="0.1"
              value={riskSettings.riskRewardRatio}
              onChange={(e) => setRiskSettings({...riskSettings, riskRewardRatio: parseFloat(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Min Stop Loss (%)</label>
            <input
              type="number"
              step="0.01"
              value={riskSettings.minStopLoss}
              onChange={(e) => setRiskSettings({...riskSettings, minStopLoss: parseFloat(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">ATR Multiplier</label>
            <input
              type="number"
              step="0.1"
              value={riskSettings.atrMultiplier}
              onChange={(e) => setRiskSettings({...riskSettings, atrMultiplier: parseFloat(e.target.value)})}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
      </div>

      {/* Notifications */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Notifications</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <DollarSign className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <div className="text-white font-medium">Email Alerts</div>
                <div className="text-sm text-gray-400">Receive signals via email</div>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={notificationSettings.emailAlerts}
                onChange={(e) => setNotificationSettings({...notificationSettings, emailAlerts: e.target.checked})}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-yellow-500" />
              </div>
              <div>
                <div className="text-white font-medium">Desktop Alerts</div>
                <div className="text-sm text-gray-400">Browser push notifications</div>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={notificationSettings.desktopAlerts}
                onChange={(e) => setNotificationSettings({...notificationSettings, desktopAlerts: e.target.checked})}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Model Management */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Model Management</h3>
        <div className="space-y-4">
          <button 
            onClick={saveConfiguration}
            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center"
          >
            <Save className="w-5 h-5 mr-2" />
            Save Configuration to JSON
          </button>
          <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors">
            Load Configuration from JSON
          </button>
          
          <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg mt-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Clock className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <div className="text-white font-medium">Auto-Retrain Schedule</div>
                <div className="text-sm text-gray-400">Automatically retrain model periodically</div>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoRetrainEnabled}
                onChange={(e) => setAutoRetrainEnabled(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Configuration Info */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Data Storage Information</h3>
        <div className="space-y-2 text-sm text-gray-400">
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Watchlist data saved to JSON configuration</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Training configuration stored in JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Model status and features tracked in JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Active signals/alerts persisted in JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Risk management settings saved to JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Notification preferences stored in JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
            <span>Auto-retrain schedule flag saved to JSON</span>
          </div>
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
            <span>Alpha Vantage API key stored separately in .env file</span>
          </div>
        </div>
      </div>
    </div>
  );
  
  return (
    <div className="min-h-screen bg-gray-950">
      <NavMenu />
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'train' && <TrainingView />}
        {activeTab === 'backtest' && <BacktestView />}
        {activeTab === 'monitor' && <MonitorView />}
        {activeTab === 'settings' && <SettingsView />}
      </div>
    </div>
  );
};

export default SwingTradingDashboard;