import React, { useEffect, useMemo, useState } from 'react';
import { io } from 'socket.io-client';
import {
  Play, Square, Wallet, TrendingUp, TrendingDown,
  Activity, BarChart3, Clock, Shield, AlertTriangle, DollarSign
} from 'lucide-react';
import './App.css';

function pickLatestSignal(lastSignals) {
  if (!lastSignals) return null;
  const entries = Object.entries(lastSignals);
  if (!entries.length) return null;

  let latest = null;
  for (const [symbol, signalData] of entries) {
    if (!signalData || !signalData.timestamp) continue;
    if (!latest || new Date(signalData.timestamp) > new Date(latest.timestamp)) {
      latest = { ...signalData, symbol };
    }
  }
  return latest;
}

function App() {
  const [connected, setConnected] = useState(false);
  const [botRunning, setBotRunning] = useState(false);
  const [balance, setBalance] = useState(null);
  const [positions, setPositions] = useState([]);
  const [orders, setOrders] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [availableSymbols, setAvailableSymbols] = useState([]);
  const [multiBotStatus, setMultiBotStatus] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastSignal, setLastSignal] = useState(null);

  const apiHost = useMemo(() => window.location.hostname || 'localhost', []);
  const API_BASE = useMemo(() => `http://${apiHost}:5000`, [apiHost]);
  const SOCKET_URL = useMemo(() => `${window.location.protocol}//${apiHost}:5000`, [apiHost]);

  const fetchConfig = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/config`);
      const data = await response.json();
      setConfig(data);
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  };

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/multi/status`);
      const data = await response.json();
      setMultiBotStatus(data);
      setBotRunning(Boolean(data.running));
      setLastSignal(pickLatestSignal(data.last_signals));
    } catch (err) {
      console.error('Failed to fetch multi status:', err);
    }
  };

  const fetchSymbols = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/symbols`);
      const data = await response.json();
      const symbols = data.top_20 || [];
      setAvailableSymbols(symbols);
      if (symbols.includes('BTCUSDT')) {
        setSelectedSymbol('BTCUSDT');
      } else if (symbols.length > 0) {
        setSelectedSymbol(symbols[0]);
      }
    } catch (err) {
      console.error('Failed to fetch symbols:', err);
    }
  };

  useEffect(() => {
    const newSocket = io(SOCKET_URL);

    newSocket.on('connect', () => {
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      setConnected(false);
    });

    newSocket.on('market_update', (data) => {
      if (data.balance) setBalance(data.balance);
      if (data.positions) setPositions(data.positions);
      if (data.current_price) setCurrentPrice(data.current_price);
      if (data.orders) setOrders(data.orders);
      if (data.multi_status) {
        setMultiBotStatus(data.multi_status);
        setBotRunning(Boolean(data.multi_status.running));
        setLastSignal(pickLatestSignal(data.multi_status.last_signals));
      } else if (data.bot_status) {
        setMultiBotStatus(data.bot_status);
        setBotRunning(Boolean(data.bot_status.running));
        setLastSignal(pickLatestSignal(data.bot_status.last_signals));
      }
      setLoading(false);
    });

    fetchConfig();
    fetchStatus();
    fetchSymbols();

    fetch(`${API_BASE}/api/balance`)
      .then(r => r.json())
      .then(data => { if (data && data.total !== undefined) setBalance(data); })
      .catch(() => {});

    fetch(`${API_BASE}/api/positions`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setPositions(data); })
      .catch(() => {});

    fetch(`${API_BASE}/api/orders`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setOrders(data); })
      .catch(() => {});

    return () => {
      newSocket.close();
    };
  }, [SOCKET_URL, API_BASE]);

  useEffect(() => {
    const statusInterval = setInterval(fetchStatus, 5000);
    const balanceInterval = setInterval(() => {
      fetch(`${API_BASE}/api/balance`)
        .then(r => r.json())
        .then(data => { if (data && data.total !== undefined) setBalance(data); })
        .catch(() => {});
      fetch(`${API_BASE}/api/positions`)
        .then(r => r.json())
        .then(data => { if (Array.isArray(data)) setPositions(data); })
        .catch(() => {});
      fetch(`${API_BASE}/api/orders`)
        .then(r => r.json())
        .then(data => { if (Array.isArray(data)) setOrders(data); })
        .catch(() => {});
    }, 10000);
    return () => { clearInterval(statusInterval); clearInterval(balanceInterval); };
  }, [API_BASE]);

  const startBot = async () => {
    try {
      const symbolCount = config?.default_multi_symbol_count || 10;
      const response = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol_count: symbolCount })
      });
      const data = await response.json();
      if (data.success) {
        setBotRunning(true);
        fetchStatus();
      } else {
        setError(data.message || 'Failed to start bot');
      }
    } catch (err) {
      setError('Failed to start bot');
    }
  };

  const stopBot = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setBotRunning(false);
        fetchStatus();
      } else {
        setError(data.message || 'Failed to stop bot');
      }
    } catch (err) {
      setError('Failed to stop bot');
    }
  };

  const closeAllPositions = async () => {
    if (!window.confirm('Are you sure you want to close all positions?')) return;
    try {
      const response = await fetch(`${API_BASE}/api/close-all`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        alert(`Closed ${data.closed_positions} positions`);
      }
    } catch (err) {
      setError('Failed to close positions');
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return '-';
    return new Date(timestamp).toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="loading">
        <Activity size={48} className="spinner" />
        <p>Connecting to trading server...</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <Activity className="logo" size={24} />
          <h1>FutureTestBot Dashboard</h1>
        </div>
        <div className="header-right">
          <div className={`status-indicator ${connected ? 'online' : 'offline'}`}>
            <span className="dot" />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          <div className={`badge ${botRunning ? 'low' : 'high'}`}>
            {botRunning ? 'Running' : 'Stopped'}
          </div>
        </div>
      </header>
      <main className="main">
        {error && <div className="alert error"><AlertTriangle size={16} />{error}</div>}
        <section className="stats-grid">
          <div className="card stat"><Wallet className="stat-icon wallet" /><div className="stat-content"><span className="stat-label">Balance</span><span className="stat-value">{balance?.total ?? '-'}</span></div></div>
          <div className="card stat"><TrendingUp className="stat-icon available" /><div className="stat-content"><span className="stat-label">Positions</span><span className="stat-value">{positions.length}</span></div></div>
          <div className="card stat"><DollarSign className="stat-icon price" /><div className="stat-content"><span className="stat-label">Price</span><span className="stat-value">{currentPrice || '-'}</span></div></div>
          <div className="card stat"><Shield className="stat-icon profit" /><div className="stat-content"><span className="stat-label">Last Signal</span><span className="stat-value">{lastSignal?.signal || '-'}</span></div></div>
        </section>
        <section className="card panel">
          <div className="panel-header">
            <h2>Trading Controls</h2>
            <div className="bot-controls">
              <button className="btn success" onClick={startBot}><Play size={16} />Start</button>
              <button className="btn danger" onClick={stopBot}><Square size={16} />Stop</button>
              <button className="btn warning" onClick={closeAllPositions}><AlertTriangle size={16} />Close All</button>
            </div>
          </div>
          <div className="panel-body">
            <p>Selected symbol: {selectedSymbol}</p>
            <p>Available symbols: {availableSymbols.length}</p>
            <p>Last signal time: {formatTime(lastSignal?.timestamp)}</p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
