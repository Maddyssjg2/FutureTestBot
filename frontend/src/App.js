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

    // Fetch initial balance
    fetch(`${API_BASE}/api/balance`)
      .then(r => r.json())
      .then(data => { if (data && data.total !== undefined) setBalance(data); })
      .catch(() => {});

    // Fetch initial positions
    fetch(`${API_BASE}/api/positions`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setPositions(data); })
      .catch(() => {});

    // Fetch initial orders
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
          <BarChart3 size={28} className="logo" />
          <h1>Binance Futures Multi Bot</h1>
          <span className="badge paper">PAPER TRADING</span>
          {config?.risk_level && (
            <span className={`badge ${config.risk_level}`}>
              {config.risk_level.toUpperCase()} RISK
            </span>
          )}
          <span className="badge multi">
            MULTI ({multiBotStatus?.symbols_monitored || config?.default_multi_symbol_count || 0} pairs)
          </span>
        </div>
        <div className="header-right">
          <div className={`status-indicator ${connected ? 'online' : 'offline'}`}>
            <div className="dot"></div>
            <span>{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div className="bot-controls">
            <button
              className={`btn ${botRunning ? 'danger' : 'success'}`}
              onClick={() => (botRunning ? stopBot() : startBot())}
            >
              {botRunning ? <><Square size={16} /> Stop Bot</> : <><Play size={16} /> Start Bot</>}
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="alert error">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      <main className="main">
        <div className="stats-grid">
          <div className="card stat">
            <div className="stat-icon wallet">
              <Wallet size={24} />
            </div>
            <div className="stat-content">
              <span className="stat-label">Total Balance</span>
              <span className="stat-value">${balance ? balance.total.toFixed(2) : '0.00'} USDT</span>
            </div>
          </div>

          <div className="card stat">
            <div className="stat-icon available">
              <DollarSign size={24} />
            </div>
            <div className="stat-content">
              <span className="stat-label">Available</span>
              <span className="stat-value">${balance ? balance.available.toFixed(2) : '0.00'} USDT</span>
            </div>
          </div>

          <div className="card stat">
            <div className={`stat-icon ${(balance?.unrealized_pnl || 0) >= 0 ? 'profit' : 'loss'}`}>
              {(balance?.unrealized_pnl || 0) >= 0 ? <TrendingUp size={24} /> : <TrendingDown size={24} />}
            </div>
            <div className="stat-content">
              <span className="stat-label">Unrealized P&amp;L</span>
              <span className={`stat-value ${(balance?.unrealized_pnl || 0) >= 0 ? 'profit' : 'loss'}`}>
                {balance?.unrealized_pnl >= 0 ? '+' : ''}${balance ? balance.unrealized_pnl.toFixed(2) : '0.00'}
              </span>
            </div>
          </div>

          <div className="card stat">
            <div className="stat-icon price">
              <Activity size={24} />
            </div>
            <div className="stat-content">
              <span className="stat-label">{selectedSymbol}</span>
              <span className="stat-value">
                ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
          </div>
        </div>

        <div className="content-grid">
          <div className="card bot-status">
            <div className="card-header">
              <h3>Bot Status</h3>
              <span className={`status-badge ${botRunning ? 'active' : 'inactive'}`}>
                {botRunning ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="status-content">
              <div className="status-item">
                <span className="status-label">Monitored Symbols</span>
                <span className="status-value">{multiBotStatus?.symbols_monitored || 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Total Trades</span>
                <span className="status-value">{multiBotStatus?.total_trades || 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Win Rate</span>
                <span className={`status-value ${(multiBotStatus?.win_rate || 0) >= 70 ? 'profit' : (multiBotStatus?.win_rate || 0) >= 50 ? 'neutral' : 'loss'}`}>
                  {multiBotStatus?.win_rate?.toFixed(1) || 0}%
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Wins / Losses</span>
                <span className="status-value">
                  <span className="profit">{multiBotStatus?.winning_trades || 0}</span>
                  {' / '}
                  <span className="loss">{multiBotStatus?.losing_trades || 0}</span>
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Open Positions</span>
                <span className="status-value">{multiBotStatus?.open_positions || 0}</span>
              </div>

              {lastSignal && (
                <div className="last-signal">
                  <h4>Latest Signal ({lastSignal.symbol})</h4>
                  <div className={`signal-box ${lastSignal.signal?.toLowerCase() || 'neutral'}`}>
                    <span className="signal-type">{lastSignal.signal || 'NONE'}</span>
                    <span className="signal-confidence">{lastSignal.confidence?.toFixed(1) || 0}% Confidence</span>
                    <span className="signal-time">{formatTime(lastSignal.timestamp)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="card multi-status">
          <div className="card-header">
            <h3>Signals by Symbol</h3>
            <span className={`status-badge ${botRunning ? 'active' : 'inactive'}`}>
              {botRunning ? 'Live' : 'Idle'}
            </span>
          </div>
          <div className="multi-symbols">
            {multiBotStatus?.last_signals && Object.entries(multiBotStatus.last_signals).map(([symbol, data]) => (
              <div key={symbol} className={`symbol-pill ${data.signal?.toLowerCase() || 'neutral'}`}>
                <span className="symbol-name">{symbol}</span>
                {data.signal && (
                  <>
                    <span className="symbol-signal">{data.signal}</span>
                    <span className="symbol-conf">{data.confidence?.toFixed(0)}%</span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="content-grid">
          <div className="card">
            <div className="card-header">
              <h3>Open Positions ({positions.length})</h3>
              {positions.length > 0 && (
                <button className="btn danger small" onClick={closeAllPositions}>
                  Close All
                </button>
              )}
            </div>
            <div className="table-container">
              {positions.length === 0 ? (
                <div className="empty-state">
                  <Shield size={48} />
                  <p>No open positions</p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Size</th>
                      <th>Entry Price</th>
                      <th>Mark Price</th>
                      <th>P&amp;L</th>
                      <th>Leverage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((pos, idx) => (
                      <tr key={idx}>
                        <td>{pos.symbol}</td>
                        <td className={pos.side === 'LONG' ? 'profit' : 'loss'}>{pos.side}</td>
                        <td>{pos.amount.toFixed(4)}</td>
                        <td>${pos.entry_price.toFixed(2)}</td>
                        <td>${pos.mark_price.toFixed(2)}</td>
                        <td className={pos.unrealized_pnl >= 0 ? 'profit' : 'loss'}>
                          {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                        </td>
                        <td>{pos.leverage}x</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3>Recent Orders</h3>
            </div>
            <div className="table-container">
              {orders.length === 0 ? (
                <div className="empty-state">
                  <Clock size={48} />
                  <p>No recent orders</p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Type</th>
                      <th>Status</th>
                      <th>Quantity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.slice(0, 10).map((order, idx) => (
                      <tr key={idx}>
                        <td>{formatTime(order.time)}</td>
                        <td>{order.symbol}</td>
                        <td className={order.side === 'BUY' ? 'profit' : 'loss'}>{order.side}</td>
                        <td>{order.type}</td>
                        <td>
                          <span className={`order-status ${order.status.toLowerCase()}`}>
                            {order.status}
                          </span>
                        </td>
                        <td>{parseFloat(order.origQty).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>

        {config && (
          <div className="card config-card">
            <div className="card-header">
              <h3>Trading Configuration</h3>
            </div>
            <div className="config-grid">
              <div className="config-item">
                <span className="config-label">Mode</span>
                <span className="config-value">MULTI ONLY</span>
              </div>
              <div className="config-item">
                <span className="config-label">Strategy</span>
                <span className="config-value">{(config.strategy_mode || 'rule').toUpperCase()}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Leverage</span>
                <span className="config-value">{config.leverage}x</span>
              </div>
              <div className="config-item">
                <span className="config-label">Trade %</span>
                <span className="config-value">{config.trade_percentage}%</span>
              </div>
              <div className="config-item">
                <span className="config-label">Stop Loss</span>
                <span className="config-value">{config.stop_loss}%</span>
              </div>
              <div className="config-item">
                <span className="config-label">Take Profit</span>
                <span className="config-value">{config.take_profit}%</span>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          <AlertTriangle size={14} />
          Trading involves risk. This is paper trading on Binance Testnet.
        </p>
      </footer>
    </div>
  );
}

export default App;

