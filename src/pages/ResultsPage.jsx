import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faDownload, faDollarSign, faWallet, faBolt, faFilter, faInfoCircle
} from '@fortawesome/free-solid-svg-icons';
import { Pie, Bar } from 'react-chartjs-2';
import Chart from 'chart.js/auto';

Chart.defaults.devicePixelRatio = window.devicePixelRatio || 1;

// Add the ComponentCard component
const ComponentCard = ({ component, room }) => {
  return (
    <div className="component-card">
      <h4>{component.Component_Name || component.name} {component.Quantity > 1 ? `(${component.Quantity}x)` : ''}</h4>
      <p><strong>Category:</strong> {component.Category || component.category}</p>
      <p><strong>Price:</strong> ₹{(component.Price_INR || component.price_inr || 0).toLocaleString()} {component.Quantity > 1 ? 
        `(Total: ₹${(component.Total_Price_INR || (component.Price_INR * component.Quantity) || 0).toLocaleString()})` : ''}</p>
      <p><strong>Efficiency:</strong> {component.Efficiency || component.efficiency || 0}/10</p>
      <p><strong>Reliability:</strong> {component.Reliability || component.reliability || 0}/10</p>
      {(component.Compatibility || component.compatibility) && 
        <p><strong>Compatibility:</strong> {component.Compatibility || component.compatibility}</p>}
    </div>
  );
};

const ResultsPage = () => {
  const navigate = useNavigate();
  const [configs, setConfigs] = useState([]);
  const [active, setActive] = useState(0);
  const [filterCategory, setFilterCategory] = useState('All');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const raw = sessionStorage.getItem('configData');
    if (!raw) return navigate('/');
    try {
      const { configurations = [] } = JSON.parse(raw);
      setConfigs(configurations);
    } catch (e) {
      setError('Failed to parse configuration data.');
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  // Active config metrics
  const cfg = configs[active] || {};
  const total = cfg.total_cost || 0;
  const budget = cfg.budget || 100000;
  const remaining = budget - total;

  // Chart data
  const pieData = useMemo(() => {
    // Check if data exists and is not empty
    const categories = Object.keys(cfg.category_counts || {});
    const counts = Object.values(cfg.category_counts || {});
    
    // Return valid data or empty placeholder if no data
    return {
      labels: categories.length ? categories : ['No Data'],
      datasets: [{
        data: counts.length ? counts : [1],
        backgroundColor: [
          '#6c63ff', '#9d50bb', '#4ecca3', '#ffb142', '#54a0ff', '#ff6b6b', '#ffe761'
        ],
        borderColor: '#fff', // White border for clarity
        borderWidth: 3,
        hoverOffset: 8,
      }]
    };
  }, [cfg]);

  const barData = useMemo(() => {
    // Check if data exists and is not empty
    const categories = Object.keys(cfg.category_costs || {});
    const costs = Object.values(cfg.category_costs || {}).map(c => 
      typeof c === 'object' ? c.total_cost : c
    );
    
    // Return valid data or empty placeholder if no data
    return {
      labels: categories.length ? categories : ['No Data'],
      datasets: [{
        label: 'Cost (₹)',
        data: costs.length ? costs : [0],
        backgroundColor: [
          '#6c63ff', '#9d50bb', '#4ecca3', '#ffb142', '#54a0ff', '#ff6b6b', '#ffe761'
        ],
        borderRadius: 10,
        barPercentage: 0.7,
        categoryPercentage: 0.7,
      }]
    };
  }, [cfg]);

  // Update the chartOptions for the pie chart
  const pieChartOptions = useMemo(() => ({
    plugins: {
      legend: { 
        position: 'bottom', 
        labels: { 
          color: '#e0e0f0', 
          font: { family: 'Montserrat', size: 16, weight: 'bold' }, // Increased font size and weight
          padding: 20,
        } 
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed !== undefined) {
              label += context.parsed.toLocaleString() + ' (' + context.parsed + '%)';
            }
            return label;
          }
        },
        bodyFont: { size: 16, weight: 'bold' }, // Larger tooltip font
        titleFont: { size: 16, weight: 'bold' }
      }
    },
    maintainAspectRatio: false,
    // Remove axes and grid lines for pie chart
    scales: {},
    elements: {
      arc: {
        borderWidth: 3, // White border for clarity
        borderColor: '#fff'
      }
    }
  }), []);

  // Update the chartOptions for the bar chart
  const barChartOptions = useMemo(() => ({
    plugins: {
      legend: { 
        position: 'bottom', 
        labels: { 
          color: '#e0e0f0', 
          font: { family: 'Montserrat', size: 16, weight: 'bold' }, // Larger font
          padding: 20,
        } 
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed !== undefined) {
              label += context.parsed.y !== undefined 
                ? '₹' + context.parsed.y.toLocaleString()
                : context.parsed.toLocaleString();
            }
            return label;
          }
        },
        bodyFont: { size: 16, weight: 'bold' },
        titleFont: { size: 16, weight: 'bold' }
      }
    },
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        ticks: { 
          color: '#e0e0f0',
          font: { family: 'Montserrat', size: 15, weight: 'bold' },
          callback: function(value) {
            return '₹' + value.toLocaleString();
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.12)',
          lineWidth: 1.5,
        }
      },
      x: {
        ticks: { 
          color: '#e0e0f0',
          font: { family: 'Montserrat', size: 15, weight: 'bold' }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.08)'
        }
      }
    },
    elements: {
      bar: {
        borderRadius: 10,
        borderSkipped: false,
        backgroundColor: '#6c63ff',
        barPercentage: 0.7, // Wider bars
        categoryPercentage: 0.7
      }
    }
  }), []);

  // Components list & filter
  const allComponents = useMemo(() =>
    (cfg.room_allocations || []).flatMap(r =>
      (r.components || []).map(c => ({
        name: c.name || c.Component_Name,
        category: c.category || c.Category,
        room: r.name,
        price: c.price_inr || c.Price_INR || 0,
        efficiency: c.efficiency || c.Efficiency,
        reliability: c.reliability || c.Reliability,
        energy: c.energy_rating || c.Energy_Rating || '–',
        quantity: c.Quantity || 1,
        totalPrice: c.Total_Price_INR || (c.Price_INR * (c.Quantity || 1)) || c.price_inr || 0
      }))
    ), [cfg]
  );
  const categories = useMemo(() => ['All', ...new Set(allComponents.map(c => c.category))], [allComponents]);
  const filtered = useMemo(() =>
    filterCategory === 'All' ? allComponents : allComponents.filter(c => c.category === filterCategory)
  , [allComponents, filterCategory]);

  if (loading) return <div className="loading-state">Loading...</div>;
  if (error) return <div className="alert alert-danger">{error}</div>;
  if (!configs.length) return (
    <div className="alert alert-warning">
      No configurations found. <button onClick={() => navigate('/')} className="btn btn-sm btn-outline-light">Home</button>
    </div>
  );

  return (
    <div className="container py-5">
      {/* Header */}
      <div className="text-center mb-5">
        <h1 className="gradient-text display-3 fw-bold">Your Smart Home Plan</h1>
        <p className="subtitle">Sleek, data-driven home automation recommendation</p>
      </div>

      {/* Key Metrics */}
      <div className="d-flex justify-content-center gap-4 mb-5 flex-wrap">
        {[
          { icon: faDollarSign, label: 'TOTAL COST', value: `₹${total.toLocaleString()}` },
          { icon: faWallet,    label: 'REMAINING', value: `₹${remaining.toLocaleString()} (${remaining>=0?'+':'-'}${Math.abs(Math.round((remaining/budget)*100))}%)` },
          { icon: faBolt,      label: 'MONTHLY ENERGY', value: `₹${(cfg.energy_estimates?.monthly_cost||0).toFixed(2)}` }
        ].map((m,i) => (
          <div key={i} className="metric-card">
            <FontAwesomeIcon icon={m.icon} className="metric-icon" />
            <div className="metric-label">{m.label}</div>
            <div className="metric-value">{m.value}</div>
          </div>
        ))}
      </div>

      {/* Tabs & Download */}
      <div className="config-tabs-container mb-5">
        <div className="d-flex justify-content-between align-items-center flex-wrap">
          <div className="config-tabs">
            {configs.map((config, idx) => {
              const configName = ['Balanced', 'Energy Efficient', 'Security Focused'][idx] || `Configuration ${idx+1}`;
              return (
                <button
                  key={idx}
                  className={`config-tab-btn ${idx === active ? 'active' : ''}`}
                  onClick={() => setActive(idx)}
                >
                  {configName} {config.total_cost ? `(₹${Math.round(config.total_cost).toLocaleString()})` : ''}
                </button>
              );
            })}
          </div>
          <button 
            className="btn px-4 py-2 d-flex align-items-center"
            style={{
              background: 'linear-gradient(45deg, #6c63ff, #4ecca3)',
              border: 'none',
              borderRadius: '8px',
              color: 'white',
              boxShadow: '0 4px 15px rgba(108, 99, 255, 0.2)',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease',
            }}
            onMouseOver={e => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 20px rgba(108, 99, 255, 0.3)';
            }}
            onMouseOut={e => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 15px rgba(108, 99, 255, 0.2)';
            }}
            onClick={() => window.open(`/api/download_report/${active}`, '_blank')}
          >
            <FontAwesomeIcon icon={faDownload} className="me-2" /> Download Report
          </button>
        </div>
      </div>

      {/* Charts */}
      <div className="row gy-4 mb-5">
        <div className="col-lg-6">
          <div className="card chart-card p-3 h-100">
            <h4 className="chart-title">Category Distribution</h4>
            <div className="chart-wrapper">
              {Object.keys(cfg.category_counts || {}).length > 0 ? (
                <Pie data={pieData} options={pieChartOptions} />
              ) : (
                <div className="empty-chart-message">
                  <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
                  No category data available
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="col-lg-6">
          <div className="card chart-card p-3 h-100">
            <h4 className="chart-title">Cost Breakdown</h4>
            <div className="chart-wrapper">
              {Object.keys(cfg.category_costs || {}).length > 0 ? (
                <Bar data={barData} options={barChartOptions} />
              ) : (
                <div className="empty-chart-message">
                  <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
                  No cost data available
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Room Allocations - Updated to show quantities */}
      <div className="mb-5">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h2 className="section-heading">Room Allocations</h2>
          <button className="btn btn-sm btn-outline-light">
            <FontAwesomeIcon icon={faFilter} /> Sort
          </button>
        </div>
        <div className="row gy-4">
        {cfg.room_allocations?.map((room, idx) => {
            // Calculate room total by summing component prices (including quantities)
            const roomTotal = room.components?.reduce((sum, comp) => {
              const price = parseFloat(comp.price_inr || comp.Price_INR || 0);
              const quantity = parseInt(comp.Quantity || 1);
              return sum + (price * quantity);
            }, 0);
            
            return (
              <div className="col-md-6 col-lg-4" key={idx}>
                <div className="card room-card h-100 shadow-sm p-3">
                  <div className="d-flex justify-content-between mb-2">
                    <h5 className="mb-0 text-light">{room.name}</h5>
                    <span className="badge bg-secondary">{room.components?.length || 0}</span>
                  </div>
                  <ul className="list-group list-group-flush mb-2">
                    {room.components && room.components.length > 0 ? (
                      room.components.map((component, j) => (
                        <li className="list-group-item d-flex justify-content-between align-items-center py-2" key={j}>
                          <div className="component-name">
                            {component.name || component.Component_Name}
                            {parseInt(component.Quantity) > 1 && 
                              <span className="ms-2 badge bg-primary">{component.Quantity}x</span>
                            }
                          </div>
                          <div className="component-price">
                            {parseInt(component.Quantity) > 1 ? (
                              <>
                                <span className="text-muted small me-2">
                                  ₹{parseFloat(component.price_inr || component.Price_INR || 0).toLocaleString()} × {component.Quantity}
                                </span>
                                ₹{(parseFloat(component.price_inr || component.Price_INR || 0) * parseInt(component.Quantity)).toLocaleString()}
                              </>
                            ) : (
                              <>₹{parseFloat(component.price_inr || component.Price_INR || 0).toLocaleString()}</>
                            )}
                          </div>
                        </li>
                      ))
                    ) : (
                      <li className="list-group-item text-center py-2">No components</li>
                    )}
                  </ul>
                  <div className="text-end room-total-container">
                    <strong className="room-total">Total: ₹{roomTotal?.toLocaleString()}</strong>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Component Details - Make sure quantity is displayed */}
      <div>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h2 className="section-heading">Component Details</h2>
          <select className="form-select w-auto" value={filterCategory} onChange={e=>setFilterCategory(e.target.value)}>
            {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
          </select>
        </div>
        <div className="sleek-table-wrapper">
          <table className="sleek-component-table align-middle">
            <thead>
              <tr>
                <th>Name</th>
                <th>Category</th>
                <th>Room</th>
                <th>Quantity</th>
                <th>Price (₹)</th>
                <th>Efficiency</th>
                <th>Reliability</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((c,i) => (
                <tr key={i}>
                  <td className="comp-name">{c.name}</td>
                  <td>{c.category}</td>
                  <td>{c.room}</td>
                  <td>{parseInt(c.quantity) || 1}</td>
                  <td>
                    {parseInt(c.quantity) > 1 ? (
                      <>₹{parseFloat(c.price).toLocaleString()} × {c.quantity} = ₹{parseFloat(c.totalPrice).toLocaleString()}</>
                    ) : (
                      <>₹{parseFloat(c.price).toLocaleString()}</>
                    )}
                  </td>
                  <td>
                    <span className="efficiency-badge">
                      {c.efficiency !== undefined && c.efficiency !== null
                        ? Math.min(Number(c.efficiency), 10).toFixed(2)
                        : '–'}
                    </span>
                  </td>
                  <td>
                    <span className="reliability-badge">
                      {c.reliability !== undefined && c.reliability !== null
                        ? Math.min(Number(c.reliability), 10).toFixed(2)
                        : '–'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;