import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faChartPie, faDownload, faArrowLeft, faCog, faCheck, faInfoCircle, faBolt, faShieldAlt } from '@fortawesome/free-solid-svg-icons';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

const ResultsPage = () => {
  const navigate = useNavigate();
  const [configurations, setConfigurations] = useState([]);
  const [activeConfig, setActiveConfig] = useState(0);
  const [priorities, setPriorities] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Try to get data from sessionStorage first
    const storedData = sessionStorage.getItem('configData');
    
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData);
        setConfigurations(parsedData.configurations || []);
        setPriorities(parsedData.priorities || {});
        setLoading(false);
      } catch (err) {
        console.error('Error parsing stored data:', err);
        setError('Failed to load configuration data. Please try again.');
        setLoading(false);
      }
    } else {
      // If no data in sessionStorage, redirect back to home
      navigate('/');
    }
  }, [navigate]);
  
  const handleConfigSelect = (index) => {
    setActiveConfig(index);
  };
  
  const handleDownloadReport = (configIndex) => {
    window.open(`/download_report/${configIndex}`, '_blank');
  };
  
  const handleBackToHome = () => {
    navigate('/');
  };
  
  if (loading) {
    return (
      <div className="container d-flex justify-content-center align-items-center" style={{ minHeight: '80vh' }}>
        <div className="text-center">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <h3 className="mt-3">Generating your smart home configurations...</h3>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="container d-flex justify-content-center align-items-center" style={{ minHeight: '80vh' }}>
        <div className="text-center">
          <div className="alert alert-danger" role="alert">
            <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
            {error}
          </div>
          <button className="btn btn-primary mt-3" onClick={handleBackToHome}>
            <FontAwesomeIcon icon={faArrowLeft} className="me-2" />
            Back to Home
          </button>
        </div>
      </div>
    );
  }
  
  if (!configurations || configurations.length === 0) {
    return (
      <div className="container d-flex justify-content-center align-items-center" style={{ minHeight: '80vh' }}>
        <div className="text-center">
          <div className="alert alert-warning" role="alert">
            <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
            No configurations available. Please try again.
          </div>
          <button className="btn btn-primary mt-3" onClick={handleBackToHome}>
            <FontAwesomeIcon icon={faArrowLeft} className="me-2" />
            Back to Home
          </button>
        </div>
      </div>
    );
  }
  
  const currentConfig = configurations[activeConfig] || {};
  
  // Prepare data for pie chart (component distribution)
  const prepareComponentDistributionData = (config) => {
    if (!config || !config.category_counts) return { labels: [], datasets: [] };
    
    const labels = Object.keys(config.category_counts);
    const data = labels.map(category => config.category_counts[category]);
    
    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: [
            'rgba(108, 99, 255, 0.7)',
            'rgba(157, 80, 187, 0.7)',
            'rgba(78, 204, 163, 0.7)',
            'rgba(255, 177, 66, 0.7)',
            'rgba(255, 107, 107, 0.7)',
            'rgba(84, 160, 255, 0.7)'
          ],
          borderColor: [
            'rgba(108, 99, 255, 1)',
            'rgba(157, 80, 187, 1)',
            'rgba(78, 204, 163, 1)',
            'rgba(255, 177, 66, 1)',
            'rgba(255, 107, 107, 1)',
            'rgba(84, 160, 255, 1)'
          ],
          borderWidth: 1
        }
      ]
    };
  };
  
  // Prepare data for bar chart (cost breakdown)
  const prepareCostBreakdownData = (config) => {
    if (!config || !config.category_costs) return { labels: [], datasets: [] };
    
    const labels = Object.keys(config.category_costs);
    const data = labels.map(category => config.category_costs[category].total_cost);
    
    return {
      labels,
      datasets: [
        {
          label: 'Cost (₹)',
          data,
          backgroundColor: 'rgba(108, 99, 255, 0.7)',
          borderColor: 'rgba(108, 99, 255, 1)',
          borderWidth: 1
        }
      ]
    };
  };
  
  const pieChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: '#ffffff'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${value} components`;
          }
        }
      }
    }
  };
  
  const barChartOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          color: '#ffffff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      x: {
        ticks: {
          color: '#ffffff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    },
    plugins: {
      legend: {
        labels: {
          color: '#ffffff'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.raw || 0;
            return `${label}: ₹${value.toLocaleString()}`;
          }
        }
      }
    }
  };
  
  return (
    <div className="container">
      <div className="header">
        <div className="header-icon">
          <FontAwesomeIcon icon={faHome} />
        </div>
        <h1>Smart Home Configuration Results</h1>
        <p className="text-muted">Review your personalized smart home configurations</p>
      </div>
      
      <div className="row mb-4">
        <div className="col-12">
          <button className="btn btn-outline-light" onClick={handleBackToHome}>
            <FontAwesomeIcon icon={faArrowLeft} className="me-2" />
            Back to Home
          </button>
        </div>
      </div>
      
      <div className="row">
        <div className="col-md-3">
          <div className="card mb-4">
            <div className="card-header">Configuration Reports</div>
            <div className="card-body p-0">
              <div className="list-group list-group-flush">
                {configurations.map((config, index) => (
                  <button
                    key={index}
                    className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${index === activeConfig ? 'active' : ''}`}
                    onClick={() => handleConfigSelect(index)}
                    style={{ backgroundColor: index === activeConfig ? 'var(--primary-color)' : 'var(--card-bg)', color: 'var(--text-light)' }}
                  >
                    <span>
                      <FontAwesomeIcon icon={faCog} className="me-2" />
                      {index === 0 ? 'Balanced' : index === 1 ? 'Energy Efficient' : 'Security Focused'}
                    </span>
                    {index === activeConfig && <FontAwesomeIcon icon={faCheck} />}
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          <div className="card mb-4">
            <div className="card-header">Your Priorities</div>
            <div className="card-body">
              <ul className="list-group list-group-flush" style={{ backgroundColor: 'transparent' }}>
                {Object.entries(priorities).map(([key, value]) => {
                  const label = key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
                  return (
                    <li key={key} className="list-group-item d-flex justify-content-between align-items-center" style={{ backgroundColor: 'transparent', color: 'var(--text-light)' }}>
                      {label}
                      <span className="priority-badge">{value}/10</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
        
        <div className="col-md-9">
          <div className="card mb-4">
            <div className="card-header">Configuration Summary</div>
            <div className="card-body">
              <div className="budget-info">
                <div className="budget-item budget-total">
                  <div>Total Budget</div>
                  <div className="budget-value">₹{currentConfig.budget?.toLocaleString()}</div>
                </div>
                <div className="budget-item budget-used">
                  <div>Total Cost</div>
                  <div className="budget-value">₹{currentConfig.total_cost?.toLocaleString()}</div>
                </div>
                <div className="budget-item budget-remaining">
                  <div>Remaining</div>
                  <div className="budget-value">₹{currentConfig.remaining_budget?.toLocaleString()}</div>
                </div>
              </div>
              
              <div className="row mt-4">
                <div className="col-md-6">
                  <div className="chart-container">
                    <h5 className="chart-title">Component Distribution</h5>
                    <Pie data={prepareComponentDistributionData(currentConfig)} options={pieChartOptions} />
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="chart-container">
                    <h5 className="chart-title">Cost Breakdown</h5>
                    <Bar data={prepareCostBreakdownData(currentConfig)} options={barChartOptions} />
                  </div>
                </div>
              </div>
              
              <div className="d-grid gap-2 mt-4">
                <button 
                  className="btn btn-primary" 
                  onClick={() => handleDownloadReport(currentConfig.config_index)}
                >
                  <FontAwesomeIcon icon={faDownload} className="me-2" />
                  Download Detailed Report
                </button>
              </div>
            </div>
          </div>
          
          <div className="card mb-4">
            <div className="card-header">Configuration Summary</div>
            <div className="card-body">
              <div className="alert alert-info" style={{ backgroundColor: 'rgba(84, 160, 255, 0.1)', color: 'var(--text-light)', border: '1px solid rgba(84, 160, 255, 0.3)' }}>
                <h5><FontAwesomeIcon icon={faInfoCircle} className="me-2" />Optimization Summary</h5>
                <p>This configuration has been optimized based on your priorities and budget constraints. It provides an optimal balance between {Object.entries(priorities).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([key]) => key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')).join(' and ')}.</p>
                <p>The total cost of ₹{currentConfig.total_cost?.toLocaleString()} represents {Math.round((currentConfig.total_cost / currentConfig.budget) * 100)}% of your budget, with ₹{currentConfig.remaining_budget?.toLocaleString()} remaining for future upgrades.</p>
              </div>
            </div>
          </div>

          {currentConfig.room_allocations && (
            <div className="card mb-4">
              <div className="card-header">Room Allocations</div>
              <div className="card-body">
                <div className="row">
                  {currentConfig.room_allocations.map((room, roomIndex) => (
                    <div key={roomIndex} className="col-md-6 mb-3">
                      <div className="room-card">
                        <h5 className="neon-text">{room.name}</h5>
                        <p><strong>Components:</strong> {room.components?.length || 0}</p>
                        <ul className="list-group list-group-flush" style={{ backgroundColor: 'transparent' }}>
                          {room.components?.map((component, compIndex) => (
                            <li 
                              key={compIndex} 
                              className="list-group-item d-flex justify-content-between align-items-center" 
                              style={{ backgroundColor: 'rgba(30, 30, 50, 0.5)', color: 'var(--text-light)', marginBottom: '5px', borderRadius: '8px' }}
                            >
                              <div>
                                <div>{component.Name}</div>
                                <small style={{ color: 'var(--text-muted)' }}>{component.Category}</small>
                                {component.Efficiency && component.Reliability && (
                                  <div className="mt-1">
                                    <small className="me-2" style={{ color: 'var(--success)' }}>
                                      <FontAwesomeIcon icon={faBolt} className="me-1" />
                                      Efficiency: {component.Efficiency}/10
                                    </small>
                                    <small style={{ color: 'var(--warning)' }}>
                                      <FontAwesomeIcon icon={faShieldAlt} className="me-1" />
                                      Reliability: {component.Reliability}/10
                                    </small>
                                  </div>
                                )}
                              </div>
                              <span>₹{component.Price_INR?.toLocaleString()}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;