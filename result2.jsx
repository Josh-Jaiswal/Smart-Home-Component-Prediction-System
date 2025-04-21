import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faHome,
  faChartPie,
  faDownload,
  faBolt,
  faWallet,
  faDollarSign,
  faInfoCircle,
  faFilter,
  faStar,
  faStarHalf
} from '@fortawesome/free-solid-svg-icons';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';
import '../styles/global.css';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

// Reusable metric card for key stats
const MetricCard = ({ label, value, icon, trend = null }) => (
  <div
    className="metric-card d-flex flex-column align-items-center p-4 position-relative overflow-hidden transition-all"
    style={{
      flex: 1,
      backgroundColor: 'var(--card-bg)',
      borderRadius: '12px',
      marginRight: '16px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
      transition: 'transform 0.2s, box-shadow 0.2s',
      cursor: 'pointer',
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-5px)';
      e.currentTarget.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.2)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    }}
  >
    <FontAwesomeIcon icon={icon} size="2x" style={{ color: 'var(--primary-color)', marginBottom: '12px' }} />
    <h3 style={{ color: '#ffffff', margin: '0', fontSize: '28px', fontWeight: '600' }}>{value}</h3>
    <small style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{label}</small>
    {trend && (
      <div className={`position-absolute top-0 end-0 p-2 badge ${trend > 0 ? 'bg-success' : 'bg-danger'}`}>
        {trend > 0 ? '+' : ''}{trend}%
      </div>
    )}
  </div>
);

// Rating component for efficiency and reliability
const RatingDisplay = ({ value, maxValue = 10, type = "efficiency" }) => {
  const percentage = (value / maxValue) * 100;
  
  // Determine color based on percentage
  let color;
  if (percentage < 40) color = '#dc3545'; // red
  else if (percentage < 70) color = '#ffc107'; // yellow
  else color = '#28a745'; // green
  
  // Star rating display
  const fullStars = Math.floor(value);
  const hasHalfStar = value - fullStars >= 0.5;
  
  return (
    <div className="d-flex align-items-center">
      <div className="me-2">
        {[...Array(5)].map((_, i) => {
          if (i < fullStars / 2) {
            return <FontAwesomeIcon key={i} icon={faStar} style={{ color }} />;
          } else if (i === Math.floor(fullStars / 2) && hasHalfStar) {
            return <FontAwesomeIcon key={i} icon={faStarHalf} style={{ color }} />;
          } else {
            return <FontAwesomeIcon key={i} icon={faStar} style={{ color: '#6c757d33' }} />;
          }
        })}
      </div>
      <div style={{ width: '60%' }}>
        <div className="progress" style={{ height: '10px', backgroundColor: 'rgba(255,255,255,0.1)' }}>
          <div 
            className="progress-bar" 
            style={{ 
              width: `${percentage}%`,
              backgroundColor: color,
              transition: 'width 0.5s ease-in-out'
            }}
          ></div>
        </div>
      </div>
      <div className="ms-2 fw-bold" style={{ width: '40px', color }}>
        {value.toFixed(1)}/{maxValue}
      </div>
      <div className="ms-1">
        <FontAwesomeIcon 
          icon={faInfoCircle} 
          style={{ color: 'var(--text-muted)', cursor: 'pointer' }}
          data-bs-toggle="tooltip" 
          data-bs-placement="top" 
          title={type === "efficiency" ? 
            "Energy efficiency rating - higher values mean lower energy consumption" : 
            "Reliability rating - higher values indicate longer expected lifespan and fewer failures"} 
        />
      </div>
    </div>
  );
};

// Room Card Component
const RoomCard = ({ room, index }) => {
  return (
    <div className="col-md-6 col-lg-4 mb-4">
      <div className="card h-100 shadow-sm hover-card border-0">
        <div className="card-header bg-transparent border-bottom-0 d-flex justify-content-between align-items-center">
          <h3 className="h5 mb-0">{room.name || `Room ${index+1}`}</h3>
          <span className="badge bg-primary rounded-pill">
            {Array.isArray(room.components) ? room.components.length : 0} items
          </span>
        </div>
        <div className="card-body p-0">
          <ul className="list-group list-group-flush">
            {Array.isArray(room.components) && room.components.length > 0 ? (
              room.components.map((component, idx) => (
                <li className="list-group-item d-flex justify-content-between align-items-center bg-transparent border-start-0 border-end-0" key={idx}>
                  <div>
                    <span>{component.name || component.Component_Name || `Component ${idx+1}`}</span>
                    <small className="d-block text-muted">{component.category || component.Category}</small>
                  </div>
                  <span className="badge bg-primary rounded-pill">
                    ₹{(component.price || component.Price_INR || 0).toLocaleString()}
                  </span>
                </li>
              ))
            ) : (
              <li className="list-group-item bg-transparent">No devices assigned</li>
            )}
          </ul>
        </div>
        <div className="card-footer bg-transparent border-top d-flex justify-content-between align-items-center">
          <span className="text-muted">
            Total: ₹{Array.isArray(room.components) 
              ? room.components.reduce((sum, comp) => sum + (comp.price || comp.Price_INR || 0), 0).toLocaleString() 
              : 0}
          </span>
          <button className="btn btn-sm btn-outline-primary">View Details</button>
        </div>
      </div>
    </div>
  );
};

const ResultsPage = () => {
  const navigate = useNavigate();
  const [configurations, setConfigurations] = useState([]);
  const [visualizations, setVisualizations] = useState([]);
  const [activeConfig, setActiveConfig] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterCategory, setFilterCategory] = useState('All');

  // Effect for tooltip initialization
  useEffect(() => {
    // Initialize Bootstrap tooltips
    if (window.bootstrap && window.bootstrap.Tooltip) {
      const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new window.bootstrap.Tooltip(tooltipTriggerEl);
      });
    }
  }, [configurations, activeConfig]);

  useEffect(() => {
    // Add debugging to see what's in sessionStorage
    console.log("Checking sessionStorage for configData");
    const stored = sessionStorage.getItem('configData');
    console.log("Stored data:", stored);
    
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        console.log("Parsed data:", parsed);
        console.log("Configurations structure:", parsed.configurations);
        
        // Log each configuration's room_allocations to debug
        if (Array.isArray(parsed.configurations)) {
          parsed.configurations.forEach((config, idx) => {
            console.log(`Configuration ${idx} room_allocations:`, config.room_allocations);
            console.log(`Configuration ${idx} components:`, config.components);
          });
        }
        
        // Ensure configurations is always an array
        const configs = Array.isArray(parsed.configurations) ? parsed.configurations : [];
        setConfigurations(configs);
        
        // Ensure visualizations is always an array
        const visuals = Array.isArray(parsed.all_visualizations) ? parsed.all_visualizations : [];
        setVisualizations(visuals);
      } catch (err) {
        console.error("Error parsing data:", err);
        setError('Failed to load configuration data: ' + err.message);
      } finally {
        setLoading(false);
      }
    } else {
      console.log("No data found in sessionStorage");
      navigate('/');
    }
  }, [navigate]);

  const handleDownload = () => window.open(`/api/download_report/${activeConfig}`, '_blank');

  if (loading) return (
    <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
      <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
        <span className="visually-hidden">Loading...</span>
      </div>
      <div className="ms-3 fs-4">Generating your smart home configurations...</div>
    </div>
  );
  
  if (error) return (
    <div className="alert alert-danger d-flex align-items-center m-5" role="alert">
      <FontAwesomeIcon icon={faInfoCircle} className="me-2" size="lg" />
      <div>{error}</div>
    </div>
  );
  
  if (!configurations.length) return (
    <div className="alert alert-warning d-flex align-items-center m-5" role="alert">
      <FontAwesomeIcon icon={faInfoCircle} className="me-2" size="lg" />
      <div>No configurations found. <button className="btn btn-sm btn-warning ms-3" onClick={() => navigate('/')}>Return to Home</button></div>
    </div>
  );

  // Safely get the current configuration
  const current = configurations[activeConfig] || {};
  
  // Add null checks for all properties
  const total = current.total_cost || 0;
  const budget = current.budget || 100000; // Default budget if not provided
  const remaining = budget - total;
  const budgetPercentage = (total / budget) * 100;
  const monthlyCost = current.energy_estimates?.monthly_cost?.toFixed(2) || "0.00";
  
  // Ensure these objects exist before trying to use them
  const categoryCountsEntries = Object.entries(current.category_counts || {});
  const categoryCostsEntries = Object.entries(current.category_costs || {});
  const roomAllocations = Array.isArray(current.room_allocations) ? current.room_allocations : [];
  
  // Extract all components from room allocations for the component table
  const allComponents = [];
  roomAllocations.forEach(room => {
    if (Array.isArray(room.components)) {
      allComponents.push(...room.components.map(component => ({
        ...component,
        room: room.name || 'Unspecified Room'
      })));
    }
  });
  
  // Filter components if a category filter is active
  const filteredComponents = filterCategory === 'All' 
    ? allComponents 
    : allComponents.filter(comp => (comp.category || comp.Category) === filterCategory);
  
  // Get unique categories for filter dropdown
  const uniqueCategories = ['All', ...new Set(allComponents.map(comp => comp.category || comp.Category).filter(Boolean))];

  // Prepare data for pie chart with improved colors and percentages
  const pieData = {
    labels: categoryCountsEntries.map(([key]) => key),
    datasets: [{ 
      data: categoryCountsEntries.map(([_, value]) => value), 
      backgroundColor: [
        'rgba(108, 99, 255, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
      ],
      borderColor: [
        'rgba(108, 99, 255, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
      ],
      borderWidth: 2
    }]
  };

  // Pie chart options with better labels
  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: '#ffffff',
          padding: 15,
          font: {
            size: 12
          },
          generateLabels: (chart) => {
            const data = chart.data;
            const total = data.datasets[0].data.reduce((sum, val) => sum + val, 0);
            
            return data.labels.map((label, i) => {
              const value = data.datasets[0].data[i];
              const percentage = Math.round((value / total) * 100);
              
              return {
                text: `${label}: ${value} (${percentage}%)`,
                fillStyle: data.datasets[0].backgroundColor[i],
                hidden: false,
                lineWidth: 1,
                strokeStyle: data.datasets[0].borderColor[i],
                index: i
              };
            });
          }
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.label || '';
            const value = context.raw || 0;
            const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
            const percentage = Math.round((value / total) * 100);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };

  // Bar chart improved data
  const barData = {
    labels: categoryCostsEntries.map(([key]) => key),
    datasets: [{ 
      label: 'Cost (₹)', 
      data: categoryCostsEntries.map(([_, value]) => value.total_cost || 0), 
      backgroundColor: 'rgba(108, 99, 255, 0.7)',
      borderColor: 'rgba(108, 99, 255, 1)',
      borderWidth: 1
    }]
  };

  // Bar chart options with value labels
  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: '#ffffff',
          callback: function(value) {
            return '₹' + value.toLocaleString();
          }
        }
      },
      x: {
        grid: {
          display: false
        },
        ticks: {
          color: '#ffffff'
        }
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return '₹' + context.raw.toLocaleString();
          }
        }
      },
      datalabels: {
        anchor: 'end',
        align: 'top',
        formatter: (value) => '₹' + value.toLocaleString(),
        color: '#ffffff',
        font: {
          weight: 'bold'
        }
      }
    }
  };

  return (
    <div className="container py-5">
      {/* Header with subtle animation */}
      <div className="header mb-5 text-center position-relative">
        <div className="header-icon mb-3">
          <FontAwesomeIcon icon={faHome} size="3x" className="text-primary" />
        </div>
        <h1 className="display-4 mb-2">Smart Home Configuration Results</h1>
        <p className="lead text-muted">Review your personalized configurations and explore the details</p>
        <div className="position-absolute" style={{ bottom: '-15px', left: '0', right: '0' }}>
          <div style={{ height: '4px', width: '10%', backgroundColor: 'var(--primary-color)', margin: '0 auto' }}></div>
        </div>
      </div>

      {/* 1. Key Metrics with improved visuals */}
      <div className="d-flex flex-wrap gap-4 mb-5">
        <MetricCard 
          label="Total Cost" 
          value={`₹${total.toLocaleString()}`} 
          icon={faDollarSign} 
        />
        <MetricCard 
          label="Budget Remaining" 
          value={`₹${remaining.toLocaleString()}`} 
          icon={faWallet} 
          trend={remaining > 0 ? Math.round((remaining / budget) * 100) : -Math.round((Math.abs(remaining) / budget) * 100)}
        />
        <MetricCard 
          label="Monthly Energy Cost" 
          value={`₹${monthlyCost}`} 
          icon={faBolt} 
        />
      </div>

      {/* 2. Configuration Tabs & Download - Improved styling */}
      <div className="card mb-5 border-0 shadow-sm">
        <div className="card-body">
          <div className="d-flex justify-content-between align-items-center flex-wrap">
            <ul className="nav nav-tabs border-0">
              {configurations.map((config, idx) => {
                // Create unique names for each configuration
                let configName;
                if (idx === 0) configName = 'Balanced';
                else if (idx === 1) configName = 'Energy Efficient';
                else if (idx === 2) configName = 'Security Focused';
                else configName = `Configuration ${idx+1}`;
                
                return (
                  <li className="nav-item" key={idx}>
                    <button
                      className={`nav-link px-4 py-3 ${idx === activeConfig ? 'active' : ''}`}
                      onClick={() => setActiveConfig(idx)}
                      style={{
                        transition: 'all 0.3s ease',
                        borderRadius: idx === activeConfig ? '8px 8px 0 0' : '8px',
                        fontWeight: idx === activeConfig ? 'bold' : 'normal',
                      }}
                    >
                      {configName} {config.total_cost ? `(₹${Math.round(config.total_cost).toLocaleString()})` : ''}
                    </button>
                  </li>
                );
              })}
            </ul>
            <button className="btn btn-success px-4 py-2 d-flex align-items-center" onClick={handleDownload}>
              <FontAwesomeIcon icon={faDownload} className="me-2" /> Download Report
            </button>
          </div>
        </div>
      </div>

      {/* Budget progress bar - enhanced with better visuals */}
      <div className="card mb-5 border-0 shadow-sm">
        <div className="card-body">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <h5 className="mb-0">Budget Utilization</h5>
            <div className="badge bg-primary">{budgetPercentage.toFixed(1)}% Used</div>
          </div>
          <div className="progress" style={{ height: '25px', borderRadius: '12px', backgroundColor: 'rgba(255,255,255,0.1)' }}>
            <div 
              className={`progress-bar ${budgetPercentage > 90 ? 'bg-danger' : budgetPercentage > 75 ? 'bg-warning' : 'bg-success'}`}
              role="progressbar" 
              style={{ 
                width: `${Math.min(budgetPercentage, 100)}%`,
                transition: 'width 1s ease-in-out'
              }} 
              aria-valuenow={budgetPercentage} 
              aria-valuemin="0" 
              aria-valuemax="100"
            >
              ₹{total.toLocaleString()} of ₹{budget.toLocaleString()}
            </div>
          </div>
          <div className="d-flex justify-content-between mt-2">
            <small className="text-muted">₹0</small>
            <small className="text-success">₹{remaining.toLocaleString()} Remaining</small>
            <small className="text-muted">₹{budget.toLocaleString()}</small>
          </div>
        </div>
      </div>

      {/* 3. Summary & Charts - Now with better visualizations */}
      <div className="row g-4 mb-5">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-transparent border-bottom">
              <h4 className="mb-0">Category Distribution</h4>
            </div>
            <div className="card-body">
              {categoryCountsEntries.length > 0 ? (
                <div style={{ height: '300px', position: 'relative' }}>
                  <Pie data={pieData} options={pieOptions} />
                </div>
              ) : (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <p className="text-center text-muted">No category data available</p>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-transparent border-bottom">
              <h4 className="mb-0">Cost Breakdown</h4>
            </div>
            <div className="card-body">
              {categoryCostsEntries.length > 0 ? (
                <div style={{ height: '300px', position: 'relative' }}>
                  <Bar data={barData} options={barOptions} />
                </div>
              ) : (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <p className="text-center text-muted">No cost data available</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Energy Usage Summary */}
      <div className="card mb-5 border-0 shadow-sm">
        <div className="card-header bg-transparent">
          <h4 className="mb-0">Energy Usage Estimates</h4>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-4">
              <div className="d-flex align-items-center p-3 bg-dark bg-opacity-25 rounded">
                <FontAwesomeIcon icon={faBolt} size="2x" className="me-3 text-warning" />
                <div>
                  <h5 className="mb-1">Daily Consumption</h5>
                  <div className="fs-4">{current.energy_estimates?.daily_wh?.toFixed(1) || "0.0"} Wh</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="d-flex align-items-center p-3 bg-dark bg-opacity-25 rounded">
                <FontAwesomeIcon icon={faBolt} size="2x" className="me-3 text-primary" />
                <div>
                  <h5 className="mb-1">Monthly Consumption</h5>
                  <div className="fs-4">{current.energy_estimates?.monthly_kwh?.toFixed(1) || "0.0"} kWh</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="d-flex align-items-center p-3 bg-dark bg-opacity-25 rounded">
                <FontAwesomeIcon icon={faDollarSign} size="2x" className="me-3 text-success" />
                <div>
                  <h5 className="mb-1">Monthly Cost</h5>
                  <div className="fs-4">₹{monthlyCost}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Room Allocations - Improved cards with better information */}
      <div className="mb-5">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h2 className="h4 mb-0">Room Allocations</h2>
          <button className="btn btn-outline-light btn-sm">
            <FontAwesomeIcon icon={faFilter} className="me-2" /> Sort Rooms
          </button>
        </div>
        
        <div className="row">
          {roomAllocations.length > 0 ? (
            roomAllocations.map((room, index) => (
              <RoomCard key={index} room={room} index={index} />
            ))
          ) : (
            <div className="col-12">
              <div className="alert alert-info">No room allocations available</div>
            </div>
          )}
        </div>
      </div>

      {/* 5. Component Details - Enhanced table with filtering */}
      <div className="mb-5">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h2 className="h4 mb-0">Component Details</h2>
          <div className="d-flex align-items-center">
            <label className="me-2">Filter by Category:</label>
            <select 
              className="form-select form-select-sm" 
              value={filterCategory} 
              onChange={(e) => setFilterCategory(e.target.value)}
              style={{ width: 'auto' }}
            >
              {uniqueCategories.map((category, idx) => (
                <option key={idx} value={category}>{category}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="card border-0 shadow-sm">
          <div className="card-body p-0">
            <div className="table-responsive">
              {filteredComponents.length > 0 ? (
                <table className="table table-hover table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Category</th>
                      <th>Room</th>
                      <th>Price</th>
                      <th>Efficiency</th>
                      <th>Reliability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredComponents.map((component, idx) => (
                      <tr key={idx} className="align-middle">
                        <td>{component.name || component.Component_Name}</td>
                        <td>
                          <span className="badge bg-secondary">{component.category || component.Category}</span>
                        </td>
                        <td>{component.room}</td>
                        <td className="fw-bold">₹{(component.price || component.Price_INR || 0).toLocaleString()}</td>
                        <td>
                          <RatingDisplay 
                            value={component.efficiency || component.Efficiency || 0} 
                            type="efficiency" 
                          />
                        </td>
                        <td>
                          <RatingDisplay 
                            value={component.reliability || component.Reliability || 0} 
                            type="reliability" 
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr>
                      <td colSpan="3" className="text-end fw-bold">Total:</td>
                      <td className="fw-bold">₹{filteredComponents.reduce((sum, comp) => sum + (comp.price || comp.Price_INR || 0), 0).toLocaleString()}</td>
                      <td colSpan="2"></td>
                    </tr>
                  </tfoot>
                </table>
              ) : (
                <div className="text-center p-4">
                  <p className="text-muted mb-0">No components found for the selected filter.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Back button - stylish and with animation */}
      <div className="text-center mt-5">
        <button 
          className="btn btn-outline-light btn-lg px-5 py-2"
          onClick={() => navigate('/')}
          style={{
            transition: 'all 0.3s ease',
            borderRadius: '30px',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'scale(1.05)';
            e.currentTarget.style.boxShadow = '0 0 15px rgba(255, 255, 255, 0.3)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default ResultsPage;