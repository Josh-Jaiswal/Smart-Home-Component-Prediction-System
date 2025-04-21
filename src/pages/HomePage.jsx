import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRupeeSign } from '@fortawesome/free-solid-svg-icons';

import { 
  faCog, 
  faBolt, 
  faShieldAlt, 
  faHandPointer, 
  faExpandArrowsAlt, 
  faMagic, 
  faHome, 
  faLightbulb, 
  faWifi, 
  faRobot,
  faBuilding,
  faMapMarkerAlt,
  faInfoCircle,
  faChevronLeft,
  faChevronRight,
  faCheck,
  faMusic,
  faThermometerHalf
} from '@fortawesome/free-solid-svg-icons';
import axios from 'axios';
import SpaceWaveBackground from '../components/SpaceWaveBackground';
import '../styles/HomePage.css';

const HomePage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    // Home Information (Step 1)
    home_type: 'apartment',
    num_rooms: 3,
    location: '',
    
    // Preferences and Budget (Step 2)
    budget: 50000,
    energy_efficiency: 5,
    security: 5,
    ease_of_use: 5,
    scalability: 5,
    
    // Device Priorities (Step 3)
    security_devices: false,
    lighting: false,
    climate_control: false,
    entertainment: false
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 3;
  const [activeSection, setActiveSection] = useState('step1');

  // Features for the feature cards
  const features = [
    { icon: faHome, title: "Smart Living", description: "Control your entire home with a single app" },
    { icon: faBolt, title: "Energy Efficient", description: "Reduce energy consumption and save money" },
    { icon: faShieldAlt, title: "Enhanced Security", description: "Keep your home safe with advanced security systems" },
    { icon: faWifi, title: "Connected Devices", description: "Seamlessly connect all your smart devices" },
    { icon: faLightbulb, title: "Intelligent Automation", description: "Automate routines based on your preferences and habits" },
    { icon: faRobot, title: "AI Integration", description: "Leverage AI for smarter, more responsive home systems" }
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setFormData({
      ...formData,
      [name]: checked
    });
  };

  const updateSliderValue = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseInt(value)
    });
  };

  const nextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
      setActiveSection(`step${currentStep + 1}`);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      setActiveSection(`step${currentStep - 1}`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // Create form data for submission
      const submitData = new FormData();
      Object.keys(formData).forEach(key => {
        submitData.append(key, formData[key]);
      });
      
      // Submit to Flask backend - update endpoint to match ResultsPage.jsx
      const response = await axios.post('/api/submit', submitData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        responseType: 'json'
      });
      
      // Store the response data in sessionStorage
      sessionStorage.setItem('configData', JSON.stringify(response.data));
      
      // Navigate to results page
      navigate('/results');
    } catch (error) {
      console.error('Error submitting form:', error);
      alert('An error occurred while generating your smart home configuration. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to render the correct step button
  const renderStepButton = () => {
    if (currentStep === totalSteps) {
      return (
        <button 
          type="submit" 
          className="btn btn-primary btn-lg submit-btn"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              Generating...
            </>
          ) : (
            <>
              <FontAwesomeIcon icon={faMagic} /> Generate Configuration
            </>
          )}
        </button>
      );
    } else {
      return (
        <button 
          type="button" 
          className="btn btn-primary btn-lg next-btn"
          onClick={nextStep}
        >
          Next <FontAwesomeIcon icon={faChevronRight} />
        </button>
      );
    }
  };

  // Helper function to determine if a step is active, completed, or upcoming
  const getStepStatus = (step) => {
    if (currentStep === step) return 'active';
    if (currentStep > step) return 'completed';
    return 'upcoming';
  };


  return (
    <div className="homepage">
      <SpaceWaveBackground />
      
      {/* Hero Section with Animated Text */}
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">Design Your <span className="text-gradient">Smart Home</span></h1>
          <p className="hero-subtitle">Create a futuristic living space with intelligent technology tailored to your lifestyle</p>
          <div className="hero-cta">
            <a href="#configurator" className="btn btn-primary btn-lg pulse-btn">
              <FontAwesomeIcon icon={faMagic} /> Configure Now
            </a>
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="container feature-section">
        <h2 className="section-title">Why Choose Smart Home Technology?</h2>
        <p className="section-subtitle">Discover the benefits of transforming your home with intelligent technology</p>
        <div className="feature-grid">
          {features.map((feature, index) => (
            <div className="feature-card" key={index}>
              <div className="feature-icon">
                <FontAwesomeIcon icon={feature.icon} />
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Configuration Form */}
      <div id="configurator" className="container configurator-section">
        <div className="section-header">
          <h2 className="section-title"><FontAwesomeIcon icon={faCog} /> Design Your Smart Home</h2>
          <p className="section-subtitle">Customize your smart home configuration based on your preferences and budget</p>
        </div>
        
        {/* Step Progress Indicator */}
        <div className="step-progress">
          <div className={`step-item ${getStepStatus(1)}`}>
            <div className="step-circle">
              {currentStep > 1 ? <FontAwesomeIcon icon={faCheck} /> : 1}
            </div>
            <div className="step-label">Home Information</div>
          </div>
          <div className="step-connector"></div>
          <div className={`step-item ${getStepStatus(2)}`}>
            <div className="step-circle">
              {currentStep > 2 ? <FontAwesomeIcon icon={faCheck} /> : 2}
            </div>
            <div className="step-label">Preferences & Budget</div>
          </div>
          <div className="step-connector"></div>
          <div className={`step-item ${getStepStatus(3)}`}>
            <div className="step-circle">3</div>
            <div className="step-label">Device Priorities</div>
          </div>
        </div>
        
        <form onSubmit={handleSubmit} className="config-form">
          {/* Step 1: Home Information */}
          <div className={`form-step ${currentStep === 1 ? 'active' : ''}`}>
            <div className="step-container">
              <h2 className="step-title">Home Information</h2>

              <div className="form-group">
                <label htmlFor="home_type">Home Type</label>
                <select 
                  className="form-select"
                  id="home_type" 
                  name="home_type" 
                  value={formData.home_type} 
                  onChange={handleInputChange}
                >
                  <option value="apartment">Apartment</option>
                  <option value="house">House</option>
                  <option value="condo">Condominium</option>
                  <option value="villa">Villa</option>
                  <option value="studio">Studio</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="num_rooms">Number of Rooms</label>
                <input 
                  className="form-control"
                  type="number" 
                  id="num_rooms" 
                  name="num_rooms" 
                  value={formData.num_rooms} 
                  onChange={handleInputChange} 
                  min="1" 
                  max="15" 
                  required 
                />
              </div>

              <div className="form-group">
                <label htmlFor="location">Location</label>
                <input 
                  className="form-control"
                  type="text" 
                  id="location" 
                  name="location" 
                  value={formData.location} 
                  onChange={handleInputChange} 
                  placeholder="City, State or Country" 
                />
                <small className="helper-text">This helps us optimize for local climate conditions</small>
              </div>

              <button 
                type="button" 
                className="next-button"
                onClick={nextStep}
              >
                Next <FontAwesomeIcon icon={faChevronRight} />
              </button>
            </div>
          </div>
          {/* Step 2: Preferences and Budget */}
          <div className={`form-step ${currentStep === 2 ? 'active' : ''}`}>
            <div className="card glass-card bg-transparent border-0">
              <div className="card-body">
              <h3 className="step-title"><FontAwesomeIcon icon={faRupeeSign} /> Preferences & Budget</h3>

                
                <div className="mb-4">
                  <label htmlFor="budget" className="form-label">Budget (₹)</label>
                  <div className="input-group">
                    <span className="input-group-text">₹</span>
                    <input 
                      type="number" 
                      className="form-control" 
                      id="budget" 
                      name="budget" 
                      value={formData.budget} 
                      onChange={handleInputChange} 
                      min="10000" 
                      step="1000" 
                      required 
                    />
                  </div>
                  <small className="form-text text-muted">Recommended minimum: ₹30,000 for basic setup</small>
                </div>
                
                <h4 className="priority-title"><FontAwesomeIcon icon={faExpandArrowsAlt} /> Set Your Priorities</h4>
                <p className="text-center mb-4">Drag the sliders to indicate how important each factor is to you (1-10)</p>
                
                <div className="slider-grid">
                  <div className="slider-container">
                    <label htmlFor="energy_efficiency" className="form-label">
                      <FontAwesomeIcon icon={faBolt} className="priority-icon" /> Energy Efficiency
                    </label>
                    <div className="position-relative">
                      <input 
                        type="range" 
                        className="form-range custom-range" 
                        id="energy_efficiency" 
                        name="energy_efficiency" 
                        min="1" 
                        max="10" 
                        value={formData.energy_efficiency} 
                        onChange={updateSliderValue} 
                      />
                      <span 
                        className="slider-value" 
                        style={{ left: `${((formData.energy_efficiency - 1) / 9) * 100}%` }}
                      >
                        {formData.energy_efficiency}
                      </span>
                    </div>
                    <small className="slider-description">How important is energy saving and efficiency to you?</small>
                  </div>
                  
                  <div className="slider-container">
                    <label htmlFor="security" className="form-label">
                      <FontAwesomeIcon icon={faShieldAlt} className="priority-icon" /> Security
                    </label>
                    <div className="position-relative">
                      <input 
                        type="range" 
                        className="form-range custom-range" 
                        id="security" 
                        name="security" 
                        min="1" 
                        max="10" 
                        value={formData.security} 
                        onChange={updateSliderValue} 
                      />
                      <span 
                        className="slider-value" 
                        style={{ left: `${((formData.security - 1) / 9) * 100}%` }}
                      >
                        {formData.security}
                      </span>
                    </div>
                    <small className="slider-description">How important is home security and monitoring to you?</small>
                  </div>
                  
                  <div className="slider-container">
                    <label htmlFor="ease_of_use" className="form-label">
                      <FontAwesomeIcon icon={faHandPointer} className="priority-icon" /> Ease of Use
                    </label>
                    <div className="position-relative">
                      <input 
                        type="range" 
                        className="form-range custom-range" 
                        id="ease_of_use" 
                        name="ease_of_use" 
                        min="1" 
                        max="10" 
                        value={formData.ease_of_use} 
                        onChange={updateSliderValue} 
                      />
                      <span 
                        className="slider-value" 
                        style={{ left: `${((formData.ease_of_use - 1) / 9) * 100}%` }}
                      >
                        {formData.ease_of_use}
                      </span>
                    </div>
                    <small className="slider-description">How important is user-friendly operation to you?</small>
                  </div>
                  
                  <div className="slider-container">
                    <label htmlFor="scalability" className="form-label">
                      <FontAwesomeIcon icon={faExpandArrowsAlt} className="priority-icon" /> Scalability
                    </label>
                    <div className="position-relative">
                      <input 
                        type="range" 
                        className="form-range custom-range" 
                        id="scalability" 
                        name="scalability" 
                        min="1" 
                        max="10" 
                        value={formData.scalability} 
                        onChange={updateSliderValue} 
                      />
                      <span 
                        className="slider-value" 
                        style={{ left: `${((formData.scalability - 1) / 9) * 100}%` }}
                      >
                        {formData.scalability}
                      </span>
                    </div>
                    <small className="slider-description">How important is future expansion and compatibility to you?</small>
                  </div>
                </div>
                
                <div className="form-actions">
                  <button 
                    type="button" 
                    className="btn btn-outline-secondary back-btn"
                    onClick={prevStep}
                  >
                    <FontAwesomeIcon icon={faChevronLeft} /> Back
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-primary next-btn"
                    onClick={nextStep}
                  >
                    Next <FontAwesomeIcon icon={faChevronRight} />
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Step 3: Device Priorities */}
          <div className={`form-step ${currentStep === 3 ? 'active' : ''}`}>
            <div className="card glass-card bg-transparent border-0">
              <div className="card-body">
                <h3 className="step-title"><FontAwesomeIcon icon={faWifi} /> Device Priorities</h3>
                <p className="text-center mb-4">Select the smart home features you're most interested in</p>
                
                <div className="device-priorities">
                  <div className="device-option">
                    <div className="form-check form-switch">
                      <input 
                        className="form-check-input" 
                        type="checkbox" 
                        id="security_devices" 
                        name="security_devices" 
                        checked={formData.security_devices} 
                        onChange={handleCheckboxChange}
                      />
                      <label className="form-check-label" htmlFor="security_devices">
                        <FontAwesomeIcon icon={faShieldAlt} className="me-2" />
                        Security Devices
                      </label>
                    </div>
                    <small className="device-description">Cameras, motion sensors, smart locks, and alarm systems</small>
                  </div>
                  
                  <div className="device-option">
                    <div className="form-check form-switch">
                      <input 
                        className="form-check-input" 
                        type="checkbox" 
                        id="lighting" 
                        name="lighting" 
                        checked={formData.lighting} 
                        onChange={handleCheckboxChange}
                      />
                      <label className="form-check-label" htmlFor="lighting">
                        <FontAwesomeIcon icon={faLightbulb} className="me-2" />
                        Smart Lighting
                      </label>
                    </div>
                    <small className="device-description">Smart bulbs, light strips, motion-activated lighting</small>
                  </div>
                  
                  <div className="device-option">
                    <div className="form-check form-switch">
                      <input 
                        className="form-check-input" 
                        type="checkbox" 
                        id="climate_control" 
                        name="climate_control" 
                        checked={formData.climate_control} 
                        onChange={handleCheckboxChange}
                      />
                      <label className="form-check-label" htmlFor="climate_control">
                        <FontAwesomeIcon icon={faThermometerHalf} className="me-2" />
                        Climate Control
                      </label>
                    </div>
                    <small className="device-description">Smart thermostats, AC controllers, air quality monitors</small>
                  </div>
                  
                  <div className="device-option">
                    <div className="form-check form-switch">
                      <input 
                        className="form-check-input" 
                        type="checkbox" 
                        id="entertainment" 
                        name="entertainment" 
                        checked={formData.entertainment} 
                        onChange={handleCheckboxChange}
                      />
                      <label className="form-check-label" htmlFor="entertainment">
                        <FontAwesomeIcon icon={faMusic} className="me-2" />
                        Entertainment Systems
                      </label>
                    </div>
                    <small className="device-description">Smart speakers, multi-room audio, voice assistants</small>
                  </div>
                </div>
                
                <div className="form-actions">
                  <button 
                    type="button" 
                    className="btn btn-outline-secondary back-btn"
                    onClick={prevStep}
                  >
                    <FontAwesomeIcon icon={faChevronLeft} /> Back
                  </button>
                  <button 
                    type="submit" 
                    className="btn btn-primary btn-lg submit-btn"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Generating...
                      </>
                    ) : (
                      <>
                        <FontAwesomeIcon icon={faMagic} /> Generate Configuration
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </form>
      </div>
      
      {/* How It Works Section */}
      <div className="container how-it-works-section">
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">Follow these simple steps to get your personalized smart home configuration</p>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Set Your Budget</h3>
              <p>Define your budget and the number of rooms in your home</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Define Priorities</h3>
              <p>Tell us what matters most to you in your smart home setup</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Get Recommendations</h3>
              <p>Receive personalized smart home configurations based on your inputs</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h3>Review Options</h3>
              <p>Explore the recommended devices and systems for your home</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">5</div>
            <div className="step-content">
              <h3>Customize Setup</h3>
              <p>Fine-tune your configuration to perfectly match your needs</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">6</div>
            <div className="step-content">
              <h3>Implement Solution</h3>
              <p>Follow our detailed guide to set up your smart home system</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>© {new Date().getFullYear()} Smart Home Configuration Planner. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;