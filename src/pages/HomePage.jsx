import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCog, faBolt, faShieldAlt, faHandPointer, faExpandArrowsAlt, faMagic } from '@fortawesome/free-solid-svg-icons';
import axios from 'axios';
import SpaceWaveBackground from '../components/SpaceWaveBackground';

const HomePage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    budget: 100000,
    num_rooms: 5,
    energy_efficiency: 5,
    security: 5,
    ease_of_use: 5,
    scalability: 5
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const updateSliderValue = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      // Create form data for submission
      const submitData = new FormData();
      Object.keys(formData).forEach(key => {
        submitData.append(key, formData[key]);
      });
      
      // Submit to Flask backend
      const response = await axios.post('/submit', submitData, {
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
    }
  };

  return (
    <div className="homepage">
      <SpaceWaveBackground />
      <div className="container">
        <div className="hero-section">
          <h1>Smart Home Configuration Planner</h1>
          <p className="subtitle">Transform Your Living Space into an Intelligent Ecosystem — Where Your Preferences Shape Tomorrow's Home</p>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="card">
            <div className="card-header">
              <FontAwesomeIcon icon={faCog} /> Smart Home Configuration System
            </div>
            <div className="card-body">
              <div className="row mb-4">
                <div className="col-md-6">
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
                </div>
                <div className="col-md-6">
                  <label htmlFor="num_rooms" className="form-label">Number of Rooms</label>
                  <input 
                    type="number" 
                    className="form-control" 
                    id="num_rooms" 
                    name="num_rooms" 
                    value={formData.num_rooms} 
                    onChange={handleInputChange} 
                    min="1" 
                    max="15" 
                    required 
                  />
                </div>
              </div>
            </div>
          </div>
          
          <div className="priority-section">
            <h3 className="priority-title"><FontAwesomeIcon icon={faExpandArrowsAlt} /> Set Your Priorities</h3>
            <p className="text-center mb-4" style={{color: 'var(--text-light)'}}>Drag the sliders to indicate how important each factor is to you (1-10)</p>
            
            <div className="slider-container">
              <label htmlFor="energy_efficiency" className="form-label">
                <FontAwesomeIcon icon={faBolt} className="priority-icon" /> Energy Efficiency
              </label>
              <div className="position-relative">
                <input 
                  type="range" 
                  className="form-range" 
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
                  className="form-range" 
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
                  className="form-range" 
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
                  className="form-range" 
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
          
          <div className="text-center mt-4">
            <button type="submit" className="btn btn-primary btn-lg">
              <FontAwesomeIcon icon={faMagic} /> Generate Smart Home Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default HomePage;