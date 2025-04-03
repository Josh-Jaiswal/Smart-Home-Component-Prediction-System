# Smart Home Configuration Planner

This project is a Smart Home Configuration Planner that helps users design their ideal smart home based on preferences and budget. It uses a Flask backend with machine learning models to generate optimized smart home configurations, and a React frontend for an interactive user experience.

## Project Structure

- `app.py` - Original Flask application with Jinja2 templates
- `app_react.py` - Modified Flask application that serves the React frontend
- `smart_home_predictor.py` - Core logic for generating smart home configurations
- `src/` - React frontend source code

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Install Python dependencies:
   ```
   pip install flask matplotlib numpy pandas scikit-learn xgboost pulp joblib seaborn
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Build the React frontend:
   ```
   npm run build
   ```

## Running the Application

### Development Mode

1. Start the Flask backend:
   ```
   python app_react.py
   ```

2. In a separate terminal, start the React development server:
   ```
   npm start
   ```

3. Access the application at http://localhost:3000

### Production Mode

1. Build the React frontend:
   ```
   npm run build
   ```

2. Run the Flask application with React integration:
   ```
   python app_react.py
   ```

3. Access the application at http://localhost:5000

## Features

- Interactive form for specifying budget, number of rooms, and priorities
- Multiple smart home configurations generated based on user preferences
- Visualization of component distribution and cost breakdown
- Room allocation details for each configuration
- Downloadable detailed reports

## Technology Stack

- **Backend**: Flask, Python, scikit-learn, XGBoost, PuLP
- **Frontend**: React, Chart.js, Bootstrap, FontAwesome
- **Data Visualization**: Chart.js, Matplotlib