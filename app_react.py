from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from smart_home_predictor import SmartHomePredictor
from flask_cors import CORS

app = Flask(__name__, static_folder='static/react-build', static_url_path='/static/react-build')
CORS(app)  # Enable CORS for all routes

# Initialize the predictor
predictor = SmartHomePredictor()

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API endpoint for form submission - support both form data and JSON
@app.route('/api/submit', methods=['POST'])
def submit():
    try:
        # Check content type to determine how to process the request
        content_type = request.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            # Process JSON data
            data = request.json
            budget = float(data.get('budget', 100000))
            num_rooms = int(data.get('num_rooms', 5))
            priorities = {
                "energy_efficiency": int(data.get('energy_efficiency', 5)),
                "security": int(data.get('security', 5)),
                "ease_of_use": int(data.get('ease_of_use', 5)),
                "scalability": int(data.get('scalability', 5)),
                "lighting": bool(data.get('lighting', False)),
                "security_devices":  bool(data.get('security_devices', False)),
                "climate_control":   bool(data.get('climate_control', False)),
                "energy_management": bool(data.get('energy_management', False))
            }
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            # Process form data
            try:
                budget = float(request.form.get('budget', 100000))
            except ValueError:
                budget = 100000
            
            try:
                num_rooms = int(request.form.get('num_rooms', 5))
            except ValueError:
                num_rooms = 5
            
            # Get priorities from sliders (1-10)
            priorities = {
                "energy_efficiency": int(request.form.get('energy_efficiency', 5)),
                "security": int(request.form.get('security', 5)),
                "ease_of_use": int(request.form.get('ease_of_use', 5)),
                "scalability": int(request.form.get('scalability', 5))
            }
        else:
            # Default fallback - try to get data from form or raw data
            try:
                if request.form:
                    # Form data is available
                    budget = float(request.form.get('budget', 100000))
                    num_rooms = int(request.form.get('num_rooms', 5))
                else:
                    # Try to parse as JSON
                    data = request.get_json(silent=True) or {}
                    budget = float(data.get('budget', 100000))
                    num_rooms = int(data.get('num_rooms', 5))
                
                priorities = {
                    "energy_efficiency": int(request.form.get('energy_efficiency', 5) if request.form else data.get('energy_efficiency', 5)),
                    "security": int(request.form.get('security', 5) if request.form else data.get('security', 5)),
                    "ease_of_use": int(request.form.get('ease_of_use', 5) if request.form else data.get('ease_of_use', 5)),
                    "scalability": int(request.form.get('scalability', 5) if request.form else data.get('scalability', 5))
                }
            except Exception as e:
                print(f"Error parsing request data: {e}")
                # Use default values
                budget = 100000
                num_rooms = 5
                priorities = {
                    "energy_efficiency": 5,
                    "security": 5,
                    "ease_of_use": 5,
                    "scalability": 5
                }
        
        # Log received data for debugging
        print(f"Received request - Budget: {budget}, Rooms: {num_rooms}, Priorities: {priorities}")
        
        # Generate multiple configurations
        configurations = predictor.generate_multiple_configurations(num_rooms, budget, priorities)
        
        # Process configurations for template rendering
        for i, config in enumerate(configurations):
            # Add config_index to each configuration
            config['config_index'] = i
            
            # Add budget and calculate remaining budget
            config['budget'] = budget
            config['remaining_budget'] = budget - config['total_cost']
            
            # Calculate category counts and costs for each configuration
            category_counts = {}
            category_costs = {}
            
            for comp in config['optimization_result']['selected_components']:
                category = comp['Category']
                price = comp['Price_INR']
                
                # Update category counts
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1
                    
                # Update category costs - simplified to work better with charts
                if category in category_costs:
                    category_costs[category] += price
                else:
                    category_costs[category] = price
            
            config['category_counts'] = category_counts
            config['category_costs'] = category_costs
            
            # Add index to room allocations for proper tab identification
            for j, room in enumerate(config.get('room_allocations', [])):
                room['index'] = j
                room['id'] = f"{room['name'].replace(' ', '-').lower()}-{i}"
        
        # Save configurations to a temporary JSON file for download purposes
        config_path = os.path.join('static', 'temp_config.json')
        with open(config_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_config = json.dumps(configurations, default=lambda x: float(x) if isinstance(x, np.number) else x)
            f.write(json_config)
        
        # Add config_index to each configuration
        for i, config in enumerate(configurations):
            config['config_index'] = i
            
            # Add room_index to each room
            for j, room in enumerate(config.get('room_allocations', [])):
                room['room_index'] = j
        
        # Generate visualizations for each configuration
        all_visualizations = []
        for config in configurations:
            visualizations = {}
            
            # Component distribution visualization as an image
            fig = predictor.visualize_component_distribution(config)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            visualizations['component_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # Cost breakdown visualization as an image (if needed elsewhere)
            # Note: In the merged template, the cost breakdown chart uses Chart.js
            fig = predictor.visualize_cost_breakdown(config)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            visualizations['cost_breakdown'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # Room allocation visualization
            fig = predictor.visualize_room_allocation(config)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            visualizations['room_allocation'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            all_visualizations.append(visualizations)
        
        # Generate HTML report for the first configuration (default)
        report_html = predictor.generate_report(configurations[0])
        
        # Return JSON response for React frontend
        return jsonify({
            'success': True,
            'configurations': configurations,
            'all_visualizations': all_visualizations,
            'report_html': report_html,
            'priorities': priorities
        })
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to download a specific report by configuration index
@app.route('/api/download_report/<int:config_index>')
def download_report(config_index=0):
    # Load the saved configurations
    config_path = os.path.join('static', 'temp_config.json')
    with open(config_path, 'r') as f:
        configurations = json.loads(f.read())
    
    # Get the selected configuration; default to index 0 if out of range
    configuration = configurations[config_index] if config_index < len(configurations) else configurations[0]
    
    # Generate the report HTML file at a static location
    report_path = os.path.join('static', f'smart_home_report_{config_index}.html')
    predictor.generate_report(configuration, report_path)
    
    return redirect(url_for('static', filename=f'smart_home_report_{config_index}.html'))

# Default route for report download (defaults to first configuration)
@app.route('/api/download_report')
def download_report_default():
    return redirect(url_for('download_report', config_index=0))

# Add this new endpoint for chart generation
@app.route('/api/charts/<int:config_index>', methods=['GET'])
def generate_charts(config_index):
    # Get the stored configurations
    if not hasattr(app, 'configurations') or not app.configurations:
        return jsonify({'error': 'No configurations available'}), 404
    
    if config_index >= len(app.configurations):
        return jsonify({'error': 'Configuration index out of range'}), 404
    
    config = app.configurations[config_index]
    
    # Create pie chart for category distribution
    fig_pie, ax_pie = plt.subplots(figsize=(8, 6), facecolor='none')
    ax_pie.set_facecolor('none')
    
    # Set text color to white for better visibility on dark background
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    # Extract category counts
    category_counts = {}
    for room in config.get('room_allocations', []):
        for component in room.get('components', []):
            category = component.get('Category', 'Other')
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
    
    # Create pie chart
    categories = list(category_counts.keys())
    values = list(category_counts.values())
    
    # Use theme colors
    colors = ['#6c63ff', '#9d50bb', '#5046e5', '#8a7fff', '#4ecca3', '#ffb142']
    
    # Remove grid and axis for pie chart
    ax_pie.axis('off')  # Turn off the axis completely
    
    wedges, texts, autotexts = ax_pie.pie(
        values, 
        labels=categories, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors[:len(categories)],
        textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Enhance the appearance of percentage labels
    plt.setp(autotexts, size=10, weight="bold", color="white")
    
    ax_pie.set_title('Category Distribution', color='white', fontsize=14)
    
    # Add a legend outside the pie chart for better readability
    ax_pie.legend(
        wedges, 
        categories,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False,
        labelcolor='white'
    )
    
    # Save to base64 string
    pie_img = BytesIO()
    fig_pie.savefig(pie_img, format='png', transparent=True, bbox_inches='tight')
    pie_img.seek(0)
    pie_base64 = base64.b64encode(pie_img.getvalue()).decode('utf-8')
    plt.close(fig_pie)
    
    # Create bar chart for cost breakdown
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6), facecolor='none')
    ax_bar.set_facecolor('none')
    
    # Extract cost data
    category_costs = {}
    for room in config.get('room_allocations', []):
        for component in room.get('components', []):
            category = component.get('Category', 'Other')
            price = component.get('Price_INR', 0)
            if category in category_costs:
                category_costs[category] += price
            else:
                category_costs[category] = price
    
    # If no costs found, use energy estimates as fallback
    if not category_costs and 'energy_estimates' in config and 'by_category' in config['energy_estimates']:
        costs = config['energy_estimates']['by_category']
        categories = list(costs.keys())
        values = [costs[cat]['monthly_cost'] for cat in categories]
    else:
        categories = list(category_costs.keys())
        values = list(category_costs.values())
    
    # Ensure we have data to display
    if not categories or not values:
        categories = ['No Data']
        values = [0]
    
    # Create bar chart with improved styling
    bars = ax_bar.bar(
        categories, 
        values, 
        color=['#7b61ff', '#ff61a6', '#61ffd6', '#ffe761'][:len(categories)],
        width=0.6
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero values
            ax_bar.annotate(
                f'₹{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', 
                va='bottom',
                color='white', 
                fontsize=10,
                fontweight='bold'
            )
    
    # Improve axis styling
    ax_bar.set_ylabel('Cost (₹)', color='white', fontsize=12)
    ax_bar.set_title('Cost Breakdown', color='white', fontsize=14)
    
    # Add grid lines for better readability but make them subtle
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Format y-axis with commas for thousands
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    
    # Ensure y-axis starts at zero and has some headroom
    if max(values) > 0:
        ax_bar.set_ylim(0, max(values) * 1.15)  # Add 15% headroom
    
    # Rotate x-axis labels for better readability if needed
    if len(max(categories, key=len)) > 10:
        plt.xticks(rotation=45, ha='right')
    
    ax_bar.tick_params(axis='x', colors='white')
    ax_bar.tick_params(axis='y', colors='white')
    
    plt.tight_layout()
    ax_bar.set_ylabel('Cost (₹)', color='white')
    
    # Add cost labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'₹{int(height):,}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom',
                      color='white')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Save to base64 string
    bar_img = BytesIO()
    fig_bar.savefig(bar_img, format='png', transparent=True, bbox_inches='tight')
    bar_img.seek(0)
    bar_base64 = base64.b64encode(bar_img.getvalue()).decode('utf-8')
    plt.close(fig_bar)
    
    return jsonify({
        'pie_chart': f'data:image/png;base64,{pie_base64}',
        'bar_chart': f'data:image/png;base64,{bar_base64}'
    })

@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        # Example parameters for generating configurations
        num_rooms = 5
        budget = 100000
        priorities = {
            "energy_efficiency": 5,
            "security": 5,
            "ease_of_use": 5,
            "scalability": 5,
            "lighting": True,
            "security_devices": True,
            "climate_control": True,
            "energy_management": True
        }
        
        # Generate configurations using the predictor
        configurations = predictor.generate_multiple_configurations(num_rooms, budget, priorities)
        
        # Return configurations as JSON
        return jsonify({'success': True, 'configurations': configurations})
    except Exception as e:
        print(f"Error generating results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/react-build', exist_ok=True)
    app.run(debug=True, port=5000)