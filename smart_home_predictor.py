# pyright: ignore
from typing import Any
from numpy._typing._array_like import NDArray
from numpy import complexfloating, floating, object_
from numpy._typing import _64Bit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from pulp import *
import random
from collections import defaultdict # type: ignore
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
import shutil, os


class SmartHomePredictor:
    def __init__(self, data_path="synthetic_smart_home_components.csv"):
        """
        Initialize the Smart Home Predictor with the dataset
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing smart home component data
        """
        # Load and preprocess the data
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
        # Initialize room types and their typical components
        self.room_types = {
            "Living Room": {
                "Lighting": 2,  # e.g., 3 light sources
                "Security": 2,  # e.g., 1 camera
                "HVAC": 1,      # e.g., 1 thermostat
                "Energy Management": 2  # e.g., 2 smart plugs
            },
            "Bedroom": {
                "Lighting": 2,
                "Security": 1,
                "HVAC": 1,
                "Energy Management": 1
            },
            "Kitchen": {
                "Lighting": 1,
                "Security": 0,
                "HVAC": 0,
                "Energy Management": 3
            },
            "Bathroom": {
                "Lighting": 1,
                "Security": 0,
                "HVAC": 1,
                "Energy Management": 0
            },
            "Hallway": {
                "Lighting": 1,
                "Security": 2,
                "HVAC": 0,
                "Energy Management": 0
            },
            "Entrance": {
                "Lighting": 1,
                "Security": 2,  # e.g., doorbell camera and smart lock
                "HVAC": 0,
                "Energy Management": 0
            }
        }
        
        # Default weights for composite score calculation
        self.default_weights = {
            "efficiency": 0.4,
            "reliability": 0.4,
            "price": 0.2
        }
        
        # Initialize ML models
        self.initialize_ml_models()
        
        # Train models with available data
        self.train_models()
    
    def initialize_ml_models(self):
        """Initialize machine learning models for component selection and optimization"""
        try:
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Check if models already exist and load them
            if os.path.exists('models/compatibility_model.pkl'):
                print("Loading pre-trained models...")
                try:
                    self.compatibility_model = joblib.load('models/compatibility_model.pkl')
                    self.performance_model = joblib.load('models/performance_model.pkl')
                    self.nn_model = joblib.load('models/nn_model.pkl')
                    self.kmeans_model = joblib.load('models/kmeans_model.pkl')
                    self.knn_model = joblib.load('models/knn_model.pkl')
                    self.decision_tree = joblib.load('models/decision_tree.pkl')
                    self.gb_model = joblib.load('models/gb_model.pkl')
                    self.scaler = joblib.load('models/scaler.pkl')
                    print("Pre-trained models loaded successfully")
                    return True
                except Exception as e:
                    print(f"Error loading models: {e}")
                    print("Will train new models")
            
            # Random Forest for component compatibility prediction
            self.compatibility_model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=8,      # Reduced from 10
                random_state=42
            )
            
            # XGBoost for performance metrics prediction
            self.performance_model = xgb.XGBRegressor(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=4,      # Reduced from 5
                random_state=42
            )
            
            # Neural Network for complex pattern recognition
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(32, 16),  
                activation='relu',
                solver='adam',
                max_iter=500,     
                random_state=42
            )
            
            # K-Means for component clustering
            self.kmeans_model = KMeans(
                n_clusters=4,     
                random_state=42
            )
            
            # KNN for similar component recommendations
            self.knn_model = KNeighborsRegressor(
                n_neighbors=3,   
                weights='distance'
            )
            
            # Decision Tree for rule extraction
            self.decision_tree = DecisionTreeClassifier(
                max_depth=4,      # Reduced from 5
                random_state=42
            )
            
            # Gradient Boosting for score refinement
            self.gb_model = GradientBoostingRegressor(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Feature scaler for preprocessing
            self.scaler = StandardScaler()
            
            # Train models with available data
            self._train_initial_models()
            
            return True
        except ImportError as e:
            print(f"Warning: ML libraries not available - {e}")
            print("Continuing with rule-based approach only")
            return False
    
    def _train_initial_models(self):
        """Train ML models with available component data"""
        try:
            # Extract features for training
            features = self._extract_component_features()
            # Scale features
            scaled_features = self._transform_with_scaler(features, self.scaler)
            # Create synthetic targets for training (since we don't have user feedback)
            compatibility_targets = self._create_synthetic_compatibility_targets()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 8, 10],
                'min_samples_split': [2, 5]
            }
            # Hyperparameter tuning for RandomForest
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(scaled_features, compatibility_targets)
            self.compatibility_model = grid_search.best_estimator_

            # For performance: use efficiency as a proxy for performance
            performance_targets = self.df['Efficiency'].values

            # 2. XGBoost for performance prediction
            self.performance_model.fit(scaled_features, performance_targets)
            # 3. Neural Network for complex patterns
            self.nn_model.fit(scaled_features, self.df['Efficiency'].values)
            # 4. K-Means for component clustering
            self.kmeans_model.fit(scaled_features)
            self.df['Cluster'] = self.kmeans_model.labels_
            # 5. KNN for similar component recommendations
            self.knn_model.fit(scaled_features, self.df['Price_INR'].values)
            # 6. Decision Tree for rule extraction
            self.decision_tree.fit(scaled_features, compatibility_targets)
            # 7. Gradient Boosting for score refinement
            self.gb_model.fit(scaled_features, self.df['Reliability'].values)

            print("ML models trained successfully")

            # New validation metrics
            cv_scores = cross_val_score(
                self.compatibility_model,
                scaled_features,
                compatibility_targets,
                cv=5,
                scoring='f1_weighted'
            )
            print(f"Compatibility Model CV F1: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

            # Feature importance visualization
            if hasattr(self.compatibility_model, 'feature_importances_'):
                importances = pd.Series(
                    self.compatibility_model.feature_importances_,
                    index=features.columns
                )
                print("Feature Importances:\n", importances.sort_values(ascending=False))

        except Exception as e:
            print(f"Error training ML models: {e}")
            print("Continuing with rule-based approach")
    
    def _extract_component_features(self):
        """Extract features from components for ML models"""
        # Select numerical features
        numerical_features = ['Price_INR', 'Efficiency', 'Reliability']
        
        # Create feature matrix
        features = self.df[numerical_features].copy()
        
        # Handle division by zero and NaN values
        epsilon = 1e-6
        features['Price_to_Efficiency'] = features['Price_INR'] / (features['Efficiency'].replace(0, epsilon) + epsilon)
        features['Reliability_per_Rupee'] = features['Reliability'] / (features['Price_INR'].replace(0, epsilon) + epsilon)
        
        # Fill any NaN results
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        # Protocol compatibility features
        if 'Compatibility' in self.df.columns:
            # Handle NaN values in Compatibility
            self.df['Compatibility'] = self.df['Compatibility'].fillna('')
            protocols = self.df['Compatibility'].str.get_dummies(',')
            features = pd.concat([features, protocols], axis=1)
        
        # Existing category encoding
        for category in self.df['Category'].unique():
            features[f'Category_{category}'] = (self.df['Category'] == category).astype(int)
        
        return features
    def _create_synthetic_compatibility_targets(self) -> np.ndarray:
        """
        Create synthetic compatibility targets for training,
        returning a 1D numpy array of ints (0, 1, or 2) with no NaNs.
        """

        # 1) Base score from Efficiency & Reliability
        base = (self.df['Efficiency']  / 10 * 0.6) + \
            (self.df['Reliability'] / 10 * 0.4)

        # 2) Fill any NaNs and clamp to [0,1]
        median = base.median()
        base = base.fillna(median)
        if 'Compatibility' in self.df.columns:
            proto_bonus = self.df['Compatibility'].apply(
                lambda x: sum(p in {'Zigbee','Z‑Wave','Matter'} for p in str(x).split(','))
            )
            base += proto_bonus * 0.2
        base = base.replace([np.inf, -np.inf], np.nan).fillna(median)
        base = base.clip(lower=0, upper=1)   # pandas.Series.clip

        # 3) Bin into three classes, returning a Series[int]
        labels_series = pd.cut(
            base,
            bins=[0, 0.5, 0.8, 1],
            include_lowest=True,
            labels=False    # ← produce integer bin indices 0,1,2
        )

        # 4) Convert to numpy array of ints
        return labels_series.to_numpy(dtype=int)
        
    def enhance_component_scores_with_ml(self, components, user_features=None, model_subset='all'):
        """
        Enhance component scores using ML models
        
        Parameters:
        -----------
        components : list
            List of components to score
        user_features : list, optional
            User features for personalization
        model_subset : str, optional
            Which subset of models to use ('all', 'minimal', 'performance')
        
        Returns:
        --------
        list
            Components with enhanced scores
        """
        try:
        # If ML models aren't initialized, return original components
            if not hasattr(self, 'compatibility_model'):
                return components
            
            comp_df = pd.DataFrame(components)
            features = self._extract_component_features_from_df(comp_df)
            scaled_features = self._transform_with_scaler(features, self.scaler)

            
            # Select which models to use based on subset parameter
            if model_subset == 'minimal':
                # Use only the most essential models
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = np.ones(len(components)) * 0.5  # Default value
                
                # Calculate ML-based score with just compatibility
                ml_scores = compatibility_scores
                
            elif model_subset == 'performance':
                # Focus on performance-related models
                compatibility_scores = np.ones(len(components)) * 0.5  # Default value
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                if len(performance_scores) > 0:
                    min_score = min(performance_scores)
                    max_score = max(performance_scores)
                    if max_score > min_score:
                        performance_scores = (performance_scores - min_score) / (max_score - min_score)
                
                # Calculate ML-based score with just performance
                ml_scores = performance_scores
                
            else:  # 'all' or any other value
                # Use all available models
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                if len(performance_scores) > 0:
                    min_score = min(performance_scores)
                    max_score = max(performance_scores)
                    if max_score > min_score:
                        performance_scores = (performance_scores - min_score) / (max_score - min_score)
                
                # Calculate ML-based score with all models
                ml_scores = 0.5 * compatibility_scores + 0.5 * performance_scores
            
            # Enhance component scores
            for i, comp in enumerate(components):
                # Blend with original score (70% original, 30% ML)
                if 'Score' in comp:
                    comp['Score'] = 0.7 * comp['Score'] + 0.3 * ml_scores[i]
                else:
                    comp['Score'] = ml_scores[i]
            
            return components
        
        except Exception as e:
            print(f"Error enhancing scores with ML: {e}")
            return components
    
    def _extract_component_features_from_df(self, df):
        numerical_features = ['Price_INR', 'Efficiency', 'Reliability']
        features = df[numerical_features].copy()
        features['Price_to_Efficiency'] = features['Price_INR'] / (features['Efficiency'] + 1e-6)
        features['Reliability_per_Rupee'] = features['Reliability'] / (features['Price_INR'] + 1e-6)
        if 'Compatibility' in df.columns:
            protocols = df['Compatibility'].str.get_dummies(',')
            features = pd.concat([features, protocols], axis=1)
        # Ensure all category columns from training are present
        for category in self.df['Category'].unique():
            col = f'Category_{category}'
            features[col] = (df['Category'] == category).astype(int) if 'Category' in df.columns else 0
        # Fill any missing columns with 0 (in case training had more protocol columns)
        for col in self._extract_component_features().columns:
            if col not in features.columns:
                features[col] = 0
        # Reorder columns to match training
        features = features[self._extract_component_features().columns]
        return features

    def preprocess_data(self):
        """
        Preprocess the data: check for missing values, duplicates, and normalize numerical features
        """
        if self.df.isnull().sum().sum() > 0:
            print(f"Warning: Found {self.df.isnull().sum().sum()} missing values")
            # Fill NaN values with appropriate defaults instead of dropping
            self.df['Price_INR'].fillna(self.df['Price_INR'].median(), inplace=True)
            self.df['Efficiency'].fillna(self.df['Efficiency'].median(), inplace=True)
            self.df['Reliability'].fillna(self.df['Reliability'].median(), inplace=True)
            # For any remaining NaNs, drop those rows
            self.df = self.df.dropna()
    
    def train_models(self):
        """Train ML models with available component data"""
        try:
            # Extract features for training
            features = self._extract_component_features()
            
            # Scale features
            scaled_features = self._transform_with_scaler(features, self.scaler)
            
            # Create synthetic targets for training (since we don't have user feedback)
            # For compatibility: create synthetic compatibility scores based on component attributes
            compatibility_targets = self._create_synthetic_compatibility_targets()
            
            # For performance: use efficiency as a proxy for performance
            performance_targets = self.df['Efficiency'].values
            
            # Train models
            # 1. Random Forest for compatibility
            self.compatibility_model.fit(scaled_features, compatibility_targets)
            
            # 2. XGBoost for performance prediction
            self.performance_model.fit(scaled_features, performance_targets)
            
            # 3. Neural Network for complex patterns
            if 'Composite_Score' in self.df.columns and not self.df['Composite_Score'].isna().any():
                nn_targets = self.df['Composite_Score'].values
            else:
                nn_targets = self.df['Efficiency'].values

            self.nn_model.fit(scaled_features, nn_targets)
            
            # 4. K-Means for component clustering
            self.kmeans_model.fit(scaled_features)
            # Add cluster labels to dataframe
            self.df['Cluster'] = self.kmeans_model.labels_
            
            # 5. KNN for similar component recommendations
            self.knn_model.fit(scaled_features, self.df['Price_INR'].values)
            
            # 6. Decision Tree for rule extraction
            self.decision_tree.fit(scaled_features, compatibility_targets)
            
            # 7. Gradient Boosting for score refinement
            self.gb_model.fit(scaled_features, self.df['Reliability'].values)
            
            print("ML models trained successfully")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            print(f"Error training ML models: {e}")
            print("Continuing with rule-based approach")
    
    def _extract_component_features(self):
        """Extract features from components for ML models"""
        # Select numerical features
        numerical_features = ['Price_INR', 'Efficiency', 'Reliability']
        
        # One-hot encode categorical features
        categorical_features = ['Category']
        
        # Create feature matrix
        features = self.df[numerical_features].copy()
        
        # Add one-hot encoded features
        for category in self.df['Category'].unique():
            features[f'Category_{category}'] = (self.df['Category'] == category).astype(int)
        
        return features  
      
    # Update model save paths to use os.path.join for cross-platform compatibility
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            joblib.dump(self.compatibility_model, os.path.join('models', 'compatibility_model.pkl'))
            joblib.dump(self.performance_model, os.path.join('models', 'performance_model.pkl'))
            joblib.dump(self.nn_model, os.path.join('models', 'nn_model.pkl'))
            joblib.dump(self.kmeans_model, os.path.join('models', 'kmeans_model.pkl'))
            joblib.dump(self.knn_model, os.path.join('models', 'knn_model.pkl'))
            joblib.dump(self.decision_tree, os.path.join('models', 'decision_tree.pkl'))
            joblib.dump(self.gb_model, os.path.join('models', 'gb_model.pkl'))
            joblib.dump(self.scaler, os.path.join('models', 'scaler.pkl'))
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def calculate_composite_score(self, weights=None):
        """
        Calculate composite score for each component based on weights and ML predictions
        
        Parameters:
        -----------
        weights : dict
            Dictionary with weights for efficiency, reliability, and price
        
        Returns:
        --------
        DataFrame with added composite score column
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                "efficiency": 0.3,
                "reliability": 0.3,
                "price": 0.4
            }
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Calculate price score (inverse of normalized price)
        price_max = self.df['Price_INR'].max()
        price_min = self.df['Price_INR'].min()
        self.df['Price_Score'] = 1 - ((self.df['Price_INR'] - price_min) / (price_max - price_min))
        
        # Calculate rule-based composite score
        self.df['Rule_Score'] = (
            weights['efficiency'] * self.df['Efficiency'] / 10 +
            weights['reliability'] * self.df['Reliability'] / 10 +
            weights['price'] * self.df['Price_Score']
        )
        
        try:
            if hasattr(self, 'compatibility_model') and hasattr(self, 'performance_model'):
                # Extract features for ML prediction
                features = self._extract_component_features()
                scaled_features = self.scaler.transform(features)
                
                # Get ML predictions
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                performance_scores = (performance_scores - performance_scores.min()) / (performance_scores.max() - performance_scores.min() + 1e-10)
                
                # Calculate ML-based score
                self.df['ML_Score'] = 0.5 * compatibility_scores + 0.5 * performance_scores
                
                # Blend rule-based and ML-based scores (70% rule, 30% ML)
                self.df['Composite_Score'] = 0.7 * self.df['Rule_Score'] + 0.3 * self.df['ML_Score']
            else:
                # If ML models aren't available, use rule-based score
                self.df['Composite_Score'] = self.df['Rule_Score']
        except Exception as e:
            print(f"Error in ML scoring: {e}")
            # Fallback to rule-based score
            self.df['Composite_Score'] = self.df['Rule_Score']
        
        # Add a small random factor to break ties (0.5% variation)
        import random
        self.df['Composite_Score'] = self.df['Composite_Score'] * [random.uniform(0.999, 1.001) for _ in range(len(self.df))]
        
        return self.df['Composite_Score']

    def adjust_weights_from_priorities(self, priorities, variant=0):
        """
        Adjust component selection weights based on user priorities
        
        Parameters:
        -----------
        priorities : dict
            Dictionary with user priorities (energy_efficiency, security, ease_of_use, scalability)
        variant : int, optional
            Variant number to generate different configurations
                
        Returns:
        --------
        dict
            Dictionary with adjusted weights for component selection
        """
        # Base weights
        weights = {
            "efficiency": 0.33,
            "reliability": 0.33,
            "price": 0.34
        }
        
        # Adjust weights based on priorities
        # Energy efficiency affects efficiency weight
        if 'energy_efficiency' in priorities:
            efficiency_factor = priorities['energy_efficiency'] / 10.0  # Normalize to 0-1
            weights["efficiency"] += efficiency_factor * 0.2
            weights["price"] -= efficiency_factor * 0.1  # Higher efficiency may cost more
        
        # Security affects reliability weight
        if 'security' in priorities:
            security_factor = priorities['security'] / 10.0  # Normalize to 0-1
            weights["reliability"] += security_factor * 0.2
            weights["price"] -= security_factor * 0.1  # Higher security may cost more
        
        # Ease of use affects both efficiency and reliability
        if 'ease_of_use' in priorities:
            ease_factor = priorities['ease_of_use'] / 10.0  # Normalize to 0-1
            weights["efficiency"] += ease_factor * 0.05
            weights["reliability"] += ease_factor * 0.05
        
        # Scalability affects price weight (more scalable = more future-proof = can invest more)
        if 'scalability' in priorities:
            scalability_factor = priorities['scalability'] / 10.0  # Normalize to 0-1
            weights["price"] -= scalability_factor * 0.1
        
        # Adjust for specific device priorities from homepage
        if 'security_devices' in priorities and priorities['security_devices']:
            if priorities['security_devices']:
                weights["reliability"] += 0.1
            else:
                weights["reliability"] -= 0.1
            
        if 'lighting' in priorities and priorities['lighting']:
            if priorities['lighting']:
                weights["efficiency"] += 0.05
            else:
                weights["efficiency"] -= 0.05
            
        if 'climate_control' in priorities and priorities['climate_control']:
            if priorities['climate_control']:
                weights["efficiency"] += 0.1
            else:
                weights["efficiency"] -= 0.1
            
        if 'entertainment' in priorities and priorities['entertainment']:
            if priorities['entertainment']:
                weights["reliability"] += 0.05
            else:
                weights["reliability"] -= 0.05
        
        # Apply variations based on variant number
        if variant == 1:  # Energy-Efficient: Strongly favor efficiency
            weights["efficiency"] += 0.15
            weights["price"] -= 0.1
            weights["reliability"] -= 0.05
        elif variant == 2:  # High-Security: Strongly favor reliability
            weights["reliability"] += 0.15
            weights["price"] -= 0.1
            weights["efficiency"] -= 0.05
        
        # Ensure weights are positive
        for key in weights:
            weights[key] = max(0.1, weights[key])
        
        # Re-normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def adjust_for_home_type_and_location(self, home_type, location, components):
        """
        Adjust component selection based on home type and location
        
        Parameters:
        -----------
        home_type : str
            Type of home (apartment, house, etc.)
        location : str
            Geographic location
        components : list
            List of selected components
            
        Returns:
        --------
        list
            Adjusted list of components
        """
        adjusted_components = components.copy()
        
        # Adjust for home type
        if home_type == 'apartment':
            # For apartments, prioritize space-efficient and less intrusive components
            for comp in adjusted_components:
                if 'Efficiency' in comp:
                    comp['Efficiency'] *= 1.1  # Increase efficiency importance
        
        elif home_type == 'house':
            # For houses, can have more comprehensive systems
            for comp in adjusted_components:
                if comp['Category'] == 'Security':
                    comp['Reliability'] *= 1.1  # Increase security reliability
        
        # Adjust for location (simplified example)
        # In a real implementation, you might use weather data, crime statistics, etc.
        if location.lower() in ['mumbai', 'delhi', 'bangalore', 'chennai']:
            # For urban areas, might prioritize security
            for comp in adjusted_components:
                if comp['Category'] == 'Security':
                    comp['Reliability'] *= 1.15
        
        return adjusted_components

    def _convert_priorities_to_features(self, priorities):
        """
        Convert user priorities to features for ML personalization
        
        Parameters:
        -----------
        priorities : dict
            Dictionary with user priorities
            
        Returns:
        --------
        list
            List of features for ML models
        """
        # Normalize priorities to 0-1 scale
        max_priority = 10.0  # Assuming priorities are on a 1-10 scale
        
        # Extract and normalize priorities
        energy_efficiency = priorities.get('energy_efficiency', 5) / max_priority
        security = priorities.get('security', 5) / max_priority
        ease_of_use = priorities.get('ease_of_use', 5) / max_priority
        scalability = priorities.get('scalability', 5) / max_priority
        
        # Create feature vector
        features = [energy_efficiency, security, ease_of_use, scalability]
        
        return features

    def _transform_with_scaler(self, features, scaler):
        """Transform features using a fitted scaler while preserving feature names"""
        try:
            # Handle NaN values before scaling
            if isinstance(features, pd.DataFrame):
                features = features.replace([np.inf, -np.inf], np.nan)
                features = features.fillna(features.median())
                
            # Check if features is a DataFrame
            if hasattr(scaler, 'n_features_in_'):
                if isinstance(features, pd.DataFrame):
                    # If it's a DataFrame, extract values for scaling
                    return scaler.transform(features)
                else:
                    # If it's already a numpy array
                    return scaler.transform(np.array(features))
            else:
                # If scaler is not fitted, fit and transform
                return scaler.fit_transform(features)
        except Exception as e:
            print(f"Error in _transform_with_scaler: {e}")
            # Return unscaled features as fallback
            if isinstance(features, pd.DataFrame):
                return features.values
            return features
    def estimate_energy_consumption(self, components, priorities=None):
        """
        Estimate energy consumption for selected components
        
        Parameters:
        -----------
        components : list
            List of selected components
        priorities : dict, optional
            User priorities to adjust estimates
            
        Returns:
        --------
        dict
            Dictionary with energy consumption estimates
        """
        # Define typical power consumption by category (in watts)
        typical_power = {
            "Lighting": 10,  # LED smart bulbs
            "Security": 5,   # Cameras, sensors
            "HVAC": 50,      # Thermostats, controllers
            "Energy Management": 2  # Smart plugs, monitors
        }
        
        # Define typical usage hours by category
        typical_usage = {
            "Lighting": 6,    # hours per day
            "Security": 24,   # hours per day (always on)
            "HVAC": 8,        # hours per day
            "Energy Management": 12  # hours per day
        }
        
        # Adjust usage based on priorities if provided
        if priorities:
            # If energy efficiency is high priority, reduce usage hours
            efficiency_factor = 1.0
            if 'energy_efficiency' in priorities:
                # Scale from 1-10 to 0.7-1.0 (higher priority = lower energy use)
                efficiency_factor = 1.0 - (priorities['energy_efficiency'] / 30.0)
                
                # Adjust typical usage hours
                for category in typical_usage:
                    typical_usage[category] *= efficiency_factor
            
            # If security is high priority, increase security device usage
            if 'security' in priorities and priorities['security'] > 7:
                # Security devices stay on 24/7 regardless
                pass
        
        # Calculate energy consumption
        daily_consumption = 0
        monthly_consumption = 0
        consumption_by_category = {}
        
        for comp in components:
            category = comp['Category']
            
            # Adjust power based on efficiency (higher efficiency = lower power)
            efficiency_factor = 1.0 - (comp['Efficiency'] / 15.0)  # Scale from 0-10 to 0.33-1.0
            adjusted_power = typical_power.get(category, 10) * efficiency_factor
            
            # Calculate daily consumption (Wh)
            daily_wh = adjusted_power * typical_usage.get(category, 12)
            daily_consumption += daily_wh
            
            # Add to category total
            if category in consumption_by_category:
                consumption_by_category[category] += daily_wh
            else:
                consumption_by_category[category] = daily_wh
        
        # Calculate monthly consumption (kWh)
        monthly_consumption = daily_consumption * 30 / 1000
        
        # Calculate estimated cost (assuming ₹7 per kWh)
        monthly_cost = monthly_consumption * 7
        
        return {
            'daily_wh': daily_consumption,
            'monthly_kwh': monthly_consumption,
            'monthly_cost': monthly_cost,
            'by_category': {
                category: {
                    'daily_wh': wh,
                    'monthly_kwh': wh * 30 / 1000,
                    'monthly_cost': wh * 30 / 1000 * 7
                } for category, wh in consumption_by_category.items()
            }
        }

    def check_component_compatibility(self, components):
        """
        Check compatibility between selected components
        
        Parameters:
        -----------
        components : list
            List of selected components
            
        Returns:
        --------
        dict
            Dictionary with compatibility information
        """
        compatibility_issues = []
        compatibility_groups = {}
        
        # Extract compatibility information
        for comp in components:
            if 'Compatibility' in comp and isinstance(comp['Compatibility'], str):
                name = comp['Component_Name']
                protocols = [p.strip() for p in comp['Compatibility'].split(',')]
                
                # Add to compatibility groups
                for protocol in protocols:
                    if protocol not in compatibility_groups:
                        compatibility_groups[protocol] = []
                    compatibility_groups[protocol].append(name)
        
        # Check if there are components that don't share any protocol
        for i, comp1 in enumerate(components):
            if 'Compatibility' not in comp1 or not isinstance(comp1['Compatibility'], str):
                continue
                
            name1 = comp1['Component_Name']
            protocols1 = [p.strip() for p in comp1['Compatibility'].split(',')]
            
            for j in range(i+1, len(components)):
                comp2 = components[j]
                
                if 'Compatibility' not in comp2 or not isinstance(comp2['Compatibility'], str):
                    continue
                    
                name2 = comp2['Component_Name']
                protocols2 = [p.strip() for p in comp2['Compatibility'].split(',')]
                
                # Check if there's any shared protocol
                shared_protocols = set(protocols1).intersection(set(protocols2))
                
                if not shared_protocols:
                    compatibility_issues.append({
                        'component1': name1,
                        'component2': name2,
                        'protocols1': protocols1,
                        'protocols2': protocols2,
                        'message': f"No shared communication protocol between {name1} and {name2}"
                    })
        
        # Identify main compatibility groups
        main_groups = {}
        for protocol, components in compatibility_groups.items():
            if len(components) >= 3:  # Consider it a main group if it has at least 3 components
                main_groups[protocol] = components
        
        return {
            'issues': compatibility_issues,
            'groups': compatibility_groups,
            'main_groups': main_groups
        }
    
    def generate_multiple_configurations(self, num_rooms, budget, priorities, home_type='apartment', location=''):
        """
        Generate multiple smart home configurations based on user inputs
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms to configure
        budget : float
            Total budget for the smart home setup
        priorities : dict
            Dictionary with user priorities (energy_efficiency, security, ease_of_use, scalability)
        home_type : str, optional
            Type of home (apartment, house, etc.)
        location : str, optional
            Geographic location
                
        Returns:
        --------
        list
            List of configuration dictionaries
        """
        configurations = []
        # Configuration names and descriptions
        config_names = [
            "Balanced Setup",
            "Energy-Efficient Setup",
            "High-Security Setup"
        ]
        
        config_descriptions = [
            "A balanced configuration that optimizes for all priorities",
            "Prioritizes energy efficiency with smart power management",
            "Focuses on security features and reliability"
        ]
        allowed_categories = []
        if priorities.get("lighting"):
            allowed_categories.append("Lighting")
        if priorities.get("security_devices"):
            allowed_categories.append("Security")
        if priorities.get("climate_control"):
            allowed_categories.append("HVAC")
        if priorities.get("energy_management"):
            allowed_categories.append("Energy Management")
        if not priorities or not any([priorities.get("lighting"), priorities.get("security_devices"), 
            priorities.get("climate_control"), priorities.get("energy_management")]):
            allowed_categories = ["Lighting", "Security", "HVAC", "Energy Management"]
        original_df = self.df.copy()
        self.df = self.df[self.df['Category'].isin(allowed_categories)].reset_index(drop=True)
        # Generate 3 different configurations with different weight variations
        for variant in range(3):
            # Adjust weights based on priorities and variant
            weights = self.adjust_weights_from_priorities(priorities, variant)
            
            # Calculate composite scores with the adjusted weights
            self.calculate_composite_score(weights)
            
            # Optimize component selection
            optimization_result = self.optimize_component_selection(num_rooms, budget, weights, priorities)
            
            # Apply home type and location adjustments
            optimization_result['selected_components'] = self.adjust_for_home_type_and_location(
                home_type, 
                location, 
                optimization_result['selected_components']
            )
            
            # Apply ML enhancements if available
            if hasattr(self, 'compatibility_model'):
                # Convert user priorities to features for personalization
                user_features = self._convert_priorities_to_features(priorities)
                
                # Enhance component scores with ML
                optimization_result['selected_components'] = self.enhance_component_scores_with_ml(
                    optimization_result['selected_components'],
                    user_features
                )
            
            # Allocate components to rooms with priorities consideration
            room_allocations = self.allocate_components_to_rooms(
                optimization_result['selected_components'], 
                num_rooms,
                priorities
            )
            
            # Calculate energy consumption estimates based on priorities
            energy_estimates = self.estimate_energy_consumption(
                optimization_result['selected_components'],
                priorities
            )
            
            # Check component compatibility
            compatibility_info = self.check_component_compatibility(
                optimization_result['selected_components']
            )
            
            # Create configuration dictionary
            configuration = {
                'variant': variant,
                'name': config_names[variant],
                'description': config_descriptions[variant],
                'weights': weights,
                'home_type': home_type,
                'location': location,
                'optimization_result': optimization_result,
                'room_allocations': room_allocations,
                'energy_estimates': energy_estimates,
                'compatibility_info': compatibility_info,
                'total_cost': optimization_result['total_cost'],
                'total_components': len(optimization_result['selected_components'])
            }
            
            configurations.append(configuration)
        self.df = original_df
        return configurations

    def optimize_component_selection(self, num_rooms, budget, weights=None, priorities=None):
        """
        Optimize component selection based on budget and room requirements
        """
        # Recompute scores if missing
        if 'Composite_Score' not in self.df.columns:
            self.calculate_composite_score(weights)

        # Quick NaN check
        n_nans = self.df['Composite_Score'].isnull().sum()
        print(f"[DEBUG] {n_nans} NaNs in Composite_Score")

        # Get required counts
        required_components = self._determine_required_components(num_rooms)

        # Build LP
        prob = LpProblem("SmartHomeComponentSelection", LpMaximize)

        # Integer vars 0–5, plus binary “used at all” vars
        x = [LpVariable(f"x_{i}", lowBound=0, upBound=5, cat="Integer")
            for i in range(len(self.df))]
        y = [LpVariable(f"y_{i}", cat="Binary")
            for i in range(len(self.df))]

        # Debug their bounds
        print("[DEBUG] Var[0] bounds:", x[0].lowBound, x[0].upBound, x[0].cat)

        # Objective = sum(score*qty) − penalty*sum(y)
        penalty = 0.5
        prob += (
            lpSum(x[i] * self.df.iloc[i]['Composite_Score'] for i in range(len(self.df)))
            - penalty * lpSum(y[i] for i in range(len(self.df)))
        )

        # Budget
        prob += lpSum(x[i] * self.df.iloc[i]['Price_INR'] for i in range(len(self.df))) <= budget

        # Category minima
        for cat, cnt in required_components.items():
            idxs = [i for i,row in self.df.iterrows() if row['Category']==cat]
            if idxs:
                prob += lpSum(x[i] for i in idxs) >= cnt

        # Link x→y
        for i in range(len(self.df)):
            prob += x[i] <= 5 * y[i]

        # Solve
        prob.solve(PULP_CBC_CMD(msg=False))

        # Dump out the raw solution quantities
        quantities = [int(var.varValue or 0) for var in x]
        print("[DEBUG] Solution quantities:", quantities[:10], "…")

        # Collect results
        selected = []
        cost = 0
        for i, row in self.df.iterrows():
            q = quantities[i]
            if q > 0:
                comp = row.to_dict()
                comp['Quantity'] = q
                comp['Total_Price_INR'] = comp['Price_INR'] * q
                selected.append(comp)
                cost += comp['Total_Price_INR']

        return {
            'selected_components': selected,
            'total_cost': cost,
            'budget': budget,
            'remaining_budget': budget - cost
        }

    
    def _determine_required_components(self, num_rooms):
        """
        Determine required components based on number of rooms
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms to configure
            
        Returns:
        --------
        dict
            Dictionary with required component counts by category
        """
        # Initialize required components dictionary
        required_components = {
            "Lighting": 0,
            "Security": 0,
            "HVAC": 0,
            "Energy Management": 0
        }
        
        # Assign room types based on number of rooms
        room_types = []
        if num_rooms >= 1:
            room_types.append("Living Room")
        if num_rooms >= 2:
            room_types.append("Bedroom")
        if num_rooms >= 3:
            room_types.append("Kitchen")
        if num_rooms >= 4:
            room_types.append("Bathroom")
        if num_rooms >= 5:
            room_types.append("Hallway")
        if num_rooms >= 6:
            room_types.append("Entrance")
        
        # Add additional bedrooms if needed
        while len(room_types) < num_rooms:
            room_types.append("Bedroom")
        
        # Calculate required components based on room types
        for room_type in room_types:
            if room_type in self.room_types:
                for category, count in self.room_types[room_type].items():
                    required_components[category] += count
        
        return required_components

    def allocate_components_to_rooms(self, components, num_rooms, priorities=None):
        """
        Allocate components to rooms based on room types and user priorities

        Parameters:
        -----------
        components : list
            List of selected components
        num_rooms : int
            Number of rooms to configure
        priorities : dict, optional
            User priorities to influence allocation

        Returns:
        --------
        list
            List of room dictionaries with allocated components
        """
        import copy
        components_copy = copy.deepcopy(components)

        # Group similar components to avoid duplicates and sum their quantities
        grouped_components = self._group_similar_components(components_copy)

        # Assign room types based on number of rooms
        room_types = []
        if num_rooms >= 1:
            room_types.append("Living Room")
        if num_rooms >= 2:
            room_types.append("Bedroom")
        if num_rooms >= 3:
            room_types.append("Kitchen")
        if num_rooms >= 4:
            room_types.append("Bathroom")
        if num_rooms >= 5:
            room_types.append("Hallway")
        if num_rooms >= 6:
            room_types.append("Entrance")

        # Add additional bedrooms if needed
        while len(room_types) < num_rooms:
            room_types.append(f"Bedroom {len([r for r in room_types if 'Bedroom' in r]) + 1}")

        # Initialize rooms with empty component lists
        rooms = [{'name': room_type, 'components': []} for room_type in room_types]

        # Calculate room priorities based on user preferences
        room_priorities = {}
        for room in rooms:
            room_type = room['name'].split(' ')[0] if ' ' in room['name'] else room['name']
            priority = 5
            if priorities:
                if room_type == "Living" and 'energy_efficiency' in priorities:
                    priority += priorities['energy_efficiency'] * 0.3
                if room_type in ["Entrance", "Hallway"] and 'security' in priorities:
                    priority += priorities['security'] * 0.4
                if room_type == "Kitchen" and 'energy_efficiency' in priorities:
                    priority += priorities['energy_efficiency'] * 0.2
            room_priorities[room['name']] = priority

        # Group components by category
        components_by_category = {}
        for component in grouped_components:
            category = component['Category']
            if category not in components_by_category:
                components_by_category[category] = []
            components_by_category[category].append(component)

        # Allocate components to rooms based on room types and preserve quantities
        for room in rooms:
            room_type = room['name'].split(' ')[0] if ' ' in room['name'] else room['name']
            if room_type in self.room_types:
                for category, count in self.room_types[room_type].items():
                    if category in components_by_category and components_by_category[category]:
                        allocated_count = 0
                        i = 0
                        while i < len(components_by_category[category]) and allocated_count < count:
                            component = components_by_category[category][i]
                            quantity = int(component.get('Quantity', 1))
                            to_allocate = min(quantity, count - allocated_count)
                            if to_allocate > 0:
                                room_component = component.copy()
                                room_component['Quantity'] = to_allocate
                                room_component['Total_Price_INR'] = component['Price_INR'] * to_allocate
                                room['components'].append(room_component)
                                remaining = quantity - to_allocate
                                if remaining > 0:
                                    component['Quantity'] = remaining
                                    component['Total_Price_INR'] = component['Price_INR'] * remaining
                                    i += 1
                                else:
                                    components_by_category[category].pop(i)
                            else:
                                i += 1

        # Distribute any remaining components to rooms with highest priority
        for category, category_components in components_by_category.items():
            while category_components:
                sorted_rooms = sorted(rooms, key=lambda r: room_priorities.get(r['name'], 0), reverse=True)
                component = max(category_components, key=lambda c: c.get('Quantity', 1))
                sorted_rooms[0]['components'].append(component)
                category_components.remove(component)

        # Ensure all components have Quantity field explicitly set as int
        for room in rooms:
            for component in room['components']:
                if 'Quantity' not in component:
                    component['Quantity'] = 1
                component['Quantity'] = int(component['Quantity'])

        return rooms

    def _group_similar_components(self, components):
        """Group similar components and sum their quantities"""
        grouped = {}
        for comp in components:
            key = comp['Component_Name']
            if key in grouped:
                grouped[key]['Quantity'] += comp.get('Quantity', 1)
                grouped[key]['Total_Price_INR'] += comp.get('Total_Price_INR', comp['Price_INR'])
            else:
                comp_copy = comp.copy()
                comp_copy['Quantity'] = comp.get('Quantity', 1)
                comp_copy['Total_Price_INR'] = comp_copy['Price_INR'] * comp_copy['Quantity']
                grouped[key] = comp_copy
        return list(grouped.values())

    def visualize_component_distribution(self, configuration):
        """
        Visualize component distribution by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with component distribution visualization
        """
        # Extract component categories
        components = configuration['optimization_result']['selected_components']
        categories = [comp['Category'] for comp in components]
        
        # Count components by category
        category_counts = {}
        for category in categories:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar chart
        bars = ax.bar(category_counts.keys(), category_counts.values())
        
        # Add labels and title
        ax.set_xlabel('Component Category')
        ax.set_ylabel('Number of Components')
        ax.set_title('Component Distribution by Category')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_cost_breakdown(self, configuration):
        """
        Visualize cost breakdown by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with cost breakdown visualization
        """
        # Extract component categories and prices
        components = configuration['optimization_result']['selected_components']
        
        # Calculate cost by category
        category_costs = {}
        for comp in components:
            category = comp['Category']
            price = comp['Price_INR']
            
            if category in category_costs:
                category_costs[category] += price
            else:
                category_costs[category] = price
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            category_costs.values(), 
            labels=category_costs.keys(),
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Add title
        ax.set_title('Cost Breakdown by Category')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_room_allocation(self, configuration):
        """
        Visualize component allocation by room
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with room allocation visualization
        """
        # Extract room allocations
        rooms = configuration.get('room_allocations', [])
        
        # Calculate component counts by room
        room_names = []
        component_counts = []
        
        for room in rooms:
            room_names.append(room['name'])
            component_counts.append(len(room['components']))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = ax.barh(room_names, component_counts)
        
        # Add labels and title
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Room')
        ax.set_title('Component Allocation by Room')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.0f}', ha='left', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, configuration, output_path=None):
        """
        Generate HTML report for a configuration
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
        output_path : str, optional
            Path to save the HTML report
            
        Returns:
        --------
        str
            HTML report content
        """
        # Extract configuration details
        variant = configuration['variant']
        weights = configuration['weights']
        total_cost = configuration['total_cost']
        components = configuration['optimization_result']['selected_components']
        rooms = configuration.get('room_allocations', [])
        
        # Generate HTML content
        html = f"""
        <html>
        <head>
            <title>Smart Home Configuration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #eef7fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .room {{ background-color: #f0f7e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Smart Home Configuration Report</h1>
            
            <div class="summary">
                <h2>Configuration Summary</h2>
                <p><strong>Variant:</strong> {variant}</p>
                <p><strong>Total Cost:</strong> ₹{total_cost:.2f}</p>
                <p><strong>Total Components:</strong> {len(components)}</p>
                <p><strong>Weights Used:</strong></p>
                <ul>
                    <li>Efficiency: {weights['efficiency']:.2f}</li>
                    <li>Reliability: {weights['reliability']:.2f}</li>
                    <li>Price: {weights['price']:.2f}</li>
                </ul>
            </div>
            
            <h2>Component List</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Category</th>
                    <th>Price (₹)</th>
                    <th>Efficiency</th>
                    <th>Reliability</th>
                </tr>
        """
        
        # Add component rows
        for comp in components:
            component_name = comp.get('Name', comp.get('Component', f"Component {components.index(comp)+1}"))
            
            html += f"""
                <tr>
                    <td>{component_name}</td>
                    <td>{comp['Category']}</td>
                    <td>{comp['Price_INR']:.2f}</td>
                    <td>{comp['Efficiency']}</td>
                    <td>{comp['Reliability']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Room Allocations</h2>
        """
        
        # Add room sections
        for room in rooms:
            html += f"""
            <div class="room">
                <h3>{room['name']}</h3>
                <p><strong>Components:</strong> {len(room['components'])}</p>
                
                <table>
                    <tr>
                        <th>Name</th>
                        <th>Category</th>
                        <th>Price (₹)</th>
                        <th>Efficiency</th>
                        <th>Reliability</th>
                    </tr>
            """
            
            # Add component rows for this room
            for comp in room['components']:
                # Check if 'Name' key exists, otherwise use a default or another key
                component_name = comp.get('Name', comp.get('Component', f"Component {room['components'].index(comp)+1}"))
                
                html += f"""
                    <tr>
                        <td>{component_name}</td>
                        <td>{comp['Category']}</td>
                        <td>{comp['Price_INR']:.2f}</td>
                        <td>{comp['Efficiency']}</td>
                        <td>{comp['Reliability']}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
        
        return html