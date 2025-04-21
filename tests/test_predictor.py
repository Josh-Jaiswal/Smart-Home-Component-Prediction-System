import pytest
from smart_home_predictor import SmartHomePredictor

@pytest.fixture
def predictor():
    return SmartHomePredictor()

def test_data_loading(predictor):
    assert not predictor.df.empty
    assert {'Price_INR', 'Efficiency', 'Category'}.issubset(predictor.df.columns)

def test_feature_engineering(predictor):
    features = predictor._extract_component_features()
    assert 'Price_to_Efficiency' in features.columns
    assert 'Reliability_per_Rupee' in features.columns

def test_model_training(predictor):
    assert hasattr(predictor, 'compatibility_model')
    assert hasattr(predictor, 'performance_model')
    
def test_prediction_flow(predictor):
    sample_components = predictor.df.sample(3).to_dict('records')
    enhanced = predictor.enhance_component_scores_with_ml(sample_components)
    assert len(enhanced) == 3
    assert all('Score' in comp for comp in enhanced)