import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Suppress joblib warning


#page configuration
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#css
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E10600;
    }
    </style>
""", unsafe_allow_html=True)




# Load model and data artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and processed data"""
    model = joblib.load('best_pitstop_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    data = pd.read_csv('pit_stops_engineered.csv')
    X = pd.read_csv('features_X.csv')
    y = pd.read_csv('target_y.csv').values.ravel()
    
    return model, scaler, data, X, y

@st.cache_resource
def compute_shap_values(_model, X_sample):
    """Compute SHAP values (cached for performance)"""
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


#Track Mappings
TRACKS = {
    'Monaco': {'overtaking': 0, 'pit_loss': 18},
    'Singapore': {'overtaking': 1, 'pit_loss': 22},
    'Hungary': {'overtaking': 1, 'pit_loss': 20},
    'Zandvoort': {'overtaking': 1, 'pit_loss': 14},
    'Suzuka': {'overtaking': 1, 'pit_loss': 21},
    'Barcelona': {'overtaking': 2, 'pit_loss': 20},
    'Melbourne': {'overtaking': 2, 'pit_loss': 20},
    'Montreal': {'overtaking': 2, 'pit_loss': 15},
    'Silverstone': {'overtaking': 3, 'pit_loss': 18},
    'Bahrain': {'overtaking': 3, 'pit_loss': 20},
    'Jeddah': {'overtaking': 3, 'pit_loss': 19},
    'Austin': {'overtaking': 3, 'pit_loss': 19},
    'Spa': {'overtaking': 3, 'pit_loss': 17},
    'Monza': {'overtaking': 3, 'pit_loss': 16},
}



#Main function
def main():
    # Header
    st.markdown('<p class="main-header">🏎️ F1 Pit Stop Effectiveness Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown("### Predict pit stop success using machine learning + SHAP explanations")
    
    # Load artifacts
    try:
        model, scaler, data, X, y = load_model_artifacts()
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all files are in the same directory.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", 
                            ["Make Prediction", "Model Insights", "Historical Analysis"])
    
    if page == "Make Prediction":
        prediction_page(model, data, X)
    elif page == "Model Insights":
        model_insights_page(model, X, y)
    else:
        historical_analysis_page(data)



# prediction page
def prediction_page(model, data, X):
    """Interactive prediction interface"""
    st.header("Pit Stop Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Race Context")
        
        track = st.selectbox("Circuit", list(TRACKS.keys()))
        year = st.slider("Year", 2022, 2024, 2024)
        total_laps = st.number_input("Total Race Laps", 50, 70, 55)
        current_lap = st.slider("Current Lap", 1, int(total_laps), 25)
        
        st.subheader("Position & Strategy")
        
        position = st.slider("Current Position", 1, 20, 5)
        laps_to_end = total_laps - current_lap
        race_progress = current_lap / total_laps
        
    with col2:
        st.subheader("Tire Condition")
        
        compound = st.selectbox("Tire Compound", ["Soft", "Medium", "Hard"])
        compound_map = {"Soft": 1, "Medium": 2, "Hard": 3}
        compound_num = compound_map[compound]
        
        tire_life = st.slider("Tire Age (laps)", 0, 50, 20)
        fresh_tire = st.checkbox("Fresh Tires?", value=(tire_life <= 2))
        
        st.subheader("Track Characteristics")
        
        overtaking_diff = TRACKS[track]['overtaking']
        pit_loss = TRACKS[track]['pit_loss']
        
        overtaking_labels = {0: "Very Hard", 1: "Hard", 2: "Medium", 3: "Easy"}
        st.info(f"**Overtaking:** {overtaking_labels[overtaking_diff]}")
        st.info(f"**Pit Loss:** ~{pit_loss} seconds")
    
    # Create feature vector
    features = create_feature_vector(
        position, laps_to_end, race_progress, tire_life, compound_num,
        fresh_tire, overtaking_diff, pit_loss, year
    )
    
    # Prediction button
    if st.button("Predict Pit Stop Effectiveness", type="primary"):
        with st.spinner("Analyzing pit stop scenario..."):
            prediction, probability = make_prediction(model, features, X.columns)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                outcome = "✅ EFFECTIVE" if prediction == 1 else "❌ INEFFECTIVE"
                color = "#28a745" if prediction == 1 else "#dc3545"
                st.markdown(f"""
                    <div style='background-color: {color}; padding: 1.5rem; 
                                border-radius: 0.5rem; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>{outcome}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Probability", f"{probability:.1%}", 
                         delta="High confidence" if probability > 0.7 or probability < 0.3 else "Uncertain")
            
            with col3:
                risk_level = get_risk_level(probability, prediction)
                st.metric("Risk Level", risk_level)
            
            # Recommendation
            st.markdown("---")
            recommendation = generate_recommendation(prediction, probability, position, 
                                                    race_progress, tire_life, overtaking_diff)
            st.info(f"💡 **Recommendation:** {recommendation}")
            
            # SHAP Explanation
            st.markdown("---")
            st.subheader("🔍 Why This Prediction?")
            
            # Compute SHAP
            feature_df = pd.DataFrame([features], columns=X.columns)
            explainer, shap_values = compute_shap_values(model, feature_df)
            
            # Get SHAP values for this prediction (class 1)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # For class 1
            else:
                shap_vals = shap_values[0]
            
            # Create SHAP waterfall plot
            fig = plot_shap_waterfall(shap_vals, feature_df.columns, features, 
                                     explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                                     else explainer.expected_value)
            st.pyplot(fig)
            
            # Feature contributions table
            st.subheader("📋 Feature Impact Analysis")
            impact_df = create_impact_table(shap_vals, feature_df.columns, features)
            st.dataframe(impact_df, use_container_width=True)



#Create feature vector 
def create_feature_vector(position, laps_to_end, race_progress, tire_life, 
                         compound_num, fresh_tire, overtaking_diff, pit_loss, year):
    """Create feature vector matching training data"""
    
    # Position features
    gap_to_leader = position - 1
    is_leader = 1 if position == 1 else 0
    is_top_three = 1 if position <= 3 else 0
    is_midfield = 1 if 6 <= position <= 15 else 0
    is_backmarker = 1 if position > 15 else 0
    
    # Race phase features
    early_race = 1 if race_progress < 0.33 else 0
    mid_race = 1 if 0.33 <= race_progress < 0.67 else 0
    late_race = 1 if race_progress >= 0.67 else 0
    
    # Strategic features
    estimated_stint = tire_life
    in_pit_window = 1 if 15 <= (50 - laps_to_end) <= 37.5 else 0  # Approximate
    
    features = [
        position, gap_to_leader, is_leader, is_top_three, is_midfield, is_backmarker,
        tire_life, compound_num, int(fresh_tire),
        race_progress, early_race, mid_race, late_race, laps_to_end,
        overtaking_diff, pit_loss,
        estimated_stint, in_pit_window,
        year
    ]
    
    return features

#prediction function
def make_prediction(model, features, feature_names):
    """Make prediction and return result"""
    feature_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(feature_df)[0]
    probability = model.predict_proba(feature_df)[0][1]
    
    return prediction, probability


#SHAP waterfall plot
def plot_shap_waterfall(shap_values, feature_names, feature_values, base_value):
    """Create SHAP waterfall plot"""
    
    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_values = [shap_values[i] for i in indices]
    sorted_feature_vals = [feature_values[i] for i in indices]
    
    colors = ['#ff6b6b' if v < 0 else '#51cf66' for v in sorted_values]
    
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{feat} = {val:.2f}" for feat, val in 
                        zip(sorted_features, sorted_feature_vals)])
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title('Top 10 Feature Contributions')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    return fig


#Impact table
def create_impact_table(shap_values, feature_names, feature_values):
    """Create feature impact table"""
    
    impact_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values,
        'SHAP Impact': shap_values,
        'Effect': ['Increases Effectiveness' if v > 0 else 'Decreases Effectiveness' 
                   for v in shap_values]
    })
    
    impact_df['Abs Impact'] = np.abs(impact_df['SHAP Impact'])
    impact_df = impact_df.sort_values('Abs Impact', ascending=False)
    impact_df = impact_df.drop('Abs Impact', axis=1)
    impact_df = impact_df.reset_index(drop=True)
    
    return impact_df[['Feature', 'Value', 'SHAP Impact', 'Effect']].head(10)



#risk level
def get_risk_level(probability, prediction):
    """Determine risk level"""
    if prediction == 1:
        if probability > 0.8:
            return "Low Risk"
        elif probability > 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    else:
        if probability < 0.2:
            return "Very High Risk"
        elif probability < 0.4:
            return "High Risk"
        else:
            return "Medium Risk"



# recommendation
def generate_recommendation(prediction, probability, position, race_progress, 
                           tire_life, overtaking_diff):
    """Generate strategic recommendation"""
    
    if prediction == 1 and probability > 0.7:
        return "Pit stop highly recommended. Good chance to maintain or improve position."
    elif prediction == 1 and probability > 0.5:
        return "Pit stop viable but monitor competitors. Consider track position."
    elif prediction == 0 and probability < 0.3:
        return "Avoid pit stop. High risk of losing positions. Extend stint if possible."
    elif prediction == 0 and probability < 0.5:
        return "Pit stop risky. Only proceed if tires critically degraded or strategic need."
    else:
        return "Borderline scenario. Evaluate real-time factors (traffic, tire condition)."



# model insights page
def model_insights_page(model, X, y):
    """Display model performance and insights"""
    st.header("📈 Model Insights")
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1 Score", "80.6%")
    with col2:
        st.metric("ROC-AUC", "87.4%")
    with col3:
        st.metric("Accuracy", "79.6%")
    with col4:
        st.metric("Data Points", "3,183")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🔑 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                     title='Top 15 Most Important Features',
                     color='Importance', color_continuous_scale='reds')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # SHAP summary plot
    st.subheader("🎨 SHAP Summary (All Predictions)")
    
    with st.spinner("Computing SHAP values for sample..."):
        # Use sample for performance
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        explainer, shap_values = compute_shap_values(model, X_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, plot_type="dot", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
        
        st.pyplot(fig)
        
        st.info("📌 **Reading the plot:** Red = high feature value, Blue = low feature value. "
                "Right = increases effectiveness, Left = decreases effectiveness.")


# historical analysis page
def historical_analysis_page(data):
    """Explore historical pit stop data"""
    st.header("📊 Historical Data Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_year = st.multiselect("Year", data['Year'].unique(), 
                                       default=data['Year'].unique())
    with col2:
        selected_circuit = st.multiselect("Circuit", data['Circuit'].unique())
    with col3:
        selected_driver = st.multiselect("Driver", data['Driver'].unique())
    
    # Apply filters
    filtered_data = data.copy()
    if selected_year:
        filtered_data = filtered_data[filtered_data['Year'].isin(selected_year)]
    if selected_circuit:
        filtered_data = filtered_data[filtered_data['Circuit'].isin(selected_circuit)]
    if selected_driver:
        filtered_data = filtered_data[filtered_data['Driver'].isin(selected_driver)]
    
    st.write(f"Total data of {len(filtered_data)} pit stops")
    
    # Effectiveness by circuit
    st.subheader("🏁 Effectiveness by Circuit")
    
    circuit_stats = filtered_data.groupby('Circuit').agg({
        'Label': ['count', 'mean']
    }).round(3)
    circuit_stats.columns = ['Total Pit Stops', 'Effectiveness Rate']
    circuit_stats = circuit_stats.sort_values('Effectiveness Rate', ascending=False)
    
    fig = px.bar(circuit_stats.reset_index(), x='Circuit', y='Effectiveness Rate',
                 title='Pit Stop Effectiveness by Circuit',
                 color='Effectiveness Rate', color_continuous_scale='rdylgn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Position change distribution
    st.subheader("📈 Position Change Distribution")
    
    fig = px.histogram(filtered_data, x='PositionChange', nbins=30,
                       title='Position Change After Pit Stops',
                       labels={'PositionChange': 'Position Change (negative = gained positions)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics 
    st.subheader("📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Pit Stops", len(filtered_data))
        st.metric("Avg Position Before", f"{filtered_data['PositionBefore'].mean():.1f}")

    with col2:
        st.metric("Effective Stops", int(filtered_data['Label'].sum()))
        st.metric("Avg Tire Life", f"{filtered_data['TyreLife'].mean():.1f} laps")

    with col3:
        effectiveness = filtered_data['Label'].mean()
        st.metric("Effectiveness Rate", f"{effectiveness:.1%}")
        avg_change = filtered_data['PositionChange'].mean()
        st.metric("Avg Position Change", f"{avg_change:+.2f}")




if __name__ == "__main__":
    main()