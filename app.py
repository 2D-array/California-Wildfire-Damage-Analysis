import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="California Wildfire Analysis Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to implement dark theme
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 {
        color: #FF6B6B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2D2D2D;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B;
        color: #1E1E1E;
    }
    .css-zt5igj {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    .stMarkdown p {
        color: #E0E0E0;
    }
    .stButton button {
        background-color: #FF6B6B;
        color: #1E1E1E;
    }
    .stButton button:hover {
        background-color: #E05E5E;
        color: #1E1E1E;
    }
    .st-bk {
        color: #E0E0E0;
    }
    .st-emotion-cache-16txtl3 {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .st-emotion-cache-6qob1r {
        background-color: #3D3D3D;
        color: #E0E0E0;
    }
    .css-81oif8 {
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔥 California Wildfire Analysis Dashboard")
st.markdown("""
This interactive dashboard provides comprehensive analysis of California wildfire data,
including temporal patterns, causes, geographic impact, and predictive analytics.
Use the sidebar to filter data and explore different aspects of wildfire impact.
""")

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset/California Wildfire Damage.csv")
        df.dropna(subset=["Date", "Area_Burned (Acres)", "Cause"], inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        # Extract year and month for seasonal analysis
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Create a severity index
        df['Severity'] = df['Area_Burned (Acres)'] * 0.4 + df['Fatalities'] * 0.3 + df['Injuries'] * 0.2 + df['Estimated_Financial_Loss (Million $)'] * 0.1
        
        # Calculate ratios
        df['Financial_Loss_Per_Acre'] = df['Estimated_Financial_Loss (Million $)'] / df['Area_Burned (Acres)']
        df['Fatalities_Per_Acre'] = df['Fatalities'] / df['Area_Burned (Acres)']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Failed to load data. Please check the file path and format.")
    st.stop()

# Sidebar for filtering
st.sidebar.header("Data Filters")

# Year range filter
years = sorted(df['Year'].unique())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

# Cause filter
all_causes = sorted(df['Cause'].unique())
selected_causes = st.sidebar.multiselect(
    "Select Causes",
    options=all_causes,
    default=all_causes
)

# Location filter
all_locations = sorted(df['Location'].unique())
selected_locations = st.sidebar.multiselect(
    "Select Locations",
    options=all_locations,
    default=all_locations[:5] if len(all_locations) > 5 else all_locations
)

# Apply filters
filtered_df = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1]) & 
    (df['Cause'].isin(selected_causes)) & 
    (df['Location'].isin(selected_locations))
]

if filtered_df.empty:
    st.warning("No data available with the selected filters. Please adjust your selection.")
    st.stop()

# Create tabs for different analysis sections
tabs = st.tabs([
    "📊 Overview", 
    "📅 Temporal Analysis", 
    "🧯 Cause Analysis", 
    "🗺️ Geographic Impact", 
    "📈 Correlation Analysis", 
    "🔍 Predictive Analytics"
])

# Configure dark theme for plotly
dark_template = go.layout.Template()
dark_template.layout.plot_bgcolor = "#2D2D2D"
dark_template.layout.paper_bgcolor = "#2D2D2D"
dark_template.layout.font = {"color": "#E0E0E0"}
dark_template.layout.xaxis = {"gridcolor": "#444444", "zerolinecolor": "#444444"}
dark_template.layout.yaxis = {"gridcolor": "#444444", "zerolinecolor": "#444444"}

# Tab 1: Overview
with tabs[0]:
    st.header("Dataset Overview")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Wildfires", 
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df)/len(df)*100:.1f}% of all records"
        )
    with col2:
        st.metric(
            "Total Area Burned (Acres)", 
            f"{filtered_df['Area_Burned (Acres)'].sum():,.0f}"
        )
    with col3:
        st.metric(
            "Total Fatalities", 
            f"{filtered_df['Fatalities'].sum():,.0f}"
        )
    with col4:
        st.metric(
            "Financial Loss (Million $)", 
            f"{filtered_df['Estimated_Financial_Loss (Million $)'].sum():,.0f}"
        )
    
    st.subheader("Distribution of Wildfire Impacts")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            filtered_df, 
            x="Area_Burned (Acres)", 
            nbins=30,
            title="Distribution of Area Burned",
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(
            template=dark_template,
            xaxis_title="Area Burned (Acres)", 
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df, 
            y="Area_Burned (Acres)",
            title="Area Burned Distribution",
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Wildfire Cause Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        cause_counts = filtered_df['Cause'].value_counts().reset_index()
        cause_counts.columns = ['Cause', 'Count']
        fig = px.bar(
            cause_counts, 
            x='Count', 
            y='Cause',
            title="Wildfires by Cause",
            color='Count',
            color_continuous_scale='Inferno',
            orientation='h'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        area_by_cause = filtered_df.groupby('Cause')['Area_Burned (Acres)'].sum().reset_index()
        fig = px.bar(
            area_by_cause, 
            x='Area_Burned (Acres)', 
            y='Cause',
            title="Area Burned by Cause",
            color='Area_Burned (Acres)',
            color_continuous_scale='Inferno',
            orientation='h'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Temporal Analysis
with tabs[1]:
    st.header("Temporal Analysis")
    
    # Yearly trend
    yearly_fires = filtered_df.groupby('Year').agg({
        'Year': 'count',
        'Area_Burned (Acres)': 'sum',
        'Fatalities': 'sum',
        'Estimated_Financial_Loss (Million $)': 'sum'
    }).rename(columns={'Year': 'Count'}).reset_index()
    
    st.subheader("Yearly Trends")
    metrics = st.selectbox(
        "Select Metric for Yearly Trend",
        ["Count", "Area_Burned (Acres)", "Fatalities", "Estimated_Financial_Loss (Million $)"],
        key="yearly_metric"
    )
    
    fig = px.line(
        yearly_fires, 
        x='Year', 
        y=metrics,
        title=f"Yearly Trend of {metrics}",
        markers=True,
        line_shape="linear",
        color_discrete_sequence=['#FF6B6B']
    )
    fig.update_layout(
        template=dark_template,
        xaxis_title="Year", 
        yaxis_title=metrics
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly patterns
    st.subheader("Seasonal Patterns")
    monthly_fires = filtered_df.groupby('Month').size().reset_index(name='Count')
    fig = px.bar(
        monthly_fires, 
        x='Month', 
        y='Count',
        title="Number of Wildfires by Month",
        color='Count',
        color_continuous_scale='Inferno'
    )
    fig.update_layout(
        template=dark_template,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        xaxis_title="Month",
        yaxis_title="Number of Wildfires"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cause trends over time
    st.subheader("Cause Trends Over Time")
    cause_by_year = pd.crosstab(filtered_df['Year'], filtered_df['Cause'])
    fig = px.area(
        cause_by_year, 
        title="Causes of Wildfires Over Years",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    fig.update_layout(
        template=dark_template,
        xaxis_title="Year", 
        yaxis_title="Number of Wildfires"
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Cause Analysis
with tabs[2]:
    st.header("Cause Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity by Cause")
        fig = px.box(
            filtered_df, 
            x='Cause', 
            y='Severity',
            title="Wildfire Severity by Cause",
            color='Cause',
            color_discrete_sequence=px.colors.sequential.Inferno
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Homes Destroyed by Cause")
        homes_by_cause = filtered_df.groupby('Cause')['Homes_Destroyed'].sum().reset_index()
        fig = px.bar(
            homes_by_cause,
            x='Cause',
            y='Homes_Destroyed',
            title="Total Homes Destroyed by Cause",
            color='Homes_Destroyed',
            color_continuous_scale='Inferno'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Financial Impact by Cause")
    financial_by_cause = filtered_df.groupby('Cause')['Estimated_Financial_Loss (Million $)'].sum().reset_index()
    fig = px.pie(
        financial_by_cause, 
        values='Estimated_Financial_Loss (Million $)', 
        names='Cause',
        title="Financial Loss Share by Cause",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    fig.update_layout(template=dark_template)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical testing
    st.subheader("Statistical Analysis: ANOVA Test")
    st.markdown("""
    Analysis of Variance (ANOVA) tests whether there are significant differences 
    in Area Burned between different causes.
    """)
    
    @st.cache_data
    def run_anova(df):
        causes = df['Cause'].unique()
        area_by_cause = [df[df['Cause'] == cause]['Area_Burned (Acres)'].dropna().values for cause in causes]
        # Filter out empty arrays
        area_by_cause = [arr for arr in area_by_cause if len(arr) > 0]
        if len(area_by_cause) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_val = stats.f_oneway(*area_by_cause)
            return f_stat, p_val, True
        return 0, 0, False
    
    f_stat, p_val, valid = run_anova(filtered_df)
    
    if valid:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", f"{f_stat:.2f}")
        with col2:
            st.metric("p-value", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success("There is a statistically significant difference in area burned between different causes (p < 0.05).")
        else:
            st.info("There is no statistically significant difference in area burned between different causes (p >= 0.05).")
    else:
        st.warning("Insufficient data for ANOVA test with current filters.")

# Tab 4: Geographic Impact
with tabs[3]:
    st.header("Geographic Impact Analysis")
    
    # Top locations
    top_locations = filtered_df.groupby('Location')['Area_Burned (Acres)'].sum().sort_values(ascending=False).head(10).reset_index()
    
    fig = px.bar(
        top_locations, 
        x='Area_Burned (Acres)', 
        y='Location',
        title="Top 10 Locations by Area Burned",
        color='Area_Burned (Acres)',
        color_continuous_scale='Inferno',
        orientation='h'
    )
    fig.update_layout(template=dark_template)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Financial loss by location
        df_grouped = filtered_df.groupby("Location")["Estimated_Financial_Loss (Million $)"].sum().reset_index()
        df_grouped = df_grouped.sort_values("Estimated_Financial_Loss (Million $)", ascending=False).head(10)
        
        fig = px.bar(
            df_grouped, 
            x="Estimated_Financial_Loss (Million $)", 
            y="Location",
            title="Top 10 Locations by Financial Loss",
            color="Estimated_Financial_Loss (Million $)",
            color_continuous_scale='Viridis',
            orientation='h'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fatalities by location
        df_fatalities = filtered_df.groupby("Location")["Fatalities"].sum().reset_index()
        df_fatalities = df_fatalities.sort_values("Fatalities", ascending=False).head(10)
        
        fig = px.bar(
            df_fatalities, 
            x="Fatalities", 
            y="Location",
            title="Top 10 Locations by Fatalities",
            color="Fatalities",
            color_continuous_scale='Inferno',
            orientation='h'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact ratio by location
    st.subheader("Impact Ratios by Location")
    ratio_metric = st.selectbox(
        "Select Impact Ratio",
        ["Financial_Loss_Per_Acre", "Fatalities_Per_Acre"],
        key="impact_ratio"
    )
    
    location_ratios = filtered_df.groupby("Location")[ratio_metric].mean().sort_values(ascending=False).head(10).reset_index()
    
    fig = px.bar(
        location_ratios,
        x=ratio_metric,
        y="Location",
        title=f"Top 10 Locations by {ratio_metric}",
        color=ratio_metric,
        color_continuous_scale='Plasma',
        orientation='h'
    )
    fig.update_layout(template=dark_template)
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Correlation Analysis
with tabs[4]:
    st.header("Correlation Analysis")
    
    # Correlation matrix
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Select Variables for Correlation Analysis",
        options=numeric_cols,
        default=["Area_Burned (Acres)", "Fatalities", "Injuries", "Estimated_Financial_Loss (Million $)", "Homes_Destroyed"]
    )
    
    if len(selected_cols) >= 2:
        corr_matrix = filtered_df[selected_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix"
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least two variables for correlation analysis.")
    
    # Scatter plots
    st.subheader("Relationship Between Variables")
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_cols, index=numeric_cols.index("Area_Burned (Acres)") if "Area_Burned (Acres)" in numeric_cols else 0)
    
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_cols, index=numeric_cols.index("Injuries") if "Injuries" in numeric_cols else 1)
    
    # Add color variable option
    color_var = st.selectbox("Color By (Optional)", ["None"] + numeric_cols)
    
    if color_var == "None":
        fig = px.scatter(
            filtered_df, 
            x=x_var, 
            y=y_var,
            title=f"{y_var} vs {x_var}",
            opacity=0.7,
            size_max=10,
            color_discrete_sequence=['#FF6B6B']
        )
    else:
        fig = px.scatter(
            filtered_df, 
            x=x_var, 
            y=y_var,
            color=color_var,
            title=f"{y_var} vs {x_var}, Colored by {color_var}",
            opacity=0.7,
            size_max=10,
            color_continuous_scale='Plasma'
        )
    
    fig.update_layout(template=dark_template)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display regression line option
    if st.checkbox("Add Regression Line"):
        fig = px.scatter(
            filtered_df, 
            x=x_var, 
            y=y_var,
            trendline="ols",
            title=f"{y_var} vs {x_var} with Regression Line",
            opacity=0.7,
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display regression results
        import statsmodels.api as sm
        X = filtered_df[x_var].dropna()
        y = filtered_df[y_var].dropna()
        # Get matching indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) > 1:  # Need at least 2 data points for regression
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            st.code(model.summary().as_text())
        else:
            st.warning("Insufficient data for regression analysis.")

# Tab 6: Predictive Analytics
with tabs[5]:
    st.header("Predictive Analytics")
    
    st.subheader("Predictive Model: Financial Loss Estimation")
    st.markdown("""
    This section uses a Random Forest model to predict financial losses from wildfires
    based on other characteristics.
    """)
    
    # Model features
    all_features = [col for col in filtered_df.columns if col not in [
        'Date', 'Estimated_Financial_Loss (Million $)', 'Year', 'Month', 
        'Severity', 'Financial_Loss_Per_Acre', 'Fatalities_Per_Acre', 'Location', 'Cause'
    ]]
    
    selected_features = st.multiselect(
        "Select Features for Prediction Model",
        options=all_features,
        default=['Area_Burned (Acres)', 'Homes_Destroyed', 'Businesses_Destroyed', 'Injuries']
    )
    
    if len(selected_features) >= 2:
        # Prepare data
        X = filtered_df[selected_features].copy()
        y = filtered_df['Estimated_Financial_Loss (Million $)'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.spinner("Training model..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        with col2:
            st.metric("R² Score", f"{r2:.2f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance, 
            x='Importance', 
            y='Feature',
            title="Feature Importance for Predicting Financial Loss",
            color='Importance',
            color_continuous_scale='Plasma',
            orientation='h'
        )
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual
        fig = px.scatter(
            x=y_test, 
            y=y_pred,
            title="Predicted vs Actual Financial Loss",
            labels={'x': 'Actual Financial Loss (Million $)', 'y': 'Predicted Financial Loss (Million $)'},
            color_discrete_sequence=['#FF6B6B']
        )
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines', 
            name='Perfect Prediction',
            line=dict(color='#FFA07A', dash='dash')
        ))
        fig.update_layout(template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive prediction
        st.subheader("Interactive Prediction")
        st.markdown("Adjust the sliders to predict financial loss for a hypothetical wildfire:")
        
        # Create prediction form with sliders based on selected features
        prediction_inputs = {}
        for feature in selected_features:
            min_val = float(filtered_df[feature].min())
            max_val = float(filtered_df[feature].max())
            mean_val = float(filtered_df[feature].mean())
            
            step = (max_val - min_val) / 100
            if step == 0:
                step = 0.01
                
            prediction_inputs[feature] = st.slider(
                f"{feature}", 
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step
            )
        
        # Make prediction
        if st.button("Predict Financial Loss"):
            input_df = pd.DataFrame([prediction_inputs])
            prediction = model.predict(input_df)[0]
            
            st.success(f"Predicted Financial Loss: ${prediction:.2f} Million")
    else:
        st.warning("Please select at least two features for the prediction model.")

# Footer
st.markdown("---")
st.markdown("2024 California Wildfire Analysis Project | Day 11 of 50Days50Projects ")