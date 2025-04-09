import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
from datetime import timedelta

# Load Dataset
df = pd.read_csv('demands.csv')

# Load Models
with open('best_rf_demand_forecasting_model.pkl', 'rb') as f:
    demand_forecast_model = joblib.load(f)

with open('best_random_forest_overstock_model.pkl', 'rb') as f:
    overstock_model = joblib.load(f)

# Streamlit App
st.set_page_config(page_title="AI Demand Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 AI-Powered Demand Forecasting and Overstock Prediction")

# Sidebar for Input
st.sidebar.header("⚙️ User Input")
product_names = df['Product_name'].dropna().unique()
search_product = st.sidebar.text_input("Search Product Name:")

if search_product:
    filtered_products = [p for p in product_names if search_product.lower() in p.lower()]
else:
    filtered_products = sorted(product_names)

product_name = st.sidebar.selectbox("Select Product Name:", filtered_products)
forecast_horizon = st.sidebar.selectbox("Select Forecast Horizon (days):", [15, 30, 60])

if product_name:
    matched_products = df[df['Product_name'] == product_name]

    if not matched_products.empty:
        selected_product = matched_products.iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔎 Product Information")
            st.markdown(f"**Product ID:** {selected_product['Product_ID']}")
            st.markdown(f"**Quantity Available:** {selected_product['Qty']}")
            st.markdown(f"**Lead Time (days):** {selected_product['Lead Time(days)']}")

        with col2:
            with st.expander("📊 View Product Sales Statistics"):
                product_data = df[df['Product_ID'] == selected_product['Product_ID']]
                st.write(f"**Total Sales:** {product_data['Sales'].sum()}")
                st.write(f"**Average Sales/Day:** {product_data['Sales'].mean():.2f}")
                st.write(f"**Max Sales:** {product_data['Sales'].max()}")
                st.write(f"**Min Sales:** {product_data['Sales'].min()}")

        # Prepare Data
        product_data['Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
        product_data = product_data.dropna(subset=['Date']).sort_values('Date')

        if product_data.empty or product_data[['Sales', 'Moving_Avg_Sales', 'Cumulative_Sales']].isnull().any().any():
            st.error("⚠️ Insufficient data for forecasting.")
        else:
            try:
                st.subheader(f"📈 Demand Forecast for Next {forecast_horizon} Days")
                future_dates = pd.date_range(start=product_data['Date'].max() + timedelta(days=1), periods=forecast_horizon)

                last_features = product_data[['Moving_Avg_Sales', 'Lead_Time_Impact', 'Month', 'Week', 'Day_of_Week']].iloc[-1].values
                future_features = [last_features] * forecast_horizon
                future_demand = demand_forecast_model.predict(future_features)

                # Clip negative forecasts to 0
                future_demand = future_demand.clip(min=0)

                # Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=product_data['Date'], y=product_data['Sales'], mode='lines+markers', name='Actual Sales'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_demand, mode='lines+markers', name='Forecasted Sales', line=dict(dash='dash', color='green')))

                # Add shaded area for forecast
                fig.add_vrect(
                    x0=future_dates.min(), x1=future_dates.max(),
                    fillcolor="lightgreen", opacity=0.3, line_width=0
                )

                fig.update_layout(
                    title='Sales Forecast',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    legend=dict(x=0, y=1),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted_Sales': future_demand
                })

                st.download_button(
                    label="📥 Download Forecasted Data",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"{selected_product['Product_name']}_forecast.csv",
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"❌ Forecasting Error: {e}")

            # Overstock Prediction
            try:
                st.subheader("🚦 Overstock Prediction")
                features = selected_product[['Qty', 'Lead Time(days)', 'Moving_Avg_Sales', 'Cumulative_Sales', 'Lead_Time_Impact', 'Month', 'Week', 'Day_of_Week']].values.reshape(1, -1)
                prediction = overstock_model.predict(features)

                if prediction[0] == 1:
                    st.success("⚡ Overstock Alert: High Inventory Detected!")
                else:
                    st.info("✅ Inventory Levels are Optimal.")
            except Exception as e:
                st.error(f"❌ Overstock Prediction Error: {e}")

    else:
        st.error("🚫 No matching product found. Please select a valid product.")
