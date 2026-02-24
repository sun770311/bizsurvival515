import os
import pandas as pd
import plotly.express as px

def main():
    # 1. Load your real data
    # Assuming this script is in your 'map' folder, and data is in '../data'
    data_path = os.path.join(os.path.dirname(__file__), "../data/business_points.csv")
    
    print(f"Loading data from {os.path.abspath(data_path)}...")
    df = pd.read_csv(data_path, low_memory=False)

    # 2. Replicate your JS data cleaning
    # Drop missing coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Replicate your `inNYC` bounding box filter
    nyc_bbox = {'west': -74.30, 'south': 40.45, 'east': -73.65, 'north': 40.95}
    df = df[
        (df['Latitude'] >= nyc_bbox['south']) & 
        (df['Latitude'] <= nyc_bbox['north']) & 
        (df['Longitude'] >= nyc_bbox['west']) & 
        (df['Longitude'] <= nyc_bbox['east'])
    ]

    # Clean Borough column like your JS does
    df['Borough'] = df['Borough'].fillna('Not Recorded').replace('Outside NYC', 'Not Recorded')

    print(f"Cleaned data: {len(df)} NYC business points ready to map.")

    # 3. Define the exact color mapping from your JS
    color_map = {
        "Bronx": "#2563eb",
        "Manhattan": "#eab308",
        "Staten Island": "#dc2626",
        "Brooklyn": "#16a34a",
        "Queens": "#7c3aed",
        "Not Recorded": "#6b7280"
    }

    # 4. Build the Map
    fig = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Borough",
        color_discrete_map=color_map,
        hover_name="Business Name",
        hover_data={
            "Latitude": False, 
            "Longitude": False,
            "Business Unique ID": True,
            "Business Status": True,
            "Borough": False # Already in color legend
        },
        zoom=10,
        center={"lat": 40.7128, "lon": -74.0060}, # NYC_CENTER
        title="Plotly NYC Businesses (Real Data)"
    )

    # Configure map styling
    fig.update_traces(marker=dict(size=6, opacity=0.9)) # Matches circle-radius: 3 and opacity: 0.9
    fig.update_layout(
        map_style="open-street-map", 
        margin={"r":0, "t":40, "l":0, "b":0}
    )

    # 5. Save and view
    output_filename = "nyc_businesses_plotly.html"
    fig.write_html(output_filename)
    print(f"Success! Map saved locally as: {os.path.abspath(output_filename)}")
    print("Double-click this HTML file in your regular file explorer to view it.")

if __name__ == "__main__":
    main()