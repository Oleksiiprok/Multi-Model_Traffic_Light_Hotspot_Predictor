# -*- coding: utf-8 -*-
"""
# Oleksii Prokopchenko All rights reserved, 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab - As-Is
"""

# Install necessary libraries
!pip install -q xgboost ipywidgets pandas numpy scikit-learn folium matplotlib

import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import folium
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# --- Фрагмент коду, що відповідає за мапу та її стабільний рендеринг ---
TILE_PROVIDER = 'Esri.WorldStreetMap'

# 1. Визначення віджетів виводу
out_map = widgets.Output(layout={'width': '100%', 'height': '320px', 'border': '2px solid #2980b9'})
out_report = widgets.Output(layout={'width': '100%'})

# 2. Функція створення інтерактивної мапи Folium
def create_interactive_map(location, zoom=14):
    return folium.Map(
        location=location,
        zoom_start=zoom,
        tiles=TILE_PROVIDER,
        zoom_control=True,
        scrollWheelZoom=True
    )

# 3. Головна функція стабільного оновлення мапи всередині Output-контейнера
def render_stable_map(out_widget, map_obj):
    with out_widget:
        clear_output(wait=True)
        display(HTML(f'<div style="width:100%; height:300px;">{map_obj._repr_html_()}</div>'))

# 1. Expanded neighboring suburbs and signalized intersections data (including Berwyn & Lyons)
regions_data = {
    "Riverside": [
        {"name": "Longcommon Rd & E Burlington St", "lat": 41.8295, "lon": -87.8182},
        {"name": "Riverside Rd & Bloomingbank Rd", "lat": 41.8312, "lon": -87.8225},
        {"name": "First Ave & 31st St Intersection", "lat": 41.8385, "lon": -87.8341},
        {"name": "Harlem Ave & Delaplaine Rd", "lat": 41.8260, "lon": -87.8052},
        {"name": "Longcommon Rd & Akenside Rd", "lat": 41.8340, "lon": -87.8170},
        {"name": "1st Ave & Forest Ave", "lat": 41.8335, "lon": -87.8338}
    ],
    "Brookfield": [
        {"name": "Ogden Ave & Grand Blvd", "lat": 41.8235, "lon": -87.8440},
        {"name": "Custer Ave & 47th St", "lat": 41.8080, "lon": -87.8490},
        {"name": "Brookfield Ave & 31st St", "lat": 41.8350, "lon": -87.8420},
        {"name": "8th Ave & 31st St", "lat": 41.8362, "lon": -87.8385},
        {"name": "Hollywood Ave & 31st St", "lat": 41.8341, "lon": -87.8280},
        {"name": "Ogden Ave & Maple Ave", "lat": 41.8210, "lon": -87.8390},
        {"name": "47th St & Grand Blvd", "lat": 41.8075, "lon": -87.8435}
    ],
    "North Riverside": [
        {"name": "Cermak Rd & Harlem Ave", "lat": 41.8520, "lon": -87.8050},
        {"name": "Desplaines Ave & 26th St", "lat": 41.8400, "lon": -87.8150},
        {"name": "Appletree Ln & 9th Ave", "lat": 41.8480, "lon": -87.8180},
        {"name": "Cermak Rd & Desplaines Ave", "lat": 41.8515, "lon": -87.8145},
        {"name": "25th St & Harlem Ave", "lat": 41.8440, "lon": -87.8050},
        {"name": "17th St & Harlem Ave", "lat": 41.8600, "lon": -87.8050},
        {"name": "9th Ave & 26th St", "lat": 41.8410, "lon": -87.8185}
    ],
    "La Grange": [
        {"name": "La Grange Rd & Burlington Ave", "lat": 41.8150, "lon": -87.8710},
        {"name": "Ogden Ave & Kensington Ave", "lat": 41.8190, "lon": -87.8630},
        {"name": "Wabash Ave & 47th St", "lat": 41.8090, "lon": -87.8680},
        {"name": "La Grange Rd & 47th St", "lat": 41.8065, "lon": -87.8715},
        {"name": "La Grange Rd & 31st St", "lat": 41.8355, "lon": -87.8705},
        {"name": "Cossitt Ave & La Grange Rd", "lat": 41.8175, "lon": -87.8710},
        {"name": "55th St & La Grange Rd", "lat": 41.7920, "lon": -87.8720}
    ],
    "Berwyn": [
        {"name": "Cermak Rd & Ridgeland Ave", "lat": 41.8518, "lon": -87.7842},
        {"name": "Ogden Ave & Ridgeland Ave", "lat": 41.8230, "lon": -87.7840},
        {"name": "Cermak Rd & Oak Park Ave", "lat": 41.8518, "lon": -87.7925},
        {"name": "16th St & Ridgeland Ave", "lat": 41.8590, "lon": -87.7840},
        {"name": "26th St & Ridgeland Ave", "lat": 41.8440, "lon": -87.7840},
        {"name": "Cermak Rd & East Ave", "lat": 41.8518, "lon": -87.7710}
    ],
    "Lyons": [
        {"name": "Ogden Ave & 1st Ave", "lat": 41.8195, "lon": -87.8340},
        {"name": "Joliet Ave & 47th St", "lat": 41.8085, "lon": -87.8250},
        {"name": "Cermak Rd & 1st Ave", "lat": 41.8520, "lon": -87.8340},
        {"name": "Joliet Ave & Ogden Ave", "lat": 41.8205, "lon": -87.8280},
        {"name": "Joliet Ave & 43rd St", "lat": 41.8140, "lon": -87.8265}
    ]
}

# Generate synthetic dataset
np.random.seed(42)
n_samples = 4500
date_start = datetime.date(2025, 1, 1)
date_end = datetime.date(2026, 12, 31)
delta_days = (date_end - date_start).days

all_rows = []
for _ in range(n_samples):
    reg_name = np.random.choice(list(regions_data.keys()))
    loc = np.random.choice(regions_data[reg_name])
    random_days = np.random.randint(0, delta_days)
    record_date = date_start + datetime.timedelta(days=random_days)

    hour = np.random.randint(0, 24)
    day_of_week = record_date.weekday()
    temperature = np.random.normal(15, 10)
    precipitation = np.random.choice([0.0, 0.5, 2.5, 10.0], p=[0.7, 0.2, 0.08, 0.02])
    traffic_flow = np.random.randint(100, 1500)
    is_roadwork = np.random.choice([0, 1], p=[0.85, 0.15])

    risk_score = (
        (1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0.5) * 0.3 +
        (1.2 if day_of_week < 5 else 0.8) * 0.1 +
        (1.4 if precipitation > 2.0 else 1.0) * 0.2 +
        (traffic_flow / 1000.0) * 0.3 +
        (1.5 if is_roadwork == 1 else 1.0) * 0.1
    )
    risk_score += np.random.normal(0, 0.1)
    target = 1 if risk_score > 1.15 else 0

    all_rows.append({
        'region': reg_name,
        'date': record_date,
        'location_name': loc['name'],
        'lat': loc['lat'],
        'lon': loc['lon'],
        'hour': hour,
        'day_of_week': day_of_week,
        'temperature': round(temperature, 1),
        'precipitation': precipitation,
        'traffic_flow': traffic_flow,
        'is_roadwork': is_roadwork,
        'target_incident': target
    })

df = pd.DataFrame(all_rows)

# 3. Interactive prediction and update logic
def update_dashboard(region, algorithm, period, intersection_idx, hour, day_of_week, temperature, precipitation, traffic_flow, is_roadwork):
    region_df = df[df['region'] == region]

    if period == '2025':
        filtered_df = region_df[region_df['date'].apply(lambda d: d.year == 2025)]
    elif period == '2026':
        filtered_df = region_df[region_df['date'].apply(lambda d: d.year == 2026)]
    elif period == 'WINTER':
        filtered_df = region_df[region_df['date'].apply(lambda d: d.month in [12, 1, 2])]
    elif period == 'SUMMER':
        filtered_df = region_df[region_df['date'].apply(lambda d: d.month in [6, 7, 8])]
    else:
        filtered_df = region_df.copy()

    if len(filtered_df) < 30:
        filtered_df = region_df.copy()

    features = ['hour', 'day_of_week', 'temperature', 'precipitation', 'traffic_flow', 'is_roadwork']
    X = filtered_df[features]
    y = filtered_df['target_incident']

    algo_names = {'XGB': 'XGBoost', 'RF': 'Random Forest', 'LR': 'Logistic Regression'}
    algo_name = algo_names[algorithm]

    if algorithm == 'XGB':
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    elif algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

    model.fit(X, y)

    locs_list = regions_data[region]
    if intersection_idx >= len(locs_list):
        intersection_idx = 0
    selected_loc = locs_list[intersection_idx]

    input_data = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'temperature': temperature,
        'precipitation': precipitation,
        'traffic_flow': traffic_flow,
        'is_roadwork': int(is_roadwork)
    }])

    probability = model.predict_proba(input_data)[0][1] * 100

    period_labels = {'ALL': 'All Periods (2025-2026)', '2025': 'Year 2025 Only', '2026': 'Year 2026 Only', 'WINTER': 'Winter Season', 'SUMMER': 'Summer Season'}
    report_html = f"""
    <div style="background-color: #f8f9fa; padding: 8px; border-left: 4px solid #007bff; border-radius: 4px; font-family: sans-serif;">
        <h5 style="margin: 0 0 4px 0; color: #333; font-size: 13px;">📊 AI Traffic Analysis Report</h5>
        <ul style="margin: 0; padding-left: 16px; color: #444; font-size: 12px;">
            <li><b>Suburb:</b> {region}</li>
            <li><b>Intersection:</b> {selected_loc['name']}</li>
            <li><b>Method:</b> <code>{algo_name}</code></li>
            <li><b>Period:</b> {period_labels[period]} ({len(filtered_df)} rec.)</li>
            <li><b>Probability:</b> <span style="color: {'red' if probability > 60 else ('orange' if probability > 30 else 'green')}; font-weight: bold;">{probability:.1f}%</span></li>
        </ul>
    </div>
    """
    with out_report:
        clear_output(wait=True)
        display(HTML(report_html))

    # Create map using user's custom function with larger font popups
    m = create_interactive_map(location=[selected_loc['lat'], selected_loc['lon']], zoom=14)

    for i, loc in enumerate(locs_list):
        p = probability if i == intersection_idx else np.random.uniform(15, 45)
        color = 'red' if p > 60 else ('orange' if p > 30 else 'green')

        # Larger font styling for popup and tooltip
        popup_html = f"""
        <div style="font-size: 14px; font-family: sans-serif; line-height: 1.4; min-width: 160px;">
            <b>{loc['name']}</b><br>
            <b>Method:</b> {algo_name}<br>
            <b>Risk:</b> <span style="color: {color}; font-weight: bold;">{p:.1f}%</span>
        </div>
        """

        folium.CircleMarker(
            location=[loc['lat'], loc['lon']],
            radius=9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{loc['name']} (Risk: {p:.1f}%)",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85
        ).add_to(m)

    render_stable_map(out_map, m)

# 4. Compact Widget Controls Setup (Left-aligned)
widget_width = '320px'
desc_width = '120px'

region_dropdown = widgets.Dropdown(options=list(regions_data.keys()), value='Riverside', description='Target Suburb:', style={'description_width': desc_width}, layout={'width': widget_width})
model_dropdown = widgets.Dropdown(options=[('XGBoost', 'XGB'), ('Random Forest', 'RF'), ('Logistic Regression', 'LR')], value='XGB', description='AI Algorithm:', style={'description_width': desc_width}, layout={'width': widget_width})
period_dropdown = widgets.Dropdown(options=[('All Periods (2025-2026)', 'ALL'), ('Year 2025 Only', '2025'), ('Year 2026 Only', '2026'), ('Winter Season', 'WINTER'), ('Summer Season', 'SUMMER')], value='ALL', description='Period:', style={'description_width': desc_width}, layout={'width': widget_width})

intersection_dropdown = widgets.Dropdown(options=[(l['name'], i) for i, l in enumerate(regions_data['Riverside'])], value=0, description='Intersection:', style={'description_width': desc_width}, layout={'width': widget_width})

def update_intersection_options(change):
    selected_reg = region_dropdown.value
    locs = regions_data[selected_reg]
    intersection_dropdown.options = [(l['name'], i) for i, l in enumerate(locs)]
    intersection_dropdown.value = 0

region_dropdown.observe(update_intersection_options, names='value')

hour_slider = widgets.IntSlider(value=8, min=0, max=23, step=1, description='Hour of Day:', style={'description_width': desc_width}, layout={'width': widget_width})
day_dropdown = widgets.Dropdown(options=[('Monday', 0), ('Tuesday', 1), ('Wednesday', 2), ('Thursday', 3), ('Friday', 4), ('Saturday', 5), ('Sunday', 6)], value=0, description='Day of Week:', style={'description_width': desc_width}, layout={'width': widget_width})
temp_slider = widgets.FloatSlider(value=18.0, min=-15.0, max=35.0, step=0.5, description='Temp (°C):', style={'description_width': desc_width}, layout={'width': widget_width})
precip_dropdown = widgets.Dropdown(options=[('None', 0.0), ('Light Rain', 0.5), ('Heavy Rain', 2.5), ('Storm / Snow', 10.0)], value=0.0, description='Precipitation:', style={'description_width': desc_width}, layout={'width': widget_width})
flow_slider = widgets.IntSlider(value=800, min=100, max=1500, step=50, description='Traffic (veh/h):', style={'description_width': desc_width}, layout={'width': widget_width})
roadwork_checkbox = widgets.Checkbox(value=False, description='Roadwork Active', layout={'width': widget_width})

# 5. Connect widgets using interactive_output
out_dashboard = widgets.interactive_output(
    update_dashboard,
    {
        'region': region_dropdown,
        'algorithm': model_dropdown,
        'period': period_dropdown,
        'intersection_idx': intersection_dropdown,
        'hour': hour_slider,
        'day_of_week': day_dropdown,
        'temperature': temp_slider,
        'precipitation': precip_dropdown,
        'traffic_flow': flow_slider,
        'is_roadwork': roadwork_checkbox
    }
)

# Trigger initial calculation
update_dashboard(
    region_dropdown.value, model_dropdown.value, period_dropdown.value,
    intersection_dropdown.value, hour_slider.value, day_dropdown.value,
    temp_slider.value, precip_dropdown.value, flow_slider.value, roadwork_checkbox.value
)

# 6. Compact Layout Design: Left sidebar (Controls + Report), Right side (Map)
left_controls = widgets.VBox([
    widgets.HTML("<b style='color:#2c3e50;'>⚙️ Control Panel</b>"),
    region_dropdown,
    model_dropdown,
    period_dropdown,
    intersection_dropdown,
    hour_slider,
    day_dropdown,
    temp_slider,
    precip_dropdown,
    flow_slider,
    roadwork_checkbox,
    widgets.HTML("<hr style='margin: 8px 0;'>"),
    out_report
], layout={'width': '345px', 'padding': '5px', 'border_right': '1px solid #ddd'})

right_display = widgets.VBox([
    widgets.HTML("<b style='color:#2c3e50;'>🗺️ Hotspot Map</b>"),
    out_map
], layout={'width': 'calc(100% - 350px)', 'padding': '5px'})

main_layout = widgets.HBox([left_controls, right_display], layout={'width': '100%'})

display(widgets.VBox([
    widgets.HTML("<h3 style='margin: 0 0 10px 0; color: #2c3e50;'>Traffic Hotspot Predictor</h3>"),
    main_layout
]))
