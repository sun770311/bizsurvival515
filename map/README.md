# Business Survival Map

## 1. Purpose of a Map in this Project

The interactive map serves as the central visualization component of this project. It allows users to explore both historical and current business activity in a geographic context. It turns raw licensing and complaint data into an intuitive, location-based decision tool.

Small business owners, entrepreneurs, and analysts often lack accessible ways to assess the commercial viability of a specific location. Examining business turnover patterns can provide valuable insight into:

* Foot traffic and neighborhood demand
* Commercial stability vs volatility
* Local risk signals (e.g., complaints, regulatory activity)
* Historical business patterns at a specific address

This map allows users to:

* View current and historical businesses at exact geographic locations
* Filter dynamically by category, borough, status, and activity signals
* Search by business name or address with instant results
* Inspect detailed popups containing license history, status, and 311 complaint signals

The map must support tens of thousands of data points while maintaining smooth, real-time interaction. Therefore, the architecture prioritizes client-side rendering and filtering.

---

## 2. Running the Map (Mapbox via Python)

The backend uses a lightweight Python server (`serve_map.py`) to:

* Serve static frontend files (`index.html`, `app.js`)
* Expose the dataset (`/data/business_points.csv`)
* Inject the Mapbox public token into the frontend

### Steps to Run

#### 1. Set your Mapbox public token

```bash
export MAPBOX_PUBLIC_TOKEN="pk.XXXX..."
```

#### 2. Start the Python server

```bash
python serve_map.py --port 8000
```

#### 3. Open in browser

```
http://localhost:8000
```

The map will automatically load NYC business points and enable search + filtering.

(Frontend layout, filters, and map UI defined in , and rendering/filtering logic implemented in `app.js`.)

---

## 3. Initial Exploration Using Plotly: Downsides and Limitations

Before implementing Mapbox, the project explored Plotly (with Dash) for interactive mapping.

Plotly provided a fast way to:

* Load and preprocess data in Python using Pandas
* Create scatter maps and interactive plots
* Build a simple UI with dropdowns and filters

However, several limitations became apparent:

### Limitations of Plotly + Dash

**1. Server-Side Filtering Latency**
Each user interaction (typing, selecting filters) requires a round trip to the Python backend, causing:

* Noticeable delay during typing
* Slower filtering with large datasets
* Re-rendering overhead for thousands of points

**2. Limited Popup Customization**
Plotly tooltips are restrictive:

* Cannot embed complex scrollable HTML
* Cannot easily show full license history
* Cannot structure multi-section popup cards

**3. Performance with Large Spatial Data**
Plotly struggles when repeatedly redrawing tens of thousands of points, especially during dynamic filtering.

**4. UI Responsiveness**
Plotly’s architecture is not optimized for instant filtering, which is critical for interactive exploration.

Because of these limitations, Plotly was useful for early exploratory visualization, but not suitable for the final interactive mapping system.

---

## 4. Mapbox GL JS: Why This Approach Is Stronger

The final architecture uses:

* Flask/Python: lightweight static data server
* Mapbox GL JS (JavaScript): client-side rendering and filtering

This design shifts heavy computation to the browser, dramatically improving responsiveness and scalability.

### Key Advantages

#### 1. True Client-Side Filtering (Instant Search)

All business points load once, and filtering happens entirely in JavaScript. This enables:

* Instant type-ahead search
* Zero network delay
* Smooth dynamic filtering of thousands of points

---

#### 2. High-Performance Rendering (WebGL)

Mapbox GL JS uses WebGL GPU rendering, allowing:

* Tens of thousands of points with no lag
* Smooth zoom and pan
* Real-time filtering without redraw delays

This makes the map scalable to production-level datasets.

---

#### 3. Fully Custom Popups

Unlike Plotly, Mapbox allows full HTML popups. The system supports:

* Scrollable popup cards
* Full license history display
* Business status + metadata
* 311 complaint signals
* Structured multi-section layout

---

#### 4. Advanced UI and Filtering Controls

The map includes:

* Instant search bar (business name + address)
* Multi-select filters (borough, category, status, complaint volume)
* Dynamic result counts
* Reset and viewport controls

---

#### 5. Separation of Concerns

* **Python** handles:

  * Serving files
  * Injecting environment variables
  * Providing dataset access

* **JavaScript / Mapbox** handles:

  * Rendering
  * Filtering
  * Search
  * UI state
  * Popups

This leads to better performance and a smoother user experience.

---

## Final Architecture Choice

Flask + Mapbox GL JS was chosen because this project requires:

* Instant filtering with no latency
* High-performance rendering of large spatial datasets
* Rich, structured popup cards
* Scalable, production-grade interactive mapping

Plotly was valuable for early exploration, but Mapbox provides the performance, UX, and flexibility required for a public-facing commercial location intelligence tool.

