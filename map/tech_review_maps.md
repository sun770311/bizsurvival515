### Background and Use Case
**Problem:** Small business owners and entrepreneurs often lack accessible tools to assess the historical commercial viability of a specific location. Understanding the turnover rate, such as how many businesses have opened and closed at a single address, provides critical insight into foot traffic, neighborhood demand, and potential risks. 

**Use Case:** We need a web-based interactive map where users can view current and historical business locations. The technology must support high-volume data points and, most importantly, **dynamic client-side filtering**. A user must be able to type a keyword (e.g., "Barbershop") into a search bar, or use dropdowns to filter by region or establishment type, and see the map instantly update to reflect only those specific points. The map must also support detailed popups showing the history and status (open/closed) of the establishment.

### Python Libraries Evaluated

**1. Plotly (with Dash)**
* **Author:** Plotly Technologies Inc.
* **Summary:** Plotly is a declarative graphing library. When paired with Dash (its Python web framework counterpart), it allows developers to build complex, interactive web applications with search bars and dropdowns using entirely Python. It relies on a server-side architecture where user interactions trigger Python callbacks that update the data and push new map renders to the browser.

**2. Flask (paired with Mapbox GL JS)**
* **Author:** Armin Ronacher (Flask) / Mapbox (Mapbox GL JS)
* **Summary:** Flask is a lightweight Python WSGI web application framework. In this architecture, Flask acts as the backend server (mounting environment variables, setting up routes, and serving raw CSV data), while Mapbox GL JS handles the heavy lifting of spatial rendering on the client side. This shifts the mapping logic entirely out of Python and into JavaScript.

### Side-by-Side Comparison

| Feature/Requirement | Plotly + Dash | Flask + Mapbox GL JS |
| :--- | :--- | :--- |
| **Architectural Focus** | Server-side (Python handles data filtering and UI state). | Client-side (JavaScript handles filtering and UI state). |
| **Dynamic Search Bar** | **Moderate.** Keystrokes must be sent to the Python backend to filter the DataFrame, causing slight network latency. | **Excellent.** Type-ahead search is instantaneous because all data is filtered locally in the browser via JavaScript. |
| **Popup Customization** | **Poor.** Limited to hover tooltips. Cannot easily embed complex, scrollable HTML elements (like lists of historical business licenses) inside the map pin popup. | **Excellent.** `mapboxgl.Popup` allows for injection of completely custom, scrollable HTML containers directly tied to map coordinates. |
| **Data Preparation** | **Excellent.** Data loading and cleaning are handled natively in Python using Pandas before rendering. | **Moderate.** Requires writing custom JavaScript (or using a JS library) to parse CSVs and handle missing values, or pre-processing the data before the Flask server runs. |
| **Performance (Large Data)** | **Good.** Can struggle with smooth transitions if filtering requires constantly redrawing thousands of points from the server. | **Exceptional.** Mapbox utilizes WebGL to render and filter tens of thousands of points locally with zero lag. |

### Final Choice
**Flask + Mapbox GL JS.** While Plotly and Dash are incredibly powerful for keeping an entire data science workflow within Python, the specific UI requirements of this project demand a frontend-heavy approach. The need for an instantaneous type-ahead search bar (where points disappear the moment "Barbershop" is typed) and rich, scrollable popups containing historical license data cannot be elegantly achieved with Plotly's standard tooltips or Dash's server-roundtrip callbacks. Using Flask to serve a Mapbox GL JS frontend provides the exact UX "video game" smoothness required for a public-facing commercial real estate tool.

### Drawbacks and Areas of Concern
The primary drawback of choosing the **Flask + Mapbox GL JS** route is the language fragmentation and loss of Python's data ecosystem on the fly. 

Because Flask is relegated to being a simple static file server, we cannot easily use Python libraries like `pandas` or `geopandas` during the user's session. All of the complex data parsing, filtering logic, and state management must be written and maintained in JavaScript. This increases the complexity of the codebase and makes it harder to iterate on data-heavy transformations, as the data must be perfectly pre-processed before the Flask server ever serves it to the browser.