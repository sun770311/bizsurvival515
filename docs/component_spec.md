# Component Specification

## Software Components

### 1. Data Manager (`DataManager`)

- **What it does:**  
  The Data Manager is the system’s main interface to the NYC Issued Business Licenses dataset. It retrieves, filters, and formats business records so other parts of the system never need to work with the raw table directly. It supports common application queries such as:
  - autocomplete / lookup by business name,
  - filtering by category, license status, borough, and time range,
  - retrieving businesses within the current map view (bounding box).
  
  It also converts raw rows into clean, standardized Business objects with consistent field names.

- **Inputs (with type information):**
  - `searchText: string (optional)` — text entered by the user (e.g., part of a business name)  
  - `filters: dictionary (optional)` — query constraints such as  
    `{ category, status, borough, dateRange, businessID, boundingBox }`  
  - `limit: integer` — maximum number of results to return

- **Outputs (with type information):**
  - `businessList: list of Business objects` — each contains (at minimum)  
    `{ id: string, name: string, category: string, status: string, lon: float, lat: float }`
  - `metadata: dictionary` — summary info such as `{ totalResults: int, appliedFilters: dict }`

- **Assumptions:**  
  The business license dataset is cleaned and indexed for fast lookup. Some fields (e.g., category or coordinates) may occasionally be missing; the Data Manager returns partial but valid Business objects whenever possible.

---

### 2. Spatial Service (`GeoService`)

- **What it does:**  
  The GeoService handles all location-based operations in the system. Given geographic coordinates and a distance (or a map region), it:
  - finds nearby businesses,
  - finds nearby 311 service requests,
  - produces a spatial join summary linking each business to neighborhood conditions derived from nearby 311 requests (using latitude/longitude proximity).  
  This enables the system to analyze how surrounding neighborhood signals may relate to business survival.

- **Inputs (with type information):**
  - `coordinates: (float, float)` — (lon, lat) for a location (e.g., a hovered pin)  
  - `radius: float` — distance in meters used for “nearby” searches  
  - `boundingBox: (float, float, float, float) (optional)` — *(minLon, minLat, maxLon, maxLat)* for viewport queries  
  - `filters: dictionary (optional)` — constraints such as `{ category, status }` for nearby business filtering  
  - `dateRange: (date, date) (optional)` — restrict 311 requests to a time window

- **Outputs (with type information):**
  - `nearbyBusinesses: list of Business objects` — businesses located within the specified radius or bounding box  
  - `nearbyServiceRequests: list of 311 Request objects` — each contains (at minimum)  
    `{ id: string, type: string, createdDate: date, lon: float, lat: float }`
  - `spatialJoinResult: dictionary` — per-business neighborhood summaries, e.g.  
    `{ business_id: { complaintCount: int, topComplaintTypes: list[string] } }`

- **Assumptions:**  
  Spatial calculations require valid coordinates. Records with missing/invalid lat/lon are excluded from spatial operations. Distance computations are approximate and intended for neighborhood-level analysis (not parcel-level surveying).

---

### 3. Recommendation & Ranking Engine (`AnalyticsEngine`)
- **What it does:**  
  Computes survival-related metrics and rankings, such as:
  - survival likelihood scores for businesses,
  - ranked lists of business categories by “success” within a selected area,
  - lists of similar businesses.  
  It outputs user-facing scores and short explanations.

- **Inputs (with type information):**
  - `businesses: list[Business]`
  - `features: dict` — engineered features derived from business fields + `GeoService.spatialJoinResult` (311 neighborhood summaries)
  - `mode: str` — one of `"survival_score"`, `"category_rank"`, `"similarity"`

- **Outputs (with type information):**
  - `scores: dict` — `{ business_id: float }` (e.g., survival score from 0 to 1)
  - `rankedCategories: list[tuple[str, float]]` — list of (category, score)
  - `explanations: dict` — `{ business_id: list[str] }` (plain-language reasons)

- **Notes/assumptions:**  
  Scores are approximate and depend on data quality and freshness. Missing features are handled with defaults or simple imputation rules.

---

### 4. Visualization Manager (`VizManager`)
- **What it does:**  
  Translates system outputs into an interactive user experience by rendering:
  - map pins and overlays,
  - hover tooltips,
  - charts and ranking tables,
  - downloadable exports (PNG/PDF).

- **Inputs (with type information):**
  - `mapPins: list[MapPin]`
  - `tooltipContent: dict`
  - `chartsData: dict (optional)`
  - `exportFormat: str (optional)` — `"png"` or `"pdf"`

- **Outputs (with type information):**
  - `renderedView: UIState` — visible map layers, panels, and selection/highlight state
  - `exportFile: bytes (optional)` — generated file when user downloads

- **Notes/assumptions:**  
  Requires stable business IDs so selections, highlights, and saved items remain consistent across filtering and map movement.

---

## Interactions to Accomplish Use Cases

> Components used:
> - **DataManager**: fetches/filter business license records (name/category/status/location/time).
> - **GeoService**: spatial queries + joins businesses to nearby 311 requests using lat/lon.
> - **AnalyticsEngine**: computes survival scores, similarity, and category rankings.
> - **VizManager**: renders map pins, tooltips, panels, charts, and exports.

### Use Case 1: User searches for a specific existing establishment in NYC
1. **UI → DataManager:** send `searchText` (partial name) + `limit` to retrieve autocomplete suggestions (`businessList`).
2. **DataManager → UI:** return suggested businesses + `metadata`.
3. **UI → DataManager:** once user selects a business, query by `filters = { business_id: ... }` (or exact name) to retrieve the full business record.
4. **UI → GeoService:** send selected business `coordinates` + `radius` to retrieve:
   - `nearbyBusinesses`
   - `nearbyServiceRequests`
   - `spatialJoinResult` (business ↔ 311 neighborhood summary)
5. **UI → AnalyticsEngine:** compute `mode="survival_score"` using selected business + features derived from `spatialJoinResult`.
6. **AnalyticsEngine → UI:** return `scores` and `explanations` for the selected business (and optionally nearby businesses).
7. **UI → VizManager:** build `mapPins` (selected + nearby), set tooltip/panel content, and render the updated view.

---

### Use Case 2: User hovers over pins on spatial map to explore surrounding businesses
1. **UI → DataManager:** use hovered `business_id` to fetch the hovered business’ summary fields (if not already cached).
2. **UI → GeoService:** send hovered business `coordinates` + `radius` to retrieve `nearbyBusinesses` and `nearbyServiceRequests` (and optionally `spatialJoinResult`).
3. **UI → VizManager:** render hover tooltip and highlight hovered pin; optionally show nearby businesses as a subtle layer.

---

### Use Case 3: User searches for a type of business (filters)
1. **UI → DataManager:** send `filters = { category, status, borough, dateRange }` (+ optional `boundingBox`) and `limit` to retrieve `businessList`.
2. **DataManager → UI:** return filtered businesses + metadata counts.
3. **UI → GeoService:** for the map viewport, retrieve 311 context or compute per-business neighborhood summaries:
   - either join only for businesses currently in view
   - or join only for a sample/aggregation to keep it fast
4. **UI → VizManager:** render pins + a filter summary panel; enable export.
5. **UI → VizManager (export):** on download click, call with `exportFormat="png"` or `"pdf"` and return `exportFile`.

---

### Use Case 4: Property owner selects business categories and gets an ordered list by success in an area
1. **UI → DataManager:** send `filters = { categories: [...], boundingBox OR borough/zip, dateRange }` to retrieve `businessList` in the chosen area.
2. **UI → GeoService:** spatially join the area’s businesses to nearby 311 requests, producing `spatialJoinResult` (neighborhood signals per business).
3. **UI → AnalyticsEngine:** compute `mode="category_rank"` using the businesses + features derived from `spatialJoinResult`.
4. **AnalyticsEngine → UI:** return `rankedCategories` + supporting `explanations` (e.g., why top categories do well).
5. **UI → VizManager:** render a ranked table + optional map overlay for the chosen area.
