# Functional Specification

## Background 🌃
Small businesses in New York City face high uncertainty and risk of failure due to factors such as location, surrounding business ecosystem, neighborhood trends, and regulatory environment. Entrepreneurs, investors, planners, and citizens often lack a unified, data-driven tool to evaluate whether a business is likely to succeed in a given location.  

This system addresses the problem by integrating datasets from NYC Open Data to visualize current businesses, historical trends, and predicted survival outcomes. The platform enables users to explore business ecosystems spatially, understand neighborhood-level dynamics, and make informed decisions about opening, supporting, funding, or leasing businesses.

---

## User Profile 

### Entrepreneur 👩‍💼
- **Who they are**: Business owners or prospective founders evaluating potential business locations.
- **Wants**: Assess whether their business is likely to succeed based on location and surrounding businesses.
- **Needs**: The tool must provide location-based business survival predictions and surrounding ecosystem analysis to help evaluate whether a business idea is viable in a specific area.
- **Domain knowledge**: Economics and business fundamentals; may understand market competition and demand.
- **Technical knowledge**: Can browse the web and has a basic understanding of how to use dashboards.
- **Interaction methods**: Search for their own business and nearby businesses, allowing them to anticipate potential outcomes and plan accordingly. Create a hypothetical business within the tool to simulate performance under different conditions.

### Urban Planning Technician 👷‍♂️
- **Who they are**: City analysts and planners responsible for monitoring commercial health and economic activity across NYC neighborhoods.  
- **Wants**: An overview of business ecosystems across boroughs, early detection of commercial decline, and identification of emerging areas of growth.  
- **Needs**: The system must provide an interactive spatial map, historical and predictive analytics, and filtering tools to analyze business activity by category, location, and time.  
- **Domain knowledge**: Urban planning, spatial analysis, economic development, and public policy.  
- **Technical knowledge**: Highly technical; comfortable using analytical dashboards and data filters.  
- **Interaction methods**: Uses map-based visualization, applies filters by borough/category/time, examines historical vs. predicted trends, and compares neighborhoods.

### Average NYC Citizen 🧒
- **Who they are**: Residents, customers, or community members interested in the survival of local businesses.  
- **Wants**: To check whether a specific business they care about is likely to survive and whether they should continue supporting it.  
- **Needs**: The system must provide a simple search interface, clear business survival outlook, and easy-to-understand visual indicators without requiring technical knowledge.  
- **Domain knowledge**: Minimal business or statistical knowledge.  
- **Technical knowledge**: Basic web usage only.  
- **Interaction methods**: Searches for a business by name, views its status and forecast, and observes its location on a map with simple visual cues.

### Property Owner 🏢
- **Who they are**: Commercial landlords or property holders deciding which type of business to lease space to.  
- **Wants**: To determine which business categories historically perform best in a specific location to maximize tenant success and property value.  
- **Needs**: The system must provide ranking of business categories by historical success, geographic performance comparison, and location-based insights.  
- **Domain knowledge**: Familiar with business categories, leasing considerations, and local market demand.  
- **Technical knowledge**: Moderate; comfortable with structured dashboards and ranking tables.  
- **Interaction methods**: Selects geographic area, compares business categories, views ranked performance metrics, and explores supporting data.

### Bank Manager / Venture Capitalist 🏦
- **Who they are**: Financial professionals evaluating whether to fund, lend to, or invest in a business.  
- **Wants**: To assess risk and likelihood of business success before making financial decisions.  
- **Needs**: The system must provide survival probability metrics, risk indicators, portfolio-level analysis, and drill-down capabilities for individual businesses.  
- **Domain knowledge**: Finance, investment analysis, risk modeling, and statistical reasoning.  
- **Technical knowledge**: Comfortable with analytical dashboards, filtering systems, and data-driven decision tools.  
- **Interaction methods**: Filters businesses by location/category/status, analyzes risk indicators, drills down to business-level data, and explores scenario-based projections.

---

## Data Sources

### [NYC Issued Business Licenses](https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data)
- **Structure**: Tabular dataset (~68K rows, 31 columns), where each row represents a licensed business and its status.
- **Key fields include**:
    - Business identifiers (ID, license number, business name, business category)
    - License status (active, expired, closed)
    - Dates (initial issuance date, expiration date)
    - Location (borough, ZIP, latitude/longitude, census tract)
    - Address and contact information

### [NYC 311 Service Requests (2020-present)](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data)
- **Structure:** Large tabular dataset (~20M rows, 44 columns), where each row represents a service request.  
- **Key fields include:**  
    - Complaint type and descriptor  
    - Status and resolution information  
    - Timestamps (created date, closed date)  
    - Agency responding  
    - Location (latitude/longitude, borough, ZIP, community district)  

The NYC Issued Business Licenses dataset provides core information about businesses, including category, license status, operational history, and precise geographic location, serving as the foundation for identifying whether businesses are active, closed, or at risk. The NYC 311 Service Requests dataset captures environmental and neighborhood conditions, such as noise, sanitation, infrastructure, and safety complaints, that may influence business performance and local commercial health. Together, these datasets enable the system to analyze how surrounding conditions correlate with business survival. Because both datasets include geocoded latitude and longitude coordinates, they can be spatially joined to link each business with nearby environmental signals, allowing location-based analytics and predictive modeling.

---

## Use Cases 🔧

### Use Case 1: Search for a Specific Establishment
**Objective:**  
User wants to view detailed information and survival outlook for a specific NYC business.

**Interaction Flow:**  
1. User enters a business name into the search bar.  
2. System provides autocomplete suggestions based on known establishments.  
3. User selects a suggestion or presses Enter.  
4. System displays:
   - Business details (category, license status, location)
   - Predicted survival outlook
   - Map highlighting the business location and nearby establishments  
5. User may explore surrounding businesses for comparison.

---

### Use Case 2: Explore Surrounding Businesses via Map
**Objective:**  
User wants to understand the local business ecosystem around a location.

**Interaction Flow:**  
1. User hovers over a business pin on the map.  
2. System shows a pop-up card containing:
   - Business name and category  
   - License status (active/closed)  
   - Key summary indicators  
3. User saves the business to Favorites.  
4. System adds the business to a saved list and visually marks it.

---

### Use Case 3: Search by Business Type
**Objective:**  
User wants to explore spatial distribution and performance of a business category.

**Interaction Flow:**  
1. User enters a category (e.g., "Barbershop") in the filter panel.  
2. System suggests known categories while typing.  
3. User selects a category and chooses additional filters (e.g., location, years active, size).  
4. System updates the map and dashboard to show matching businesses.  
5. User downloads the visualization (image or PDF).

---

### Use Case 4: Rank Business Categories for a Location
**Objective:**  
Property owner wants to determine which business types are most successful in a given area.

**Interaction Flow:**  
1. User selects a geographic area and a set of business categories.  
2. System computes success metrics using historical data.  
3. System returns a ranked list of categories based on survival and performance indicators.  
4. User may expand to view detailed statistics or compare categories.
