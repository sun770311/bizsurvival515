const GEOJSON_URL = "/businesses.geojson";

// NYC default view
const NYC_CENTER = [-73.94, 40.72];
const NYC_ZOOM = 10.5;

const BOROUGH_COLORS = {
  Manhattan: "#d4af37",
  Brooklyn: "#2e8b57",
  "Staten Island": "#dc2626",
  Bronx: "#2563eb",
  Queens: "#7c3aed"
};

const statusEl = document.getElementById("status");
const filterPanelEl = document.getElementById("filterPanel");
const resetFiltersBtn = document.getElementById("resetFilters");

let map;
let allGeoJSON;
let popup;
let currentFilters = {
  boroughs: new Set(),
  active: new Set(),
  complaintBins: new Set(),
  licenseBands: new Set()
};

function setStatus(message) {
  console.log(message);
  if (statusEl) statusEl.textContent = message;
}

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function isMissing(value) {
  if (value === null || value === undefined) return true;
  const normalized = String(value).trim().toLowerCase();
  return normalized === "" || normalized === "nan" || normalized === "none" || normalized === "null";
}

function formatValue(value, fallback = "N/A") {
  if (isMissing(value)) return fallback;
  return escapeHtml(value);
}

function formatActive(active) {
  return Number(active) === 1
    ? '<span class="status-active">Active</span>'
    : '<span class="status-inactive">Inactive</span>';
}

function normalizeBorough(value) {
  if (isMissing(value)) return "Unknown";
  const trimmed = String(value).trim();
  return BOROUGH_COLORS[trimmed] ? trimmed : trimmed;
}

function getComplaintBin(value) {
  const num = Number(value);

  if (!Number.isFinite(num)) return "Unknown";
  if (num === 0) return "0";
  if (num <= 2) return "1-2";
  if (num <= 5) return "3-5";
  if (num <= 10) return "6-10";
  if (num <= 20) return "11-20";
  if (num <= 50) return "21-50";
  if (num <= 100) return "51-100";

  return "101+";
}

function getLicenseBand(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "Unknown";
  return num <= 1 ? "Single license" : "Multiple licenses";
}

function activeLabel(value) {
  return Number(value) === 1 ? "Active" : "Inactive";
}

function buildLicenseRecordHtml(record, index) {
  return `
    <div class="license-record">
      <div><span class="popup-label">License ${index + 1}</span></div>
      <div><span class="popup-label">Business Name:</span> ${formatValue(record.business_name)}</div>
      <div><span class="popup-label">Address:</span> ${formatValue(record.address)}</div>
      <div><span class="popup-label">Borough:</span> ${formatValue(record.borough)}</div>
      <div><span class="popup-label">Phone:</span> ${formatValue(record.contact_phone)}</div>
      <div><span class="popup-label">License Number:</span> ${formatValue(record.license_number)}</div>
      <div><span class="popup-label">License Type:</span> ${formatValue(record.license_type)}</div>
      <div><span class="popup-label">License Status:</span> ${formatValue(record.license_status)}</div>
      <div><span class="popup-label">Business Category:</span> ${formatValue(record.business_category)}</div>
    </div>
  `;
}

function buildPopupHtml(properties) {
  const licenseRecords = Array.isArray(properties.license_records)
    ? properties.license_records
    : [];

  const licensesHtml = licenseRecords.length
    ? licenseRecords.map((record, i) => buildLicenseRecordHtml(record, i)).join("")
    : "<div>No license records available.</div>";

  return `
    <div class="popup-card">
      <h3>${formatValue(properties.business_id)}</h3>

      <div class="popup-section">
        <div><span class="popup-label">Business ID:</span> ${formatValue(properties.business_id)}</div>
        <div><span class="popup-label">Active:</span> ${formatActive(properties.active)}</div>
        <div><span class="popup-label">Last Month:</span> ${formatValue(properties.last_month)}</div>
        <div><span class="popup-label">Complaint Sum:</span> ${formatValue(properties.complaint_sum)}</div>
        <div><span class="popup-label">License Count:</span> ${formatValue(properties.license_count)}</div>
      </div>

      <div class="popup-section">
        <div class="popup-label">License Records</div>
        ${licensesHtml}
      </div>
    </div>
  `;
}

async function loadGeoJSON() {
  if (typeof EMBEDDED_GEOJSON !== "undefined") {
    const data = EMBEDDED_GEOJSON;

    if (!data || data.type !== "FeatureCollection" || !Array.isArray(data.features)) {
      throw new Error("Embedded GeoJSON is not a valid FeatureCollection.");
    }

    let validCount = 0;
    for (const feature of data.features) {
      const coords = feature?.geometry?.coordinates;
      if (
        Array.isArray(coords) &&
        coords.length === 2 &&
        Number.isFinite(coords[0]) &&
        Number.isFinite(coords[1])
      ) {
        validCount += 1;
      }
    }

    if (validCount === 0) {
      throw new Error("Embedded GeoJSON loaded, but no valid point coordinates were found.");
    }

    data.features = data.features.map((feature) => {
      const properties = feature?.properties || {};
      let licenseRecords = properties.license_records;
      if (typeof licenseRecords === "string") {
        try {
          licenseRecords = JSON.parse(licenseRecords);
        } catch {
          licenseRecords = [];
        }
      }
      if (!Array.isArray(licenseRecords)) licenseRecords = [];

      const derivedBorough = normalizeBorough(licenseRecords[0]?.borough || properties.borough);

      return {
        ...feature,
        properties: {
          ...properties,
          borough_display: derivedBorough
        }
      };
    });

    setStatus(`Loaded embedded GeoJSON successfully. Features: ${data.features.length}`);
    return data;
  }

  setStatus(`Fetching ${GEOJSON_URL} ...`);
  const response = await fetch(GEOJSON_URL);

  if (!response.ok) {
    throw new Error(`GeoJSON fetch failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();

  if (!data || data.type !== "FeatureCollection" || !Array.isArray(data.features)) {
    throw new Error("GeoJSON is not a valid FeatureCollection.");
  }

  let validCount = 0;
  for (const feature of data.features) {
    const coords = feature?.geometry?.coordinates;
    if (
      Array.isArray(coords) &&
      coords.length === 2 &&
      Number.isFinite(coords[0]) &&
      Number.isFinite(coords[1])
    ) {
      validCount += 1;
    }
  }

  if (validCount === 0) {
    throw new Error("GeoJSON loaded, but no valid point coordinates were found.");
  }

  data.features = data.features.map((feature) => {
    const properties = feature?.properties || {};
    let licenseRecords = properties.license_records;
    if (typeof licenseRecords === "string") {
      try {
        licenseRecords = JSON.parse(licenseRecords);
      } catch {
        licenseRecords = [];
      }
    }
    if (!Array.isArray(licenseRecords)) licenseRecords = [];

    const derivedBorough = normalizeBorough(licenseRecords[0]?.borough || properties.borough);

    return {
      ...feature,
      properties: {
        ...properties,
        borough_display: derivedBorough
      }
    };
  });

  setStatus(`Loaded GeoJSON successfully. Features: ${data.features.length}`);
  return data;
}

function createMap(token) {
  mapboxgl.accessToken = token;

  return new mapboxgl.Map({
    container: "map",
    style: "mapbox://styles/mapbox/dark-v11",
    center: NYC_CENTER,
    zoom: NYC_ZOOM
  });
}

function getUniqueSortedValues(features, extractor) {
  return Array.from(new Set(features.map(extractor))).sort((a, b) => String(a).localeCompare(String(b)));
}

function buildCheckboxGroup(title, items, groupName, formatter = (v) => v) {
  return `
    <div class="filter-group">
      <div class="filter-title">${escapeHtml(title)}</div>
      <div class="filter-options">
        ${items.map((item) => `
          <label class="filter-option">
            <input type="checkbox" data-filter-group="${escapeHtml(groupName)}" value="${escapeHtml(item)}" />
            <span>${escapeHtml(formatter(item))}</span>
          </label>
        `).join("")}
      </div>
    </div>
  `;
}

function renderFilters(geojson) {
  const features = geojson.features || [];
  const boroughs = getUniqueSortedValues(
    features,
    (f) => normalizeBorough(f.properties?.license_records?.[0]?.borough || f.properties?.borough)
  );
  const activeVals = ["Active", "Inactive"];
  const complaintBins = [
    "0",
    "1-2",
    "3-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "101+"
  ];
  const licenseBands = ["Single license", "Multiple licenses"];

  filterPanelEl.innerHTML = `
    ${buildCheckboxGroup("Borough", boroughs, "boroughs")}
    ${buildCheckboxGroup("Status", activeVals, "active")}
    ${buildCheckboxGroup("Complaint Sum", complaintBins, "complaintBins")}
    ${buildCheckboxGroup("License Count", licenseBands, "licenseBands")}
  `;

  filterPanelEl.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.addEventListener("change", handleFilterChange);
  });
}

function featureMatchesFilterSet(selectedSet, value) {
  return selectedSet.size === 0 || selectedSet.has(value);
}

function filterFeatures(features) {
  return features.filter((feature) => {
    const props = feature.properties || {};

    let licenseRecords = props.license_records;
    if (typeof licenseRecords === "string") {
      try {
        licenseRecords = JSON.parse(licenseRecords);
      } catch {
        licenseRecords = [];
      }
    }
    if (!Array.isArray(licenseRecords)) licenseRecords = [];

    const borough = normalizeBorough(licenseRecords[0]?.borough || props.borough);
    const active = activeLabel(props.active);
    const complaintBin = getComplaintBin(props.complaint_sum);
    const licenseBand = getLicenseBand(props.license_count);

    return (
      featureMatchesFilterSet(currentFilters.boroughs, borough) &&
      featureMatchesFilterSet(currentFilters.active, active) &&
      featureMatchesFilterSet(currentFilters.complaintBins, complaintBin) &&
      featureMatchesFilterSet(currentFilters.licenseBands, licenseBand)
    );
  });
}

function updateMapData() {
  const filteredFeatures = filterFeatures(allGeoJSON.features || []);
  map.getSource("businesses").setData({
    type: "FeatureCollection",
    features: filteredFeatures
  });

  setStatus(`Showing ${filteredFeatures.length} of ${(allGeoJSON.features || []).length} businesses.`);

  if (popup) {
    popup.remove();
    popup = null;
  }
}

function handleFilterChange(event) {
  const checkbox = event.target;
  const group = checkbox.dataset.filterGroup;
  const value = checkbox.value;

  if (!group || !(group in currentFilters)) return;

  if (checkbox.checked) {
    currentFilters[group].add(value);
  } else {
    currentFilters[group].delete(value);
  }

  updateMapData();
}

function resetFilters() {
  currentFilters = {
    boroughs: new Set(),
    active: new Set(),
    complaintBins: new Set(),
    licenseBands: new Set()
  };

  filterPanelEl.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.checked = false;
  });

  updateMapData();
}

async function init() {
  try {
    const token = window.__CONFIG__?.MAPBOX_PUBLIC_TOKEN;

    if (!token || token === "YOUR_MAPBOX_PUBLIC_TOKEN") {
      throw new Error("Mapbox token is missing. Replace YOUR_MAPBOX_PUBLIC_TOKEN in index.html.");
    }

    setStatus("Creating map...");
    map = createMap(token);

    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    map.on("error", (e) => {
      console.error("Mapbox error:", e);
      if (e?.error?.message) {
        setStatus(`Mapbox error: ${e.error.message}`);
      }
    });

    resetFiltersBtn?.addEventListener("click", resetFilters);

    map.on("load", async () => {
      try {
        setStatus("Map loaded. Loading GeoJSON...");
        allGeoJSON = await loadGeoJSON();
        renderFilters(allGeoJSON);

        map.addSource("businesses", {
          type: "geojson",
          data: allGeoJSON
        });

        map.addLayer({
          id: "business-points",
          type: "circle",
          source: "businesses",
          paint: {
            "circle-radius": 3.5,
            "circle-stroke-width": 0.75,
            "circle-stroke-color": "#ffffff",
            "circle-color": [
              "match",
              ["get", "borough_display"],
              "Manhattan", BOROUGH_COLORS.Manhattan,
              "Brooklyn", BOROUGH_COLORS.Brooklyn,
              "Staten Island", BOROUGH_COLORS["Staten Island"],
              "Bronx", BOROUGH_COLORS.Bronx,
              "Queens", BOROUGH_COLORS.Queens,
              "#6b7280"
            ],
            "circle-opacity": 0.85
          }
        });

        updateMapData();

        map.on("click", "business-points", (e) => {
          const feature = e.features?.[0];
          if (!feature) return;

          const coordinates = feature.geometry.coordinates.slice();
          const properties = { ...(feature.properties || {}) };

          if (typeof properties.license_records === "string") {
            try {
              properties.license_records = JSON.parse(properties.license_records);
            } catch (err) {
              properties.license_records = [];
            }
          }

          if (popup) popup.remove();
          popup = new mapboxgl.Popup({ offset: 12, maxWidth: "420px" })
            .setLngLat(coordinates)
            .setHTML(buildPopupHtml(properties))
            .addTo(map);
        });

        map.on("mouseenter", "business-points", () => {
          map.getCanvas().style.cursor = "pointer";
        });

        map.on("mouseleave", "business-points", () => {
          map.getCanvas().style.cursor = "";
        });

        setStatus("Map loaded successfully.");
      } catch (err) {
        console.error(err);
        setStatus(`Error after map load: ${err.message}`);
      }
    });
  } catch (err) {
    console.error(err);
    setStatus(`Initialization error: ${err.message}`);
  }
}

init();
