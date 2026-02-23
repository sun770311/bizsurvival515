// app.js
// Renders one point per business unique ID.
//  A single business can hold multiple licenses (different categories, types).
//
// When a business is selected, display a popup with a scrollable card.
//
// Popup card display order:
//   1. Business Name
//   2. Address (Building Number, Street1, Street2, Street3, Unit Type, Apt/Suite, 
//               City, State, ZIP Code)
//   3. Borough, Business Unique ID, Contact Phone, Business Status (derived)
//   4. Licenses (licenses_json: License Number, Business Category, License Type, 
//                License Status, Initial Issuance Date, Expiration Date)
//   5. 2025 (for now) 311 Service Reports: total_311, problem_counts_json 
//        (e.g. {"Problem A": 12, ...})
// 
// Map position: Latitude, Longitude


const CSV_URL = "/data/business_points.csv";

// NYC viewport 
const NYC_CENTER = [-74.0060, 40.7128]; // [lng, lat]
const NYC_ZOOM = 11.2;
const NYC_BBOX = {
  west: -74.30,
  south: 40.45,
  east: -73.65,
  north: 40.95
};

// Show data source in the overlay
document.getElementById("csvPath").textContent = CSV_URL;

const token = window.__CONFIG__?.MAPBOX_PUBLIC_TOKEN;
if (!token) {
  document.getElementById("status").textContent =
    "Missing MAPBOX_PUBLIC_TOKEN. Check server env var.";
  throw new Error("Missing token");
}

mapboxgl.accessToken = token;

/*
Parse CSV text into an array of row objects keyed by header names.
Handles quoted fields and escaped quotes ("").
*/
function parseCSV(text) {
  const rows = [];
  let row = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const n = text[i + 1];

    if (c === '"' && inQuotes && n === '"') { cur += '"'; i++; }
    else if (c === '"') { inQuotes = !inQuotes; }
    else if (c === "," && !inQuotes) { row.push(cur); cur = ""; }
    else if ((c === "\n" || c === "\r") && !inQuotes) {
      if (c === "\r" && n === "\n") i++;
      row.push(cur);
      rows.push(row);
      row = [];
      cur = "";
    } else {
      cur += c;
    }
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }

  while (rows.length && rows[rows.length - 1].every(v => v.trim() === "")) rows.pop();

  const header = rows.shift().map(h => h.trim());
  return rows
    .filter(r => r.some(v => v.trim() !== ""))
    .map(r => {
      const obj = {};
      header.forEach((h, idx) => obj[h] = (r[idx] ?? "").trim());
      return obj;
    });
}

// Parse a value as a number; returns null if not finite (e.g. empty or NaN).
function safeNum(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

// True if (lat, lng) falls within the NYC bounding box. 
function inNYC(lat, lng) {
  if (lat == null || lng == null) return false;
  if (lat < NYC_BBOX.south || lat > NYC_BBOX.north) return false;
  if (lng < NYC_BBOX.west || lng > NYC_BBOX.east) return false;
  return true;
}

// Escape a string for safe use in HTML
function escHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Parse a string as JSON; returns a plain object or null if invalid/not an object
function parseJsonObjectMaybe(raw) {
  if (!raw) return null;
  try {
    const obj = JSON.parse(raw);
    return obj && typeof obj === "object" && !Array.isArray(obj) ? obj : null;
  } catch {
    return null;
  }
}

// Parse a string as JSON; returns an array or null if invalid/not an array
function parseJsonArrayMaybe(raw) {
  if (!raw) return null;
  try {
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : null;
  } catch {
    return null;
  }
}

// Extract 311 problem-type counts from CSV row. Returns object { "Problem A": count, ... }.
function parseProblemCounts(props) {
  const raw = props["problem_counts_json"] || props["problemCountsJson"] || props["problem_counts"] || "";
  return parseJsonObjectMaybe(raw) || {};
}

// Extract licenses array from CSV row. Returns array of license objects.
function parseLicenses(props) {
  const raw = props["licenses_json"] || props["licensesJson"] || "";
  const arr = parseJsonArrayMaybe(raw);
  return arr ? arr.filter(x => x && typeof x === "object") : [];
}

// Join non-empty trimmed strings with separator; used for address and license text. 
function joinNonEmpty(parts, sep = " ") {
  return parts
    .map(x => String(x ?? "").trim())
    .filter(Boolean)
    .join(sep);
}

// Build address lines from CSV row (Building Number, Street1–3, Unit, City/State/ZIP). 
function buildAddress(p) {
  const streetLine = joinNonEmpty([
    p["Building Number"],
    p["Street1"],
    p["Street2"],
    p["Street3"]
  ], " ");

  const unitLine = joinNonEmpty([p["Unit Type"], p["Apt/Suite"]], " ");

  const cityStateZip = joinNonEmpty([p["City"], p["State"], p["ZIP Code"]], ", ")
    .replace(", ,", ",");

  const lines = [];
  if (streetLine) lines.push(streetLine);
  if (unitLine) lines.push(unitLine);
  if (cityStateZip) lines.push(cityStateZip);

  return lines;
}

// Single-line address string for search display and popup.
function fullAddressString(p) {
  return buildAddress(p).join(" ");
}

// Format 311 problem counts as HTML for popup
function formatProblemCounts(problemCounts, limit = 50, includeSignalsHeading = true) {
  const entries = Object.entries(problemCounts || {})
    .map(([k, v]) => [k, Number(v)])
    .filter(([_, v]) => Number.isFinite(v) && v > 0)
    .sort((a, b) => b[1] - a[1]);

  if (!entries.length) return "";

  const shown = entries.slice(0, limit);
  const extra = entries.length - shown.length;

  const lines = shown.map(([k, v]) => `${escHtml(k)}: ${escHtml(String(v))}`).join("<br/>");
  const more = extra > 0 ? `<div style="margin-top:6px;opacity:0.8;">+ ${extra} more…</div>` : "";
  const signalsHeader = includeSignalsHeading ? "<strong>311 signals</strong><br/>" : "";

  return `
    <div style="margin-top:8px;">
      ${signalsHeader}
      ${lines}
      ${more}
    </div>
  `;
}

// From a licenses array, get the latest (most recent forward in time) expiration date and return its year, or null if none parseable.
function getLatestExpirationYear(licenses) {
  if (!licenses || !licenses.length) return null;
  let latest = null;
  for (const lic of licenses) {
    const raw = lic["Expiration Date"] ?? "";
    if (!raw) continue;
    const d = parseExpirationDate(raw);
    if (d && (latest === null || d > latest)) latest = d;
  }
  return latest ? latest.getFullYear() : null;
}

// Parse common date strings (e.g. MM/DD/YYYY, YYYY-MM-DD) and return Date or null.
function parseExpirationDate(s) {
  const str = String(s ?? "").trim();
  if (!str) return null;
  const mdy = str.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (mdy) {
    const month = parseInt(mdy[1], 10) - 1;
    const day = parseInt(mdy[2], 10);
    const year = parseInt(mdy[3], 10);
    const d = new Date(year, month, day);
    if (Number.isFinite(d.getTime()) && d.getMonth() === month && d.getDate() === day) return d;
  }
  const iso = str.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (iso) {
    const d = new Date(parseInt(iso[1], 10), parseInt(iso[2], 10) - 1, parseInt(iso[3], 10));
    if (Number.isFinite(d.getTime())) return d;
  }
  const yearOnly = str.match(/\b(19|20)\d{2}\b/);
  if (yearOnly) {
    const y = parseInt(yearOnly[0], 10);
    if (y >= 1900 && y <= 2100) return new Date(y, 11, 31);
  }
  return null;
}

// Format licenses array as HTML for popup, limited to first 10.
function formatLicenses(licenses, limit = 10) {
  if (!licenses || !licenses.length) return "";

  const shown = licenses.slice(0, limit);
  const extra = licenses.length - shown.length;

  const rows = shown.map((lic) => {
    const num = lic["License Number"] ?? "";
    const cat = lic["Business Category"] ?? "";
    const typ = lic["License Type"] ?? "";
    const stat = lic["License Status"] ?? "";
    const iss = lic["Initial Issuance Date"] ?? "";
    const exp = lic["Expiration Date"] ?? "";

    const header = joinNonEmpty([num, typ], " — ");
    const sub = joinNonEmpty([cat], " ");
    const dates = joinNonEmpty(
      [iss ? `Issued: ${iss}` : "", exp ? `Expires: ${exp}` : ""],
      " • "
    );
    const status = stat ? `Status: ${stat}` : "";

    const detailParts = [sub, status, dates].filter(Boolean);

    return `
      <div style="margin-top:6px;">
        <div><strong>${escHtml(header || "(License)")}</strong></div>
        ${detailParts.length ? `<div style="opacity:0.92;margin-top:2px;">${detailParts.map(escHtml).join("<br/>")}</div>` : ""}
      </div>
    `;
  }).join("");

  const more = extra > 0 ? `<div style="margin-top:6px;opacity:0.8;">+ ${extra} more license(s)…</div>` : "";

  return `
    <div style="margin-top:8px;">
      <strong style="display:block;padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.08);">Licenses</strong>
      ${rows}
      ${more}
    </div>
  `;
}

// Make Pop-up card scrollable
function makePopupScrollable(popup) {
  const el = popup.getElement();
  if (!el) return;

  // Ensure popup content itself can receive wheel events
  el.style.pointerEvents = "auto";

  // Stop wheel + touchmove from propagating to the map
  const stop = (e) => {
    e.stopPropagation();
  };

  // Use capture to intercept before Mapbox handlers
  el.addEventListener("wheel", stop, { passive: true, capture: true });
  el.addEventListener("touchmove", stop, { passive: true, capture: true });
  el.addEventListener("mousewheel", stop, { passive: true, capture: true });
}

/*
Create map, load CSV, build GeoJSON, wire filters and search, show points
*/
async function main() {
  const statusEl = document.getElementById("status");

  const map = new mapboxgl.Map({
    container: "map",
    style: "mapbox://styles/mapbox/streets-v12",
    center: NYC_CENTER,
    zoom: NYC_ZOOM
  });

  map.addControl(new mapboxgl.NavigationControl(), "top-right");

  let rows;
  try {
    const resp = await fetch(CSV_URL, { cache: "no-store" });
    if (!resp.ok) throw new Error(`Failed to fetch CSV: ${resp.status}`);
    rows = parseCSV(await resp.text());
  } catch (e) {
    statusEl.textContent = "Error loading CSV. Check console.";
    console.error(e);
    return;
  }

  const features = [];
  let droppedMissing = 0;
  let droppedOutOfNYC = 0;

  for (const r of rows) {
    const lat = safeNum(r["Latitude"]);
    const lng = safeNum(r["Longitude"]);

    if (lat === null || lng === null) { droppedMissing++; continue; }
    if (!inNYC(lat, lng)) { droppedOutOfNYC++; continue; }

    const problemCounts = parseProblemCounts(r);
    const licenses = parseLicenses(r);

    const total311 =
      safeNum(r["total_311"]) ??
      safeNum(r["total311"]) ??
      Object.values(problemCounts).reduce((acc, v) => acc + (Number(v) || 0), 0);

    // Bucket total 311 count for the "Total 311 requests" filter
    const n = Number(total311 ?? 0);
    let bucket311 = "26+";
    if (n <= 0) bucket311 = "0";
    else if (n <= 5) bucket311 = "1-5";
    else if (n <= 10) bucket311 = "6-10";
    else if (n <= 25) bucket311 = "11-25";

    // Concatenated name + address for search (case-insensitive)
    const name = (r["Business Name"] || "").trim();
    const addrStr = fullAddressString(r);
    const searchText = [name, addrStr].filter(Boolean).join(" ").toLowerCase();

    const rawBorough = (r["Borough"] || "").trim();
    const borough = rawBorough === "Outside NYC" ? "Not Recorded" : rawBorough;

    const props = {
      ...r,
      Borough: borough,
      problemCountsJson: JSON.stringify(problemCounts),
      licensesJson: JSON.stringify(licenses),
      total311: String(total311 ?? 0),
      _searchText: searchText,
      _311Bucket: bucket311
    };

    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: [lng, lat] },
      properties: props
    });
  }

  const fc = { type: "FeatureCollection", features };
  statusEl.textContent =
    `Loaded ${features.length} NYC business point(s). `;

  // Collect unique values for filter checkboxes from all features
  const boroughSet = new Set();
  const statusSet = new Set();
  const categorySet = new Set();
  const buckets311 = ["0", "1-5", "6-10", "11-25", "26+"];
  for (const f of features) {
    const p = f.properties || {};
    const b = (p["Borough"] || "").trim();
    const s = (p["Business Status"] || "").trim();
    if (b) boroughSet.add(b);
    if (s) statusSet.add(s);
    const licenses = parseLicenses(p);
    for (const lic of licenses) {
      const cat = (lic["Business Category"] || "").trim();
      if (cat) categorySet.add(cat);
    }
  }
  const boroughs = [...boroughSet].sort();
  const statuses = [...statusSet].sort();
  const categories = [...categorySet].sort();

  // Render list of checkboxes in the given container
  function renderFilterList(containerId, values, namePrefix) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = values
      .map((v, i) => {
        const id = `${namePrefix}-${i}`;
        return `<div class="filter-item"><input type="checkbox" id="${id}" data-value="${escHtml(v)}" /><label for="${id}">${escHtml(v)}</label></div>`;
      })
      .join("");
  }

  // Return array of data-value strings for checked checkboxes in the filter container
  function getSelected(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return [];
    return [...el.querySelectorAll("input:checked")].map((i) => i.dataset.value);
  }

  // True if feature's _searchText (name + address) includes the query string
  function featureMatchesSearch(f, q) {
    if (!q) return true;
    return (f.properties && (f.properties._searchText || "").includes(q));
  }

  // True if feature has at least one license whose Business Category is in the selected list
  function featureMatchesCategory(f, selected) {
    if (!selected.length) return true;
    const licenses = parseLicenses(f.properties || {});
    const cats = new Set(licenses.map((l) => (l["Business Category"] || "").trim()).filter(Boolean));
    return selected.some((s) => cats.has(s));
  }

  // When user picks a search result, show only that feature until filters/search change
  let searchSelectedFeature = null;

  // Apply borough/status/category/311 filters to the map source or show single feature if search result selected.
  function applyFilter() {
    if (searchSelectedFeature) {
      const source = map.getSource("businesses");
      if (source) source.setData({ type: "FeatureCollection", features: [searchSelectedFeature] });
      const countEl = document.getElementById("result-count");
      if (countEl) countEl.textContent = "Showing 1 business (search result)";
      return;
    }
    const selectedBoroughs = getSelected("filter-borough");
    const selectedStatuses = getSelected("filter-status");
    const selectedCategories = getSelected("filter-category");
    const selected311 = getSelected("filter-311");

    const filtered = features.filter((f) => {
      if (selectedBoroughs.length && !selectedBoroughs.includes((f.properties || {}).Borough)) return false;
      if (selectedStatuses.length && !selectedStatuses.includes((f.properties || {})["Business Status"])) return false;
      if (selectedCategories.length && !featureMatchesCategory(f, selectedCategories)) return false;
      if (selected311.length && !selected311.includes((f.properties || {})._311Bucket)) return false;
      return true;
    });

    const source = map.getSource("businesses");
    if (source) source.setData({ type: "FeatureCollection", features: filtered });
    const countEl = document.getElementById("result-count");
    if (countEl) countEl.textContent = `Showing ${filtered.length} of ${features.length} businesses`;
  }

  // Build popup HTML for a business: name, address, borough, ID, phone, status, licenses, 311 section
  function buildPopupHtml(p) {
    const name = p["Business Name"] || "(No name)";
    const bizId = p["Business Unique ID"] || "";
    const bizStatus = p["Business Status"] || "";
    const phone = p["Contact Phone"] || "";
    const borough = p["Borough"] || "";
    const total311 = p["total311"] || p["total_311"] || "";
    let problemCounts = {};
    try { problemCounts = JSON.parse(p["problemCountsJson"] || "{}"); } catch { problemCounts = {}; }
    let licenses = [];
    try { licenses = JSON.parse(p["licensesJson"] || "[]"); } catch { licenses = []; }
    const addrLines = buildAddress(p);
    const addrHtml = addrLines.length
      ? `<div style="margin-top:2px;">${addrLines.map(escHtml).join("<br/>")}</div>`
      : "";
    const licensesHtml = formatLicenses(licenses, 8);
    const complaintsHtml = formatProblemCounts(problemCounts, 25, false);
    const has311 = total311 || (Object.keys(problemCounts || {}).length > 0);

    /*
    to-do: This we will change later on when dataset has more years included other than 2025 only
    */
    const isInactivePriorTo2025 = (bizStatus && bizStatus.trim() !== "Active") && (() => {
      const latestYear = getLatestExpirationYear(licenses);
      return latestYear !== null && latestYear < 2025;
    })();
    let section311Html = "";
    if (isInactivePriorTo2025) {
      section311Html = `
    <div style="margin-top:14px;padding-top:0;">
      <strong style="display:block;padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.08);">2025 311 Service Reports</strong>
      <div style="margin-top:6px;">Business was inactive prior to 2025.</div>
    </div>`;
    } else if (has311) {
      section311Html = `
    <div style="margin-top:14px;padding-top:0;">
      <strong style="display:block;padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.08);">2025 311 Service Reports</strong>
      ${total311 ? `<div style="margin-top:6px;"><strong>Total 311</strong>: ${escHtml(total311)}</div>` : ""}
      ${complaintsHtml}
    </div>`;
    }
    return `
        <div style="
          font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
          font-size:13px;
          line-height:1.35;
          max-width:340px;
        ">
          <div style="font-weight:700;font-size:14px;">${escHtml(name)}</div>
          ${addrHtml}
          <div style="margin-top:8px;">
            ${borough ? `<strong>Borough</strong>: ${escHtml(borough)}<br/>` : ""}
            ${bizId ? `<strong>Business ID</strong>: <code>${escHtml(bizId)}</code><br/>` : ""}
            ${phone ? `<strong>Phone</strong>: ${escHtml(phone)}<br/>` : ""}
            ${bizStatus ? `<strong>Status</strong>: ${escHtml(bizStatus)}<br/>` : ""}
          </div>
          <div style="margin-top:10px;max-height:260px;overflow-y:auto;overflow-x:hidden;padding:10px 8px 10px 10px;background:#fffef0;border-radius:6px;overscroll-behavior:contain;-webkit-overflow-scrolling:touch;">
            ${licensesHtml}
            ${section311Html}
          </div>
        </div>`;
  }

  // Open a Mapbox popup at lngLat with content from buildPopupHtml, make inner content scrollable.
  function openPopupForFeature(map, feature, lngLat) {
    const p = feature.properties || {};
    const html = buildPopupHtml(p);
    const popup = new mapboxgl.Popup({ closeButton: true, closeOnClick: true, maxWidth: "360px" })
      .setLngLat(lngLat)
      .setHTML(html)
      .addTo(map);
    setTimeout(() => makePopupScrollable(popup), 0);
  }

  map.on("load", () => {
    // GeoJSON source; data is replaced by applyFilter() when filters or search change
    map.addSource("businesses", { type: "geojson", data: fc });

    // Circle layer: color by Borough (fallback gray if Not Recorded)
    map.addLayer({
      id: "business-points",
      type: "circle",
      source: "businesses",
      paint: {
        "circle-radius": 3,
        "circle-color": [
          "match",
          ["get", "Borough"],
          "Bronx", "#2563eb",
          "Manhattan", "#eab308",
          "Staten Island", "#dc2626",
          "Brooklyn", "#16a34a",
          "Queens", "#7c3aed",
          "#6b7280"
        ],
        "circle-stroke-width": 0.5,
        "circle-stroke-color": "#ffffff",
        "circle-opacity": 0.9
      }
    });

    // Clicking on a point opens popup with business details
    map.on("click", "business-points", (e) => {
      const f = e.features && e.features[0];
      if (!f) return;
      const [lng, lat] = f.geometry?.coordinates || e.lngLat;
      openPopupForFeature(map, f, [lng, lat]);
    });

    map.on("mouseenter", "business-points", () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", "business-points", () => { map.getCanvas().style.cursor = ""; });

    // Populate filter panels with checkbox lists
    renderFilterList("filter-borough", boroughs, "borough");
    renderFilterList("filter-status", statuses, "status");
    renderFilterList("filter-category", categories, "category");
    renderFilterList("filter-311", buckets311, "311");

    // Search: type-ahead over business name + address
    const searchInput = document.getElementById("search-input");
    const searchDropdown = document.getElementById("search-dropdown");
    const MAX_SEARCH_RESULTS = 10;
    let searchDebounce = null;

    function hideSearchDropdown() {
      if (searchDropdown) {
        searchDropdown.classList.remove("is-open");
        searchDropdown.innerHTML = "";
        searchDropdown.setAttribute("aria-hidden", "true");
      }
    }

    // Show dropdown with up to MAX_SEARCH_RESULTS; each item flies to point and opens popup on click.
    function showSearchDropdown(matches) {
      if (!searchDropdown) return;
      searchDropdown.innerHTML = matches.slice(0, MAX_SEARCH_RESULTS).map((f, i) => {
        const p = f.properties || {};
        const name = (p["Business Name"] || "(No name)").trim();
        const address = fullAddressString(p);
        return `<div class="search-dropdown-item" data-index="${i}"><div class="name">${escHtml(name)}</div>${address ? `<div class="address">${escHtml(address)}</div>` : ""}</div>`;
      }).join("");
      searchDropdown.classList.add("is-open");
      searchDropdown.setAttribute("aria-hidden", "false");
      searchDropdown.querySelectorAll(".search-dropdown-item").forEach((el, i) => {
        el.addEventListener("click", () => {
          const feature = matches[i];
          const p = feature.properties || {};
          const name = (p["Business Name"] || "(No name)").trim();
          const addr = fullAddressString(p);
          searchSelectedFeature = feature;
          hideSearchDropdown();
          if (searchInput) searchInput.value = addr ? `${name}, ${addr}` : name;
          const source = map.getSource("businesses");
          if (source) source.setData({ type: "FeatureCollection", features: [feature] });
          const [lng, lat] = feature.geometry?.coordinates || [];
          if (lng != null && lat != null) {
            map.flyTo({ center: [lng, lat], zoom: 16, duration: 800 });
            setTimeout(() => openPopupForFeature(map, feature, [lng, lat]), 400);
          }
          const countEl = document.getElementById("result-count");
          if (countEl) countEl.textContent = "Showing 1 business (search result)";
        });
      });
    }

    // On input: clear search state if empty and re-apply filters; else debounce and show matching results. 
    function onSearchInput() {
      const query = (searchInput?.value || "").trim().toLowerCase();
      if (query.length === 0) {
        hideSearchDropdown();
        searchSelectedFeature = null;
        applyFilter();
        return;
      }
      if (searchDebounce) clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        const matches = features.filter((f) => featureMatchesSearch(f, query));
        if (matches.length === 0) {
          hideSearchDropdown();
          return;
        }
        showSearchDropdown(matches);
      }, 200);
    }

    searchInput?.addEventListener("input", onSearchInput);
    searchInput?.addEventListener("focus", () => {
      const q = (searchInput?.value || "").trim().toLowerCase();
      if (q.length > 0) {
        const matches = features.filter((f) => featureMatchesSearch(f, q));
        if (matches.length > 0) showSearchDropdown(matches);
      }
    });
    searchInput?.addEventListener("blur", () => {
      setTimeout(hideSearchDropdown, 150);
    });

    // When any filter changes: clear search selection if set, then re-apply filters
    ["filter-borough", "filter-status", "filter-category", "filter-311"].forEach((id) => {
      document.getElementById(id)?.addEventListener("change", () => {
        if (!searchSelectedFeature) applyFilter();
        else {
          searchSelectedFeature = null;
          if (searchInput) searchInput.value = "";
          applyFilter();
        }
      });
    });

    // Reset: clear search, uncheck all filters, show all features, reset map view
    document.getElementById("reset-btn")?.addEventListener("click", () => {
      searchSelectedFeature = null;
      if (searchInput) searchInput.value = "";
      hideSearchDropdown();
      ["filter-borough", "filter-status", "filter-category", "filter-311"].forEach((id) => {
        const el = document.getElementById(id);
        el?.querySelectorAll("input[type=checkbox]").forEach((cb) => { cb.checked = false; });
      });
      applyFilter();
      map.flyTo({ center: NYC_CENTER, zoom: NYC_ZOOM, duration: 600 });
    });

    // Initial filter pass (all checkboxes unchecked = show all features)
    applyFilter();
  });
}

main();