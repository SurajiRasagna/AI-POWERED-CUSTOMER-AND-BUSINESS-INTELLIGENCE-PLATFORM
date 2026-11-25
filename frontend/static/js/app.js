/* Shared frontend helpers: API access, formatting, charts, toasts. */

const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";

const PALETTE = [
  "#2563eb",
  "#0ea5e9",
  "#0f766e",
  "#7c3aed",
  "#f59e0b",
  "#e11d48",
  "#64748b",
  "#10b981",
  "#8b5cf6",
  "#f97316",
];

/* ---------- Fetch wrapper ---------- */

async function api(path, options = {}) {
  const headers = { Accept: "application/json" };
  const config = { method: options.method || "GET", headers };

  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    config.body = JSON.stringify(options.body);
  }

  if (options.form !== undefined) {
    config.body = options.form;
  }

  if (config.method !== "GET") {
    headers["X-CSRF-Token"] = csrfToken;
  }

  const response = await fetch(path, config);

  if (response.status === 401) {
    const next = encodeURIComponent(window.location.pathname);
    window.location.href = `/login?next=${next}`;
    throw new Error("Your session expired. Please sign in again.");
  }

  const isJson = (response.headers.get("content-type") || "").includes(
    "application/json"
  );
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    const message =
      (payload && (payload.error || payload.message)) ||
      `Request failed (${response.status}).`;
    throw new Error(message);
  }

  return payload;
}

/* ---------- Loading state ---------- */

let pendingLoads = 0;

function beginLoad() {
  pendingLoads += 1;
  document.body.classList.remove("load-done");
  document.body.classList.add("is-loading");
}

function endLoad() {
  pendingLoads = Math.max(0, pendingLoads - 1);
  if (pendingLoads === 0) {
    document.body.classList.remove("is-loading");
    document.body.classList.add("load-done");
  }
}

async function withLoader(task) {
  beginLoad();
  try {
    return await task();
  } catch (error) {
    toast(error.message || "Something went wrong.", "error");
    throw error;
  } finally {
    endLoad();
  }
}

/* ---------- Formatting ---------- */

const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const nf2 = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function num(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? nf0.format(parsed) : "—";
}

function decimal(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? nf2.format(parsed) : "—";
}

function money(value, digits = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "BRL",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(parsed);
}

function compactMoney(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  const abs = Math.abs(parsed);
  if (abs >= 1_000_000) return `R$ ${(parsed / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `R$ ${(parsed / 1_000).toFixed(1)}K`;
  return money(parsed);
}

function monthLabel(value) {
  if (!value) return "—";
  const parts = String(value).split("-");
  if (parts.length < 2) return value;
  const date = new Date(Number(parts[0]), Number(parts[1]) - 1, 1);
  return date.toLocaleDateString("en-US", {
    month: "short",
    year: "numeric",
  });
}

function prettyLabel(value) {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sentimentBadge(label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("positive")) return "badge-green";
  if (key.includes("negative")) return "badge-rose";
  if (key.includes("neutral")) return "badge-amber";
  return "badge-slate";
}

function riskBadge(label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("high")) return "badge-rose";
  if (key.includes("medium")) return "badge-amber";
  if (key.includes("low")) return "badge-green";
  return "badge-slate";
}

function signalBadge(label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("high")) return "badge-blue";
  if (key.includes("low")) return "badge-amber";
  return "badge-green";
}

function statusBadge(status) {
  return String(status).toLowerCase() === "ok" ? "badge-green" : "badge-slate";
}

/* ---------- Tables ---------- */

function fillTable(tbodyId, rows, renderRow, emptyMessage = "No data available.") {
  const body = document.getElementById(tbodyId);
  if (!body) return;

  if (!rows || rows.length === 0) {
    const columns = body.closest("table")?.querySelectorAll("thead th").length || 1;
    body.innerHTML = `<tr><td colspan="${columns}"><div class="empty">${escapeHtml(
      emptyMessage
    )}</div></td></tr>`;
    return;
  }

  body.innerHTML = rows.map(renderRow).join("");
}

/* ---------- Toasts ---------- */

function toast(message, kind = "info") {
  let host = document.getElementById("toastHost");
  if (!host) {
    host = document.createElement("div");
    host.id = "toastHost";
    document.body.appendChild(host);
  }

  const node = document.createElement("div");
  node.className = `toast toast-${kind}`;
  node.textContent = message;
  host.appendChild(node);

  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transform = "translateY(6px)";
    node.style.transition = "0.25s ease";
    setTimeout(() => node.remove(), 260);
  }, 3800);
}

/* ---------- Charts ---------- */

const chartRegistry = new Map();

function destroyChart(canvasId) {
  if (chartRegistry.has(canvasId)) {
    chartRegistry.get(canvasId).destroy();
    chartRegistry.delete(canvasId);
  }
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace("#", "");
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function baseOptions(options = {}) {
  const {
    legend = true,
    currency = false,
    horizontal = false,
    yLabel,
  } = options;

  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? "y" : "x",
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: legend,
        position: "bottom",
        labels: {
          usePointStyle: true,
          boxWidth: 7,
          boxHeight: 7,
          padding: 16,
          color: "#64748b",
          font: { size: 11, family: "Inter" },
        },
      },
      tooltip: {
        backgroundColor: "#0f172a",
        padding: 10,
        cornerRadius: 8,
        titleFont: { size: 12, family: "Inter" },
        bodyFont: { size: 12, family: "Inter" },
        callbacks: currency
          ? {
              label: (ctx) =>
                `${ctx.dataset.label}: ${money(ctx.parsed[horizontal ? "x" : "y"])}`,
            }
          : {},
      },
    },
    scales: horizontal
      ? {
          x: {
            beginAtZero: true,
            grid: { color: "#eef2f7" },
            border: { display: false },
            ticks: {
              color: "#94a3b8",
              font: { size: 10.5, family: "Inter" },
              maxTicksLimit: 6,
              maxRotation: 0,
              callback: (value) => (currency ? compactMoney(value) : num(value)),
            },
          },
          y: {
            grid: { display: false },
            border: { display: false },
            ticks: {
              color: "#475569",
              font: { size: 10.5, family: "Inter" },
              autoSkip: false,
            },
          },
        }
      : {
          x: {
            grid: { display: false },
            border: { display: false },
            ticks: {
              color: "#94a3b8",
              font: { size: 10.5, family: "Inter" },
              maxRotation: 0,
              autoSkipPadding: 12,
            },
          },
          y: {
            beginAtZero: true,
            grid: { color: "#eef2f7" },
            border: { display: false },
            title: yLabel
              ? {
                  display: true,
                  text: yLabel,
                  color: "#94a3b8",
                  font: { size: 10.5, family: "Inter" },
                }
              : undefined,
            ticks: {
              color: "#94a3b8",
              font: { size: 10.5, family: "Inter" },
              maxTicksLimit: 6,
              callback: (value) => (currency ? compactMoney(value) : num(value)),
            },
          },
        },
  };
}

function lineChart(canvasId, labels, datasets, options = {}) {
  if (!window.Chart) return null;
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const built = datasets.map((item, index) => {
    const color = item.color || PALETTE[index % PALETTE.length];
    return {
      label: item.label,
      data: item.data,
      borderColor: color,
      backgroundColor: item.fill === false ? color : hexToRgba(color, 0.12),
      borderWidth: 2.5,
      tension: 0.35,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: color,
      pointHoverBorderColor: "#fff",
      pointHoverBorderWidth: 2,
      fill: item.fill !== false,
    };
  });

  const chart = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: built },
    options: baseOptions({ currency: false, ...options }),
  });

  chartRegistry.set(canvasId, chart);
  return chart;
}

function barChart(canvasId, labels, datasets, options = {}) {
  if (!window.Chart) return null;
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const built = datasets.map((item, index) => ({
    label: item.label,
    data: item.data,
    backgroundColor: item.colors || hexToRgba(item.color || PALETTE[index % PALETTE.length], 0.85),
    borderRadius: 6,
    borderSkipped: false,
    maxBarThickness: 38,
  }));

  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: built },
    options: baseOptions({ legend: false, ...options }),
  });

  chartRegistry.set(canvasId, chart);
  return chart;
}

function doughnutChart(canvasId, labels, values, options = {}) {
  if (!window.Chart) return null;
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const chart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: options.colors || PALETTE,
          borderWidth: 2,
          borderColor: "#ffffff",
          hoverOffset: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: options.cutout || "64%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            boxWidth: 7,
            boxHeight: 7,
            padding: 14,
            color: "#64748b",
            font: { size: 11, family: "Inter" },
          },
        },
        tooltip: {
          backgroundColor: "#0f172a",
          padding: 10,
          cornerRadius: 8,
          bodyFont: { size: 12, family: "Inter" },
        },
      },
    },
  });

  chartRegistry.set(canvasId, chart);
  return chart;
}

/* ---------- CSV download ---------- */

function downloadCsv(filename, headers, rows) {
  const escapeCell = (value) => {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };

  const lines = [headers.map(escapeCell).join(",")];
  rows.forEach((row) => lines.push(row.map(escapeCell).join(",")));

  const blob = new Blob([lines.join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/* ---------- Sidebar ---------- */

function initSidebar() {
  const sidebar = document.getElementById("sidebar");
  const scrim = document.getElementById("sidebarScrim");
  const button = document.getElementById("menuBtn");

  if (!sidebar || !button) return;

  const setExpanded = (expanded) => {
    button.setAttribute("aria-expanded", String(expanded));
  };

  const close = () => {
    sidebar.classList.remove("open");
    scrim?.classList.remove("show");
    setExpanded(false);
  };

  button.addEventListener("click", () => {
    const expanded = sidebar.classList.toggle("open");
    scrim?.classList.toggle("show", expanded);
    setExpanded(expanded);
  });

  scrim?.addEventListener("click", close);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sidebar.classList.contains("open")) {
      close();
      button.focus();
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 980) close();
  });

  sidebar.querySelectorAll("a").forEach((link) =>
    link.addEventListener("click", () => {
      if (window.innerWidth <= 980) close();
    })
  );
}

document.addEventListener("DOMContentLoaded", initSidebar);

if (window.Chart) {
  Chart.defaults.font.family = "Inter";
  Chart.defaults.color = "#64748b";
}
