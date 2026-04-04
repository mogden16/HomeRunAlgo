const state = {
  dashboard: null,
  filteredHistory: [],
  filteredLatestPicks: [],
  latestTierFilters: new Set(["elite", "strong"]),
  historyTierFilters: new Set(["elite", "strong"]),
  selectedHistoryDate: "",
};

const DEFAULT_LATEST_PICKS_EMPTY_MESSAGE =
  "Today's public picks have not been posted yet. Publish reruns every 15 minutes before first pitch and settle reruns every 15 minutes once games begin.";
const DEFAULT_HISTORY_EMPTY_MESSAGE = "No published picks match those filters.";
const DEFAULT_YESTERDAY_RECAP_EMPTY_MESSAGE = "No published picks were recorded for the previous tracked date.";
const DEFAULT_MODEL_EXPLAINER_MESSAGE = "Metric details are not available for the current dashboard build.";
const MANUAL_REFRESH_KEY_STORAGE = "manualRefreshKey";
const CONFIDENCE_TIERS = ["elite", "strong", "watch", "longshot"];
const ALL_DATES_FILTER_VALUE = "__all_dates__";
const DEFAULT_TIER_GUIDE = [
  {
    confidence_tier: "elite",
    label: "elite",
    description: "Most selective subset on the board.",
  },
  {
    confidence_tier: "strong",
    label: "strong",
    description: "Main public board after the elite subset is carved out.",
  },
  {
    confidence_tier: "watch",
    label: "watch",
    description: "Worth monitoring, but lower-confidence than the main public board.",
  },
  {
    confidence_tier: "longshot",
    label: "longshot",
    description: "Visible only when all tiers are enabled.",
  },
];

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(1);
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  return new Date(`${value}T00:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatTime(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatGameTime(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatElitePolicySummary(dashboard) {
  const policy = dashboard?.confidence_policy || {};
  const topK = Number(policy?.elite_top_k);
  const probabilityFloor = Number(policy?.elite_probability_floor);
  if (Number.isInteger(topK) && topK > 0) {
    const pickLabel = topK === 1 ? "pick" : "picks";
    if (Number.isFinite(probabilityFloor)) {
      return `Current elite policy caps the tier at the top ${topK} ${pickLabel} per slate above ${formatPercent(probabilityFloor, 1)}.`;
    }
    return `Current elite policy caps the tier at the top ${topK} ${pickLabel} per slate.`;
  }
  if (Number.isFinite(probabilityFloor)) {
    return `Elite rows must clear at least ${formatPercent(probabilityFloor, 1)} predicted HR probability.`;
  }
  return "Elite is the most selective subset within the public board.";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeTier(value) {
  return String(value || "").trim().toLowerCase();
}

function tierClass(value) {
  return `tier-tag tier-${normalizeTier(value)}`;
}

function resultClass(value) {
  if (value === "HR") {
    return "result-hit";
  }
  if (value === "Pending") {
    return "result-pending";
  }
  return "result-miss";
}

function formatLineupSource(value) {
  return String(value || "").trim().toLowerCase() === "confirmed" ? "Confirmed lineup" : "Projected lineup";
}

function lineupSourceClass(value) {
  return String(value || "").trim().toLowerCase() === "confirmed" ? "lineup-badge-confirmed" : "lineup-badge-projected";
}

function renderLineupBadge(value) {
  return `<span class="lineup-badge ${lineupSourceClass(value)}">${escapeHtml(formatLineupSource(value))}</span>`;
}

function formatGameState(value) {
  const token = String(value || "").trim().toLowerCase();
  if (token === "final") {
    return "Final";
  }
  if (token === "live") {
    return "Live";
  }
  return "Pregame";
}

function formatStadium(row) {
  const name = String(row.ballpark_name || "").trim();
  const region = String(row.ballpark_region_abbr || "").trim();
  if (!name) {
    return "-";
  }
  return region ? `${name}, ${region}` : name;
}

function weatherIcon(value) {
  const token = String(value || "").trim().toLowerCase();
  if (token === "clear") {
    return "☀️";
  }
  if (token === "cloudy") {
    return "☁️";
  }
  if (token === "fog") {
    return "🌫️";
  }
  if (token === "rain") {
    return "🌧️";
  }
  if (token === "snow") {
    return "🌨️";
  }
  if (token === "storm") {
    return "⛈️";
  }
  return "❔";
}

function formatTemperature(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "";
  }
  return `${Math.round(Number(value))}\u00B0F`;
}

function normalizeDegrees(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  const normalized = Number(value) % 360;
  return normalized >= 0 ? normalized : normalized + 360;
}

function windArrow(windDirectionDeg, fieldBearingDeg) {
  const windFrom = normalizeDegrees(windDirectionDeg);
  const fieldBearing = normalizeDegrees(fieldBearingDeg);
  if (windFrom === null || fieldBearing === null) {
    return "";
  }
  const blowTo = normalizeDegrees(windFrom + 180);
  const relative = normalizeDegrees(blowTo - fieldBearing);
  if (relative < 22.5 || relative >= 337.5) {
    return "↑";
  }
  if (relative < 67.5) {
    return "↗";
  }
  if (relative < 112.5) {
    return "→";
  }
  if (relative < 157.5) {
    return "↘";
  }
  if (relative < 202.5) {
    return "↓";
  }
  if (relative < 247.5) {
    return "↙";
  }
  if (relative < 292.5) {
    return "←";
  }
  return "↖";
}

function formatWind(row) {
  const speed = row.wind_speed_mph;
  if (speed === null || speed === undefined || Number.isNaN(Number(speed))) {
    return "-";
  }
  const arrow = windArrow(row.wind_direction_deg, row.field_bearing_deg);
  const speedText = `${Math.round(Number(speed))} mph`;
  return arrow ? `${arrow} ${speedText}` : speedText;
}

function weatherIcon(value) {
  const token = String(value || "").trim().toLowerCase();
  if (token === "clear") {
    return "\u2600\uFE0F";
  }
  if (token === "cloudy") {
    return "\u2601\uFE0F";
  }
  if (token === "fog") {
    return "\uD83C\uDF2B\uFE0F";
  }
  if (token === "rain") {
    return "\uD83C\uDF27\uFE0F";
  }
  if (token === "snow") {
    return "\uD83C\uDF28\uFE0F";
  }
  if (token === "storm") {
    return "\u26C8\uFE0F";
  }
  return "\u2753";
}

function windArrow(windDirectionDeg, fieldBearingDeg) {
  const windFrom = normalizeDegrees(windDirectionDeg);
  const fieldBearing = normalizeDegrees(fieldBearingDeg);
  if (windFrom === null || fieldBearing === null) {
    return "";
  }
  const blowTo = normalizeDegrees(windFrom + 180);
  const relative = normalizeDegrees(blowTo - fieldBearing);
  if (relative < 22.5 || relative >= 337.5) {
    return "\u2191";
  }
  if (relative < 67.5) {
    return "\u2197";
  }
  if (relative < 112.5) {
    return "\u2192";
  }
  if (relative < 157.5) {
    return "\u2198";
  }
  if (relative < 202.5) {
    return "\u2193";
  }
  if (relative < 247.5) {
    return "\u2199";
  }
  if (relative < 292.5) {
    return "\u2190";
  }
  return "\u2196";
}

function renderGameMeta(row) {
  const weatherLabel = String(row.weather_label || "").trim() || "Unknown";
  const temperatureText = formatTemperature(row.temperature_f);
  const weatherMeta = [weatherIcon(weatherLabel), weatherLabel, temperatureText].filter(Boolean).join(" ");
  const mobileLineOne = [formatGameTime(row.game_datetime), formatStadium(row)]
    .filter((value) => value && value !== "-")
    .join(" | ") || "-";
  const mobileLineTwo = [weatherIcon(weatherLabel), temperatureText || weatherLabel, formatWind(row)]
    .filter((value) => value && value !== "-")
    .join(" | ") || "Unknown";
  return `
    <div class="pick-meta-block pick-meta-block-desktop">
      <div class="pick-meta-line"><span class="pick-meta-label">Gametime</span><span class="pick-meta-value">${escapeHtml(formatGameTime(row.game_datetime))}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Stadium</span><span class="pick-meta-value">${escapeHtml(formatStadium(row))}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Conditions</span><span class="pick-meta-value">${escapeHtml(weatherMeta)}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Wind</span><span class="pick-meta-value">${escapeHtml(formatWind(row))}</span></div>
    </div>
    <div class="pick-meta-block-mobile">
      <span>${escapeHtml(mobileLineOne)}</span>
      <span>${escapeHtml(mobileLineTwo)}</span>
    </div>
  `;
}

function renderProbabilityCell(row) {
  const probability = formatPercent(row.predicted_hr_probability);
  const modelScore = formatScore(row.predicted_hr_score);
  return renderMobileCellStack(
    "HR chance",
    `
      <div class="probability-cell">
        <strong>${escapeHtml(probability)}</strong>
        <span class="probability-subtext">Model score ${escapeHtml(modelScore)}</span>
      </div>
    `,
  );
}

function renderMobileCellStack(label, content, extraClass = "") {
  return `
    <div class="mobile-cell-stack ${extraClass}">
      <span class="mobile-cell-label">${escapeHtml(label)}</span>
      <div class="mobile-cell-content">${content}</div>
    </div>
  `;
}

function renderMobileWhyDetails(row, extraClass = "") {
  const reasons = [row.top_reason_1, row.top_reason_2, row.top_reason_3].filter(Boolean);
  if (!reasons.length) {
    return "";
  }
  const detailsClass = ["mobile-why-details", extraClass].filter(Boolean).join(" ");
  return `
    <details class="${detailsClass}">
      <summary>Why</summary>
      <ul class="reason-list">${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>
    </details>
  `;
}

function renderGameMeta(row) {
  const weatherLabel = String(row.weather_label || "").trim() || "Unknown";
  const temperatureText = formatTemperature(row.temperature_f);
  const weatherMeta = [weatherIcon(weatherLabel), weatherLabel, temperatureText].filter(Boolean).join(" ");
  const mobileLineOne = [formatGameTime(row.game_datetime), formatStadium(row)].filter((value) => value && value !== "-").join(" • ") || "-";
  const mobileLineTwo = [weatherIcon(weatherLabel), temperatureText || weatherLabel, formatWind(row)]
    .filter((value) => value && value !== "-")
    .join(" • ") || "❔ Unknown";
  return `
    <div class="pick-meta-block pick-meta-block-desktop">
      <div class="pick-meta-line"><span class="pick-meta-label">Gametime</span><span class="pick-meta-value">${escapeHtml(formatGameTime(row.game_datetime))}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Stadium</span><span class="pick-meta-value">${escapeHtml(formatStadium(row))}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Conditions</span><span class="pick-meta-value">${escapeHtml(weatherMeta)}</span></div>
      <div class="pick-meta-line"><span class="pick-meta-label">Wind</span><span class="pick-meta-value">${escapeHtml(formatWind(row))}</span></div>
    </div>
    <div class="pick-meta-block-mobile">
      <span>${escapeHtml(mobileLineOne)}</span>
      <span>${escapeHtml(mobileLineTwo)}</span>
    </div>
  `;
}

function findConfidenceSummary(rows, tier) {
  return (rows || []).find((row) => normalizeTier(row.confidence_tier) === tier) || null;
}

function filterRowsByTierSelection(rows, selectedTiers) {
  if (!(selectedTiers instanceof Set) || !selectedTiers.size) {
    return [];
  }
  return (rows || []).filter((row) => selectedTiers.has(normalizeTier(row.confidence_tier)));
}

function formatWholeNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString();
}

function enrichLatestRanks(rows) {
  const list = Array.isArray(rows) ? rows.map((row) => ({ ...row })) : [];
  const morningOrdered = list
    .filter((row) => Number.isFinite(Number(row.morning_rank)))
    .sort((left, right) => Number(left.morning_rank) - Number(right.morning_rank));
  const morningDisplayRankById = new Map(morningOrdered.map((row, index) => [String(row.pick_id || `${index}`), index + 1]));
  return list.map((row, index) => ({
    ...row,
    display_rank: index + 1,
    morning_display_rank: morningDisplayRankById.get(String(row.pick_id || `${index}`)) ?? null,
  }));
}

function renderRankTrend(row) {
  const currentRank = Number(row.display_rank);
  const morningRank = Number(row.morning_display_rank);
  if (!Number.isFinite(currentRank) || !Number.isFinite(morningRank) || currentRank === morningRank) {
    return "";
  }
  const movedUp = currentRank < morningRank;
  const delta = Math.abs(morningRank - currentRank);
  const symbol = movedUp ? "▲" : "▼";
  const title = movedUp
    ? `Up ${delta} spot${delta === 1 ? "" : "s"} since morning`
    : `Down ${delta} spot${delta === 1 ? "" : "s"} since morning`;
  return `<span class="rank-trend ${movedUp ? "rank-trend-up" : "rank-trend-down"}" title="${escapeHtml(title)}">${symbol}${escapeHtml(delta)}</span>`;
}

function renderRankCell(row) {
  const rankValue = row.display_rank ?? row.rank;
  return `<div class="rank-cell"><strong>${escapeHtml(formatWholeNumber(rankValue))}</strong>${renderRankTrend(row)}</div>`;
}

function formatModelValue(value) {
  const text = String(value ?? "").trim();
  return text || "Not available";
}

function buildRefreshScheduleSummary(schedule) {
  const runs = Array.isArray(schedule?.runs) ? schedule.runs : [];
  const prepareRun = runs.find((run) => run.type === "prepare");
  const publishRuns = runs.filter((run) => run.type === "publish");
  const settleRun = runs.find((run) => run.type === "settle");
  const publishTimes = publishRuns.map((run) => run.time_et).filter(Boolean);

  if (!settleRun && !prepareRun && !publishTimes.length) {
    return "Schedule unavailable.";
  }

  const publishText = publishTimes.length
    ? publishTimes.every((value) => String(value).toLowerCase().startsWith("every"))
      ? `Publish reruns ${publishTimes.join(", ")}.`
      : `Publish runs at ${publishTimes.join(", ")}.`
    : "";
  const settleText = settleRun?.time_et
    ? String(settleRun.time_et).toLowerCase().startsWith("every")
      ? `Settle reruns ${settleRun.time_et}.`
      : `Settle run at ${settleRun.time_et}.`
    : "";
  const prepareText = prepareRun?.time_et ? `Prepare run at ${prepareRun.time_et}.` : "";
  return [settleText, prepareText, publishText].filter(Boolean).join(" ");
}

function buildRefreshScheduleInlineText(schedule) {
  const runs = Array.isArray(schedule?.runs) ? schedule.runs : [];
  const settleRun = runs.find((run) => run.type === "settle");
  const prepareRun = runs.find((run) => run.type === "prepare");
  const publishRuns = runs.filter((run) => run.type === "publish");
  const publishTimes = publishRuns.map((run) => run.time_et).filter(Boolean);

  if (!settleRun && !prepareRun && !publishTimes.length) {
    return "";
  }

  const parts = ["Refresh schedule:"];
  if (settleRun?.time_et) {
    parts.push(
      String(settleRun.time_et).toLowerCase().startsWith("every")
        ? `Settle reruns ${settleRun.time_et}.`
        : `${settleRun.time_et} settle.`,
    );
  }
  if (prepareRun?.time_et) {
    parts.push(`${prepareRun.time_et} prepare.`);
  }
  if (publishTimes.length) {
    parts.push(
      publishTimes.every((value) => String(value).toLowerCase().startsWith("every"))
        ? `Publish reruns ${publishTimes.join(", ")}.`
        : `Publish runs at ${publishTimes.join(", ")}.`,
    );
  }
  return parts.join(" ");
}

function renderDashboardAlerts(alerts) {
  const target = document.getElementById("dashboard-alerts");
  const rows = Array.isArray(alerts) ? alerts : [];
  if (!rows.length) {
    target.hidden = true;
    target.innerHTML = "";
    return;
  }

  target.hidden = false;
  target.innerHTML = rows
    .map(
      (alert) => `
        <article class="dashboard-alert dashboard-alert-${escapeHtml(alert.kind || "warning")}">
          <strong>${escapeHtml(alert.title || "Operational note")}</strong>
          <p>${escapeHtml(alert.message || "")}</p>
        </article>
      `,
    )
    .join("");
}

function renderOverviewCards(dashboard) {
  const overview = dashboard?.overview || {};
  const confidenceSummary = dashboard?.confidence_summary || [];
  const latestPicks = Array.isArray(dashboard?.latest_picks) ? dashboard.latest_picks : [];
  const eliteSummary = findConfidenceSummary(confidenceSummary, "elite");
  const elitePicks = eliteSummary?.picks ?? null;
  const eliteHomers = eliteSummary?.homers ?? null;
  const confirmedCount = latestPicks.filter((row) => String(row.lineup_source || "").toLowerCase() === "confirmed").length;
  const settledPicks = Number(overview.settled_picks) || 0;
  const trackedHomers = Number(overview.tracked_homers) || 0;
  const overallHitRate = settledPicks > 0 ? trackedHomers / settledPicks : null;
  const cards = [
    {
      label: "Latest board",
      value: formatWholeNumber(overview.latest_slate_size),
      subtext: `${formatDate(dashboard?.latest_available_date)} public slate.`,
    },
    {
      label: "Elite subset hit rate",
      value: formatPercent(eliteSummary?.hit_rate),
      subtext:
        elitePicks && eliteHomers !== null
          ? `${formatWholeNumber(eliteHomers)} home runs across ${formatWholeNumber(elitePicks)} elite picks. ${formatElitePolicySummary(dashboard)}`
          : formatElitePolicySummary(dashboard),
    },
    {
      label: "Settled hit rate",
      value: formatPercent(overallHitRate),
      subtext: `${formatWholeNumber(trackedHomers)} homers across ${formatWholeNumber(settledPicks)} settled picks.`,
    },
    {
      label: "Lineups confirmed",
      value: `${formatWholeNumber(confirmedCount)}/${formatWholeNumber(latestPicks.length)}`,
      subtext: latestPicks.length
        ? "Current board rows with confirmed lineups."
        : "No picks posted on the latest board yet.",
    },
    {
      label: "Last refresh",
      value: formatTime(dashboard?.generated_at),
      subtext: `${formatWholeNumber(overview.tracked_picks)} tracked picks across ${formatWholeNumber(overview.tracked_dates)} slate dates.`,
    },
  ];

  document.getElementById("overview-cards").innerHTML = cards
    .map(
      (card) => `
        <article class="stat-card">
          <p class="eyebrow">${escapeHtml(card.label)}</p>
          <span class="value">${escapeHtml(card.value)}</span>
          <p class="subtext">${escapeHtml(card.subtext)}</p>
        </article>
      `,
    )
    .join("");
}

function renderTierLegend(entries) {
  const target = document.getElementById("tier-legend");
  if (!target) {
    return;
  }
  const guide = Array.isArray(entries) && entries.length ? entries : DEFAULT_TIER_GUIDE;
  target.innerHTML = guide
    .map(
      (entry) => `
        <span class="legend-item">
          <span class="tier-tag tier-${escapeHtml(normalizeTier(entry.confidence_tier || entry.label || "watch"))}">${escapeHtml(entry.label || entry.confidence_tier || "watch")}</span>
          <span>${escapeHtml(entry.description || "")}</span>
        </span>
      `,
    )
    .join("");
}

function renderSimpleTable(targetId, columns, rows, emptyMessage = "No rows available.", options = {}) {
  const target = document.getElementById(targetId);
  if (!rows.length) {
    target.innerHTML = `<p class="empty-state">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const tableClass = ["data-table", options.mobileCards === false ? "" : "mobile-cards", options.tableClass || ""].filter(Boolean).join(" ");
  const headers = columns
    .map((column) => `<th class="${escapeHtml(column.headerClass || column.cellClass || "")}">${escapeHtml(column.label)}</th>`)
    .join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map(
          (column) =>
            `<td class="${escapeHtml(column.cellClass || "")}" data-label="${escapeHtml(column.label)}">${column.render(row)}</td>`,
        )
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  target.innerHTML = `<table class="${tableClass}"><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderConfidenceTable(rows) {
  renderSimpleTable(
    "confidence-table",
    [
      {
        label: "Tier",
        render: (row) => `<span class="${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>`,
      },
      { label: "Picks", render: (row) => escapeHtml(row.picks) },
      { label: "Homers", render: (row) => escapeHtml(row.homers) },
      { label: "Hit rate", render: (row) => escapeHtml(formatPercent(row.hit_rate)) },
      { label: "Avg prob.", render: (row) => escapeHtml(formatPercent(row.avg_probability)) },
    ],
    rows,
  );
}

function renderPicksTable(targetId, rows, emptyMessage, { includeGameMeta = false } = {}) {
  const displayRows = includeGameMeta ? enrichLatestRanks(rows) : rows;
  const columns = [
    { label: "Date", cellClass: "col-date", render: (row) => escapeHtml(formatDate(row.game_date)) },
    {
      label: "Rank",
      cellClass: "col-rank",
      render: (row) => renderMobileCellStack("Rank", renderRankCell(row), "stack-center"),
    },
    {
      label: "Hitter",
      cellClass: "col-hitter",
      render: (row) => `
        ${renderMobileCellStack(
          "Hitter",
          `
            <div class="name-block">
              <strong>${escapeHtml(row.batter_name)}</strong>
              <span>${escapeHtml(row.team)} vs ${escapeHtml(row.opponent_team || "-")}</span>
              <span class="mobile-inline-pitcher">vs ${escapeHtml(row.pitcher_name || "-")}</span>
              <span>${renderLineupBadge(row.lineup_source)}${row.batting_order ? ` <span class="lineup-order">batting ${escapeHtml(row.batting_order)}</span>` : ""} <span class="lineup-separator">|</span> ${escapeHtml(formatGameState(row.game_state))}</span>
            </div>
            ${renderMobileWhyDetails(row, "mobile-why-inline")}
          `,
        )}
      `,
    },
    { label: "Pitcher", cellClass: "col-pitcher", render: (row) => escapeHtml(row.pitcher_name || "-") },
    { label: "HR chance", cellClass: "col-probability", render: (row) => renderProbabilityCell(row) },
    {
      label: "Confidence",
      cellClass: "col-tier",
      render: (row) => `<span class="${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>`,
    },
  ];
  if (includeGameMeta) {
    columns.push({
      label: "Game Meta",
      cellClass: "col-game-meta",
      render: (row) => renderGameMeta(row),
    });
  }
  columns.push(
    {
      label: "Result",
      cellClass: "col-result",
      render: (row) => `
        ${renderMobileCellStack(
          "Result",
          `<span class="${resultClass(row.result_label)}">${escapeHtml(row.result_label)}</span>`,
          "stack-center",
        )}
      `,
    },
    {
      label: "Why",
      cellClass: "col-why",
      render: (row) => {
        const reasons = [row.top_reason_1, row.top_reason_2, row.top_reason_3].filter(Boolean);
        if (!reasons.length) {
          return '<span class="muted">No model reasons exported.</span>';
        }
        return `<ul class="reason-list">${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>`;
      },
    },
  );
  renderSimpleTable(targetId, columns, displayRows, emptyMessage, {
    mobileCards: false,
    tableClass: "mobile-picks-table",
  });
}

function renderLineupPanels(rows) {
  const target = document.getElementById("lineup-panels");
  const panels = (Array.isArray(rows) ? rows : []).filter((panel) =>
    Array.isArray(panel.teams) && panel.teams.some((teamPanel) =>
      Array.isArray(teamPanel.hitters) && teamPanel.hitters.some((hitter) => hitter.selected_for_pick),
    ),
  );
  if (!panels.length) {
    target.innerHTML = '<p class="empty-state">No current published picks are waiting on lineup context.</p>';
    return;
  }

  target.innerHTML = panels
    .map(
      (panel) => `
        <article class="lineup-card">
          <div class="lineup-card-head">
            <div>
              <p class="eyebrow">Game</p>
              <h3>${escapeHtml(panel.matchup || "")}</h3>
              <p class="muted">${escapeHtml(formatDateTime(panel.game_datetime))} | ${escapeHtml(panel.game_status || formatGameState(panel.game_state))}</p>
            </div>
          </div>
          <div class="lineup-team-grid">
            ${(panel.teams || [])
              .map(
                (teamPanel) => `
                  <section class="lineup-team">
                    <div class="lineup-team-head">
                      <strong>${escapeHtml(teamPanel.team)}</strong>
                      <span class="pill lineup-pill">${escapeHtml(formatLineupSource(teamPanel.lineup_source))}</span>
                    </div>
                    <ol class="lineup-list">
                      ${(teamPanel.hitters || [])
                        .map(
                          (hitter) => `
                            <li class="${hitter.selected_for_pick ? "is-selected" : ""}">
                              <span class="lineup-slot">${escapeHtml(hitter.batting_order || "-")}</span>
                              <span>${escapeHtml(hitter.batter_name || "")}</span>
                            </li>
                          `,
                        )
                        .join("")}
                    </ol>
                  </section>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderTierFilterControls(targetId, selectedTiers) {
  const target = document.getElementById(targetId);
  target.innerHTML = CONFIDENCE_TIERS.map((tier) => {
    const active = selectedTiers.has(tier);
    return `
      <button
        class="tier-filter-chip ${active ? "is-active" : ""}"
        type="button"
        data-tier-filter="${escapeHtml(tier)}"
        aria-pressed="${active ? "true" : "false"}"
      >
        <span class="${tierClass(tier)}">${escapeHtml(tier)}</span>
      </button>
    `;
  }).join("");
}

function renderHistoryDateOptions(historyDates, defaultDate) {
  const target = document.getElementById("history-date-filter");
  const rows = Array.isArray(historyDates) ? historyDates : [];
  const resolvedDefault = rows.includes(defaultDate) ? defaultDate : (rows[0] || ALL_DATES_FILTER_VALUE);
  state.selectedHistoryDate = resolvedDefault;
  target.innerHTML = [
    `<option value="${ALL_DATES_FILTER_VALUE}">All dates</option>`,
    ...rows.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(formatDate(value))}</option>`),
  ].join("");
  target.value = resolvedDefault || ALL_DATES_FILTER_VALUE;
}

function applyLatestPicksFilters() {
  if (!state.dashboard) {
    return;
  }
  state.filteredLatestPicks = filterRowsByTierSelection(state.dashboard.latest_picks, state.latestTierFilters);
  renderPicksTable("latest-picks-table", state.filteredLatestPicks, "No published picks match the selected confidence tiers.", { includeGameMeta: true });
}

function renderSeasonLeaders(rows) {
  renderSimpleTable(
    "leaderboard-table",
    [
      {
        label: "Player",
        render: (row) => `
          <div class="name-block">
            <strong>${escapeHtml(row.batter_name)}</strong>
            <span>${escapeHtml(row.team || "-")}</span>
          </div>
        `,
      },
      { label: "2026 HR", render: (row) => `<strong>${escapeHtml(formatWholeNumber(row.home_runs_2026))}</strong>` },
      { label: "PA", render: (row) => escapeHtml(formatWholeNumber(row.plate_appearances_2026)) },
      { label: "Games", render: (row) => escapeHtml(formatWholeNumber(row.games_2026)) },
    ],
    rows,
    "No 2026 season leaders are available yet.",
  );
}

function renderYesterdayRecap(dashboard) {
  const summaryTarget = document.getElementById("yesterday-summary");
  const previousDate = dashboard?.yesterday_homer_date || dashboard?.history_default_date || "";
  const rows = (Array.isArray(dashboard?.history) ? dashboard.history : [])
    .filter((row) => row.game_date === previousDate)
    .sort((left, right) => Number(left.rank) - Number(right.rank));

  if (!rows.length) {
    summaryTarget.innerHTML = "";
    document.getElementById("yesterday-table").innerHTML = `<p class="empty-state">${escapeHtml(DEFAULT_YESTERDAY_RECAP_EMPTY_MESSAGE)}</p>`;
    return;
  }

  const homerCount = rows.filter((row) => row.result_label === "HR").length;
  const missCount = rows.filter((row) => row.result_label === "No HR").length;
  const pendingCount = rows.filter((row) => row.result_label === "Pending").length;
  const hitRate = rows.length ? homerCount / rows.length : null;

  summaryTarget.innerHTML = [
    { label: "Date", value: formatDate(previousDate) },
    { label: "Record", value: `${formatWholeNumber(homerCount)}-${formatWholeNumber(missCount)}` },
    { label: "Hit rate", value: formatPercent(hitRate) },
    { label: "Pending", value: formatWholeNumber(pendingCount) },
  ]
    .map(
      (item) => `
        <article class="yesterday-stat">
          <span class="yesterday-stat-label">${escapeHtml(item.label)}</span>
          <strong class="yesterday-stat-value">${escapeHtml(item.value)}</strong>
        </article>
      `,
    )
    .join("");

  renderSimpleTable(
    "yesterday-table",
    [
      { label: "Rank", render: (row) => escapeHtml(formatWholeNumber(row.rank)) },
      { label: "Player", render: (row) => escapeHtml(row.batter_name) },
      { label: "Team", render: (row) => escapeHtml(row.team) },
      { label: "Pitcher", render: (row) => escapeHtml(row.pitcher_name || "-") },
      { label: "HR chance", render: (row) => escapeHtml(formatPercent(row.predicted_hr_probability)) },
      {
        label: "Confidence",
        render: (row) => `<span class="${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>`,
      },
      {
        label: "Result",
        render: (row) => `<span class="${resultClass(row.result_label)}">${escapeHtml(row.result_label)}</span>`,
      },
    ],
    rows,
    DEFAULT_YESTERDAY_RECAP_EMPTY_MESSAGE,
  );
}

function renderRefreshScheduleInline(schedule) {
  const target = document.getElementById("refresh-schedule-inline");
  target.textContent = buildRefreshScheduleInlineText(schedule);
}

function renderModelExplainer(explainer) {
  const button = document.getElementById("model-explainer-button");
  const title = document.getElementById("model-explainer-title");
  const summary = document.getElementById("model-explainer-summary");
  const list = document.getElementById("model-explainer-list");
  const features = Array.isArray(explainer?.features) ? explainer.features : [];

  title.textContent = explainer?.title || "Metric guide";
  summary.textContent = explainer?.summary || DEFAULT_MODEL_EXPLAINER_MESSAGE;

  if (!explainer?.available || !features.length) {
    button.hidden = true;
    list.innerHTML = `<p class="empty-state">${escapeHtml(DEFAULT_MODEL_EXPLAINER_MESSAGE)}</p>`;
    return;
  }

  button.hidden = false;
  list.innerHTML = features
    .map((feature) => {
      const strengthScore = feature.strength_score === null || feature.strength_score === undefined
        ? 0.25
        : Math.max(0.08, Math.min(1, Number(feature.strength_score)));
      return `
        <article class="model-metric-card">
          <div class="model-metric-top">
            <div>
              <strong>${escapeHtml(feature.label)}</strong>
              <p>${escapeHtml(feature.description || "")}</p>
            </div>
            <div class="model-metric-meta">
              <span class="metric-strength">${escapeHtml(feature.strength || "Included")}</span>
            </div>
          </div>
          <div class="metric-strength-bar" aria-hidden="true">
            <span style="width:${Math.round(strengthScore * 100)}%"></span>
          </div>
          <p class="metric-direction">${escapeHtml(feature.direction || "")}</p>
        </article>
      `;
    })
    .join("");
}

function applyHistoryFilters() {
  if (!state.dashboard) {
    return;
  }

  const searchValue = document.getElementById("history-search").value.trim().toLowerCase();
  const selectedDate = state.selectedHistoryDate || state.dashboard.history_default_date || ALL_DATES_FILTER_VALUE;
  const tierFilteredRows = filterRowsByTierSelection(state.dashboard.history, state.historyTierFilters);

  state.filteredHistory = tierFilteredRows.filter((row) => {
    const matchesDate = selectedDate === ALL_DATES_FILTER_VALUE || row.game_date === selectedDate;
    const haystack = [
      row.batter_name,
      row.team,
      row.opponent_team,
      row.pitcher_name,
      row.game_date,
    ]
      .join(" ")
      .toLowerCase();
    const matchesSearch = !searchValue || haystack.includes(searchValue);
    return matchesDate && matchesSearch;
  });

  renderPicksTable("history-table", state.filteredHistory, DEFAULT_HISTORY_EMPTY_MESSAGE);
}

function handleTierFilterToggle(event) {
  const button = event.target.closest("[data-tier-filter]");
  if (!button) {
    return;
  }

  const tier = normalizeTier(button.dataset.tierFilter);
  const group = button.closest(".tier-filter-row");
  if (!group || !CONFIDENCE_TIERS.includes(tier)) {
    return;
  }

  const selectedTiers = group.id === "latest-confidence-filters" ? state.latestTierFilters : state.historyTierFilters;
  if (selectedTiers.has(tier)) {
    selectedTiers.delete(tier);
  } else {
    selectedTiers.add(tier);
  }

  renderTierFilterControls(group.id, selectedTiers);
  if (group.id === "latest-confidence-filters") {
    applyLatestPicksFilters();
  } else {
    applyHistoryFilters();
  }
}

async function loadDashboard() {
  const response = await fetch("./data/dashboard.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load dashboard data (${response.status})`);
  }

  state.dashboard = await response.json();

  document.getElementById("data-note").textContent = state.dashboard.data_note;
  document.getElementById("latest-date").textContent = formatDate(state.dashboard.latest_available_date);
  document.getElementById("generated-at").textContent = `Refreshed ${formatDateTime(state.dashboard.generated_at)}`;
  document.getElementById("model-family").textContent = formatModelValue(
    state.dashboard.model_family || state.dashboard.model_explainer?.model_family,
  );
  document.getElementById("feature-profile").textContent = formatModelValue(
    state.dashboard.feature_profile || state.dashboard.model_explainer?.feature_profile,
  );
  document.getElementById("usable-status").textContent = `Tracking since ${formatDate(state.dashboard.tracking_start_date)}`;

  renderDashboardAlerts(state.dashboard.operational_alerts);
  renderTierLegend(state.dashboard.tier_guide);
  renderOverviewCards(state.dashboard);
  renderConfidenceTable(state.dashboard.confidence_summary);
  renderTierFilterControls("latest-confidence-filters", state.latestTierFilters);
  renderTierFilterControls("history-confidence-filters", state.historyTierFilters);
  renderHistoryDateOptions(state.dashboard.history_dates, state.dashboard.history_default_date);
  applyLatestPicksFilters();
  renderLineupPanels(state.dashboard.lineup_panels || []);
  renderSeasonLeaders(state.dashboard.season_hr_leaders_2026 || []);
  renderYesterdayRecap(state.dashboard);
  renderRefreshScheduleInline(state.dashboard.refresh_schedule);
  renderModelExplainer(state.dashboard.model_explainer);
  applyHistoryFilters();
}

function handleLoadError(error) {
  document.getElementById("data-note").textContent = error.message;
  document.getElementById("overview-cards").innerHTML = `
    <article class="stat-card">
      <p class="eyebrow">Dashboard error</p>
      <span class="value">Unavailable</span>
      <p class="subtext">${escapeHtml(error.message)}</p>
    </article>
  `;
}

function setManualRefreshStatus(message, kind = "neutral") {
  const target = document.getElementById("manual-refresh-status");
  target.textContent = message;
  target.dataset.kind = kind;
}

function setManualButtonsDisabled(disabled) {
  document.getElementById("manual-settle-button").disabled = disabled;
  document.getElementById("manual-prepare-button").disabled = disabled;
  document.getElementById("manual-publish-button").disabled = disabled;
}

function manualModeLabel(mode) {
  if (mode === "publish") {
    return "prediction";
  }
  return mode;
}

async function triggerManualRefresh(mode) {
  const keyInput = document.getElementById("manual-refresh-key");
  const adminKey = keyInput.value.trim();
  const modeLabel = manualModeLabel(mode);
  if (!adminKey) {
    setManualRefreshStatus("Enter the admin key before triggering a manual refresh.", "error");
    keyInput.focus();
    return;
  }

  localStorage.setItem(MANUAL_REFRESH_KEY_STORAGE, adminKey);
  setManualButtonsDisabled(true);
  setManualRefreshStatus(`Triggering ${modeLabel} refresh...`, "pending");

  try {
    const response = await fetch("/api/manual-refresh", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mode, adminKey }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `Manual ${modeLabel} refresh failed.`);
    }

    const workflowText = payload.workflowUrl ? ` Track it at ${payload.workflowUrl}` : "";
    setManualRefreshStatus(`${payload.message}${workflowText}`, "success");
  } catch (error) {
    setManualRefreshStatus(error.message || `Manual ${modeLabel} refresh failed.`, "error");
  } finally {
    setManualButtonsDisabled(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const modelExplainerDialog = document.getElementById("model-explainer-dialog");
  document.getElementById("model-explainer-button").addEventListener("click", () => {
    modelExplainerDialog.showModal();
  });
  document.getElementById("model-explainer-close").addEventListener("click", () => {
    modelExplainerDialog.close();
  });
  document.getElementById("history-search").addEventListener("input", applyHistoryFilters);
  document.getElementById("history-date-filter").addEventListener("change", (event) => {
    state.selectedHistoryDate = event.target.value;
    applyHistoryFilters();
  });
  document.getElementById("latest-confidence-filters").addEventListener("click", handleTierFilterToggle);
  document.getElementById("history-confidence-filters").addEventListener("click", handleTierFilterToggle);
  const savedKey = localStorage.getItem(MANUAL_REFRESH_KEY_STORAGE);
  if (savedKey) {
    document.getElementById("manual-refresh-key").value = savedKey;
  }
  document.getElementById("manual-settle-button").addEventListener("click", () => {
    triggerManualRefresh("settle");
  });
  document.getElementById("manual-prepare-button").addEventListener("click", () => {
    triggerManualRefresh("prepare");
  });
  document.getElementById("manual-publish-button").addEventListener("click", () => {
    triggerManualRefresh("publish");
  });
  loadDashboard().catch(handleLoadError);
});
