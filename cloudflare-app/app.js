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
const DEFAULT_YESTERDAY_SUCCESSES_EMPTY_MESSAGE = "No published picks homered yesterday.";
const DEFAULT_MODEL_EXPLAINER_MESSAGE = "Metric details are not available for the current dashboard build.";
const MANUAL_REFRESH_KEY_STORAGE = "manualRefreshKey";
const CONFIDENCE_TIERS = ["elite", "strong", "watch", "longshot"];
const ALL_DATES_FILTER_VALUE = "__all_dates__";

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

function renderOverviewCards(overview, confidenceSummary, refreshSchedule) {
  const eliteSummary = findConfidenceSummary(confidenceSummary, "elite");
  const elitePicks = eliteSummary?.picks ?? null;
  const eliteHomers = eliteSummary?.homers ?? null;
  const cards = [
    {
      label: "Elite hit rate",
      value: formatPercent(eliteSummary?.hit_rate),
      subtext:
        elitePicks && eliteHomers !== null
          ? `${formatWholeNumber(eliteHomers)} home runs across ${formatWholeNumber(elitePicks)} elite picks.`
          : "No settled elite picks yet.",
    },
    {
      label: "Published picks",
      value: formatWholeNumber(overview.tracked_picks),
      subtext: `${formatWholeNumber(overview.tracked_dates)} tracked slate dates since public tracking started.`,
    },
    {
      label: "Settled picks",
      value: formatWholeNumber(overview.settled_picks),
      subtext: "Only settled picks count toward the public hit-rate tracker.",
    },
    {
      label: "Open picks",
      value: formatWholeNumber(overview.open_picks),
      subtext: "These picks have been published but do not have a recorded result yet.",
    },
    {
      label: "Latest slate",
      value: formatWholeNumber(overview.latest_slate_size),
      subtext: "Current published picks for the latest tracked date.",
    },
    {
      label: "Refresh schedule",
      value: "15 min cadence",
      subtext: buildRefreshScheduleSummary(refreshSchedule),
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

function renderSimpleTable(targetId, columns, rows, emptyMessage = "No rows available.") {
  const target = document.getElementById(targetId);
  if (!rows.length) {
    target.innerHTML = `<p class="empty-state">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const headers = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns.map((column) => `<td>${column.render(row)}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  target.innerHTML = `<table><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table>`;
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

function renderPicksTable(targetId, rows, emptyMessage) {
  renderSimpleTable(
    targetId,
    [
      { label: "Date", render: (row) => escapeHtml(formatDate(row.game_date)) },
      { label: "Rank", render: (row) => `<strong>${escapeHtml(row.rank)}</strong>` },
      {
        label: "Hitter",
        render: (row) => `
          <div class="name-block">
            <strong>${escapeHtml(row.batter_name)}</strong>
            <span>${escapeHtml(row.team)} vs ${escapeHtml(row.opponent_team || "-")}</span>
            <span>${escapeHtml(formatLineupSource(row.lineup_source))}${row.batting_order ? `, batting ${escapeHtml(row.batting_order)}` : ""} | ${escapeHtml(formatGameState(row.game_state))}</span>
          </div>
        `,
      },
      { label: "Pitcher", render: (row) => escapeHtml(row.pitcher_name || "-") },
      { label: "Score", render: (row) => escapeHtml(formatScore(row.predicted_hr_score)) },
      {
        label: "Tier",
        render: (row) => `<span class="${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>`,
      },
      {
        label: "Result",
        render: (row) => `<span class="${resultClass(row.result_label)}">${escapeHtml(row.result_label)}</span>`,
      },
      {
        label: "Why",
        render: (row) => {
          const reasons = [row.top_reason_1, row.top_reason_2, row.top_reason_3].filter(Boolean);
          if (!reasons.length) {
            return '<span class="muted">No model reasons exported.</span>';
          }
          return `<ul class="reason-list">${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>`;
        },
      },
    ],
    rows,
    emptyMessage,
  );
}

function renderLineupPanels(rows) {
  const target = document.getElementById("lineup-panels");
  const panels = Array.isArray(rows) ? rows : [];
  if (!panels.length) {
    target.innerHTML = '<p class="empty-state">No active lineup context is available for the current slate.</p>';
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
  renderPicksTable("latest-picks-table", state.filteredLatestPicks, "No published picks match the selected confidence tiers.");
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

function renderSuccesses(rows) {
  renderSimpleTable(
    "success-table",
    [
      { label: "Date", render: (row) => escapeHtml(formatDate(row.game_date)) },
      { label: "Player", render: (row) => escapeHtml(row.batter_name) },
      { label: "Team", render: (row) => escapeHtml(row.team) },
      { label: "Pitcher", render: (row) => escapeHtml(row.pitcher_name || "-") },
      { label: "Rank", render: (row) => escapeHtml(row.rank) },
      { label: "Score", render: (row) => escapeHtml(formatScore(row.predicted_hr_score)) },
    ],
    rows,
    DEFAULT_YESTERDAY_SUCCESSES_EMPTY_MESSAGE,
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
  renderOverviewCards(state.dashboard.overview, state.dashboard.confidence_summary, state.dashboard.refresh_schedule);
  renderConfidenceTable(state.dashboard.confidence_summary);
  renderTierFilterControls("latest-confidence-filters", state.latestTierFilters);
  renderTierFilterControls("history-confidence-filters", state.historyTierFilters);
  renderHistoryDateOptions(state.dashboard.history_dates, state.dashboard.history_default_date);
  applyLatestPicksFilters();
  renderLineupPanels(state.dashboard.lineup_panels || []);
  renderSeasonLeaders(state.dashboard.season_hr_leaders_2026 || []);
  renderSuccesses(state.dashboard.recent_successes || []);
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
