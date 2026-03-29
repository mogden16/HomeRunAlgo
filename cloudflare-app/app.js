const state = {
  dashboard: null,
  filteredHistory: [],
};

const DEFAULT_LATEST_PICKS_EMPTY_MESSAGE =
  "Today's public picks have not been posted yet. Next publish windows are 11:00 AM, 1:00 PM, 3:00 PM, and 6:00 PM ET.";
const MANUAL_REFRESH_KEY_STORAGE = "manualRefreshKey";

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

function tierClass(value) {
  return `tier-tag tier-${String(value || "").toLowerCase()}`;
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

function findConfidenceSummary(rows, tier) {
  return (rows || []).find((row) => String(row.confidence_tier || "").toLowerCase() === tier) || null;
}

function formatWholeNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString();
}

function buildRefreshScheduleSummary(schedule) {
  const runs = Array.isArray(schedule?.runs) ? schedule.runs : [];
  const settleRun = runs.find((run) => run.type === "settle");
  const prepareRun = runs.find((run) => run.type === "prepare");
  const publishRuns = runs.filter((run) => run.type === "publish");
  const publishTimes = publishRuns.map((run) => run.time_et).filter(Boolean);

  if (!settleRun && !prepareRun && !publishTimes.length) {
    return "Schedule unavailable.";
  }

  const publishText = publishTimes.length ? `Publish runs at ${publishTimes.join(", ")}.` : "";
  const settleText = settleRun?.time_et ? `Settle run at ${settleRun.time_et}.` : "";
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
    parts.push(`${settleRun.time_et} settle.`);
  }
  if (prepareRun?.time_et) {
    parts.push(`${prepareRun.time_et} prepare.`);
  }
  if (publishTimes.length) {
    parts.push(`Publish runs at ${publishTimes.join(", ")}.`);
  }
  return parts.join(" ");
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
      value: overview.tracked_picks.toLocaleString(),
      subtext: `${overview.tracked_dates.toLocaleString()} tracked slate dates since public tracking started.`,
    },
    {
      label: "Settled picks",
      value: overview.settled_picks.toLocaleString(),
      subtext: "Only settled picks count toward the public hit-rate tracker.",
    },
    {
      label: "Open picks",
      value: overview.open_picks.toLocaleString(),
      subtext: "These picks have been published but do not have a recorded result yet.",
    },
    {
      label: "Latest slate",
      value: overview.latest_slate_size.toLocaleString(),
      subtext: "Current published picks for the latest tracked date.",
    },
    {
      label: "Refresh schedule",
      value: `${Array.isArray(refreshSchedule?.runs) ? refreshSchedule.runs.length : 0} runs/day`,
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

function renderTopKTable(rows) {
  renderSimpleTable(
    "top-k-table",
    [
      { label: "Slice", render: (row) => `<strong>Top ${escapeHtml(row.top_k)}</strong>` },
      { label: "Dates", render: (row) => escapeHtml(row.dates) },
      { label: "Picks", render: (row) => escapeHtml(row.picks) },
      { label: "Homers", render: (row) => escapeHtml(row.homers) },
      { label: "Hit rate", render: (row) => escapeHtml(formatPercent(row.hit_rate)) },
      { label: "Avg score", render: (row) => escapeHtml(formatScore(row.avg_score)) },
    ],
    rows,
  );
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

function renderLeaderboard(rows) {
  renderSimpleTable(
    "leaderboard-table",
    [
      {
        label: "Player",
        render: (row) => `
          <div class="name-block">
            <strong>${escapeHtml(row.batter_name)}</strong>
            <span>${escapeHtml(row.team)}</span>
          </div>
        `,
      },
      { label: "Picks", render: (row) => escapeHtml(row.picks) },
      { label: "Homers", render: (row) => escapeHtml(row.homers) },
      { label: "Hit rate", render: (row) => escapeHtml(formatPercent(row.hit_rate)) },
      { label: "Avg score", render: (row) => escapeHtml(formatScore(row.avg_score)) },
    ],
    rows,
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
  );
}

function renderRefreshScheduleInline(schedule) {
  const target = document.getElementById("refresh-schedule-inline");
  target.textContent = buildRefreshScheduleInlineText(schedule);
}

function applyHistoryFilters() {
  if (!state.dashboard) {
    return;
  }

  const searchValue = document.getElementById("history-search").value.trim().toLowerCase();
  const tierValue = document.getElementById("confidence-filter").value.trim().toLowerCase();

  state.filteredHistory = state.dashboard.history.filter((row) => {
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
    const matchesTier = !tierValue || String(row.confidence_tier).toLowerCase() === tierValue;
    return matchesSearch && matchesTier;
  });

  renderPicksTable("history-table", state.filteredHistory, "No published picks match those filters.");
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
  document.getElementById("model-family").textContent = state.dashboard.model_family;
  document.getElementById("usable-status").textContent = `Tracking since ${formatDate(state.dashboard.tracking_start_date)}`;

  renderOverviewCards(state.dashboard.overview, state.dashboard.confidence_summary, state.dashboard.refresh_schedule);
  renderTopKTable(state.dashboard.top_k_summary);
  renderConfidenceTable(state.dashboard.confidence_summary);
  renderPicksTable("latest-picks-table", state.dashboard.latest_picks, DEFAULT_LATEST_PICKS_EMPTY_MESSAGE);
  renderLeaderboard(state.dashboard.player_leaderboard);
  renderSuccesses(state.dashboard.recent_successes);
  renderRefreshScheduleInline(state.dashboard.refresh_schedule);
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
  document.getElementById("history-search").addEventListener("input", applyHistoryFilters);
  document.getElementById("confidence-filter").addEventListener("change", applyHistoryFilters);
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
