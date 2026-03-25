const state = {
  dashboard: null,
  filteredHistory: [],
};

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

function renderOverviewCards(overview) {
  const cards = [
    {
      label: "Tracked hit rate",
      value: formatPercent(overview.tracked_hit_rate),
      subtext: `${overview.tracked_homers} home runs across ${overview.settled_picks.toLocaleString()} settled picks.`,
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
      label: "Tracking mode",
      value: "Forward only",
      subtext: "Public performance starts on March 25, 2026 and excludes 2024 and 2025 picks.",
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

function renderSimpleTable(targetId, columns, rows) {
  const target = document.getElementById(targetId);
  if (!rows.length) {
    target.innerHTML = '<p class="empty-state">No rows available.</p>';
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

function renderPicksTable(targetId, rows) {
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

  renderPicksTable("history-table", state.filteredHistory);
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
  document.getElementById("usable-status").textContent = `Tracking since ${state.dashboard.tracking_start_date}`;

  renderOverviewCards(state.dashboard.overview);
  renderTopKTable(state.dashboard.top_k_summary);
  renderConfidenceTable(state.dashboard.confidence_summary);
  renderPicksTable("latest-picks-table", state.dashboard.latest_picks);
  renderLeaderboard(state.dashboard.player_leaderboard);
  renderSuccesses(state.dashboard.recent_successes);
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

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("history-search").addEventListener("input", applyHistoryFilters);
  document.getElementById("confidence-filter").addEventListener("change", applyHistoryFilters);
  loadDashboard().catch(handleLoadError);
});
