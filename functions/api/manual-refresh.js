const DEFAULT_REPO = "mogden16/HomeRunAlgo";
const DEFAULT_WORKFLOW = "manual-live-refresh.yml";
const ALLOWED_MODES = new Set(["settle", "prepare", "publish"]);

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

export async function onRequestPost(context) {
  const { request, env } = context;
  let payload;

  try {
    payload = await request.json();
  } catch {
    return jsonResponse({ ok: false, error: "Invalid JSON payload." }, 400);
  }

  const mode = String(payload?.mode || "").trim().toLowerCase();
  const adminKey = String(payload?.adminKey || "");

  if (!ALLOWED_MODES.has(mode)) {
    return jsonResponse({ ok: false, error: "Unsupported refresh mode." }, 400);
  }

  if (!env.MANUAL_REFRESH_KEY || !env.GITHUB_WORKFLOW_TOKEN) {
    return jsonResponse(
      {
        ok: false,
        error: "Manual refresh is not configured. Set MANUAL_REFRESH_KEY and GITHUB_WORKFLOW_TOKEN in Cloudflare Pages.",
      },
      500,
    );
  }

  if (adminKey !== env.MANUAL_REFRESH_KEY) {
    return jsonResponse({ ok: false, error: "Unauthorized." }, 401);
  }

  const repository = env.GITHUB_REPOSITORY || DEFAULT_REPO;
  const workflow = env.GITHUB_WORKFLOW_FILE || DEFAULT_WORKFLOW;
  const ref = env.GITHUB_WORKFLOW_REF || "master";

  const dispatchResponse = await fetch(
    `https://api.github.com/repos/${repository}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_WORKFLOW_TOKEN}`,
        "Content-Type": "application/json",
        "User-Agent": "HomeRunAlgoManualRefresh",
      },
      body: JSON.stringify({
        ref,
        inputs: {
          mode,
        },
      }),
    },
  );

  if (!dispatchResponse.ok) {
    const errorText = await dispatchResponse.text();
    return jsonResponse(
      {
        ok: false,
        error: `GitHub workflow dispatch failed (${dispatchResponse.status}).`,
        details: errorText,
      },
      502,
    );
  }

  return jsonResponse(
    {
      ok: true,
      mode,
      workflowUrl: `https://github.com/${repository}/actions/workflows/${workflow}`,
      message: `Triggered ${mode === "publish" ? "prediction" : mode} refresh workflow.`,
    },
    202,
  );
}

export async function onRequestGet() {
  return jsonResponse({ ok: false, error: "Use POST." }, 405);
}
