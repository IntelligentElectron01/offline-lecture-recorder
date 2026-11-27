"use strict";
// Lecture Recorder UI. Plain JS, no external code. All text from the server is inserted with
// textContent (never innerHTML), so transcript text can't inject markup.

const $ = (sel) => document.querySelector(sel);
const root = $("#view");
let pollTimer = null;   // per-view polling (exam material)
let routeSeq = 0;
// Each view renders through its own stage, taken before its first await: a slower, older render
// (e.g. a re-render after a delete elsewhere) can't replace the page of a newer route.
// Empty slots (null, false) are skipped: replaceChildren would show them as the text "null".
const stage = (seq = routeSeq) => ({ live: () => seq === routeSeq, replaceChildren: (...n) => { if (seq === routeSeq) root.replaceChildren(...n.filter((x) => x != null && x !== false)); } });

// Polling runs only while the tab is visible (efficiency rule): hidden tabs make no requests.
const polls = new Set();
function startPoll(fn, ms) {
  const p = { fn, ms, id: document.hidden ? null : setInterval(fn, ms) };
  polls.add(p);
  return p;
}
function stopPoll(p) { if (p) { clearInterval(p.id); polls.delete(p); } }
document.addEventListener("visibilitychange", () => {
  for (const p of polls) {
    clearInterval(p.id); p.id = null;
    if (!document.hidden) { p.fn(); p.id = setInterval(p.fn, p.ms); }
  }
});
let lastRecording = null;

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

// What this OS offers (a Mac has no per-app recording) and its advice when the microphone is silent (a Mac's
// microphone permission). Asked once; the REC bar on every page uses the hint.
let micHint = "";
const capabilities = api("/api/capabilities").catch(() => ({}));
capabilities.then((c) => { micHint = c.mic_hint || ""; });

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (init.method && init.method !== "GET") init.headers["X-Lecture-Recorder"] = "1";
  if (init.json !== undefined) { init.body = JSON.stringify(init.json); init.headers["Content-Type"] = "application/json"; delete init.json; }
  const r = await fetch(path, init);
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw Object.assign(new Error((body && body.detail) || `${r.status} ${r.statusText}`), { status: r.status, body });
  return body;
}

// action = { label, run }: a button in the toast (e.g. Undo). It stays longer, and not while hovered or focused.
function toast(msg, action) {
  const t = $("#toast");
  t.replaceChildren(msg);
  if (action) {
    const b = h("button", { type: "button", class: "link toast-action", text: action.label });
    b.addEventListener("click", () => { if (action.run() !== false) t.hidden = true; });  // false: not started, try again
    t.append(" ", b);
  }
  t.hidden = false;
  toast.ms = action ? 8000 : 3500;
  toast.later();
}
toast.later = () => { const t = $("#toast"); clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.hidden = true), toast.ms); };  // a one-shot, not a poll
$("#toast").addEventListener("mouseenter", () => clearTimeout(toast.timer));
$("#toast").addEventListener("focusin", () => clearTimeout(toast.timer));
$("#toast").addEventListener("mouseleave", () => { if (!$("#toast").contains(document.activeElement)) toast.later(); });
$("#toast").addEventListener("focusout", (e) => { if (!$("#toast").matches(":hover") && !$("#toast").contains(e.relatedTarget)) toast.later(); });
const hms = (s) => { s = Math.max(0, Math.floor(s || 0)); return [s / 3600, (s % 3600) / 60, s % 60].map((n) => String(Math.floor(n)).padStart(2, "0")).join(":"); };
// A date with no time (an import dated by its file name) is that day here, shown without a time.
const DAY_ONLY = /^\d{4}-\d{2}-\d{2}$/;
const fmtDate = (iso) => !iso ? "" : DAY_ONLY.test(iso) ? fmtDay(iso)
  : new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
function fmtDay(iso) {
  return iso ? new Date(DAY_ONLY.test(iso) ? `${iso}T00:00` : iso).toLocaleDateString([], { dateStyle: "medium" }) : "";
}
const fileUrl = (c, s, f) => `/files/${encodeURIComponent(c)}/${encodeURIComponent(s)}/${encodeURIComponent(f)}`;
const STATE_LABEL = { processed: "Transcribed", queued: "Queued", running: "Processing", failed: "Failed",
  "not processed": "Not processed", recording: "Recording", unfinished: "Unfinished", "no audio": "No usable audio" };
const chip = (state) => h("span", { class: `chip ${state.replace(" ", "-")}`, text: STATE_LABEL[state] || state });

// ------------------------------------------------------------------ theme
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("#theme-toggle").textContent = t === "dark" ? "Light" : "Dark";
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked */ }
  applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  });
})();

// ------------------------------------------------------------------ recording bar (every page)
let unanswered = 0;  // status polls in a row that got no answer at all
async function refreshRecording() {
  let s;
  // No answer 3 times in a row (~6 s; the server is on this computer): the app is closed. The poll goes
  // on (visible tab only), so the page comes back by itself if the app is still or again running.
  try { s = await api("/api/record/status"); } catch (e) {
    if (e.status === 401) { showClosed("This tab was opened before the app was restarted. Start run.bat again: it reopens the app."); return; }
    if (e.status === undefined && ++unanswered >= 3) showClosed();
    return;
  }
  unanswered = 0;
  if ($("#closed")) { location.reload(); return; }
  trashChanged(s.trash_version);
  onProcessing(s.processing);
  const bar = $("#recbar");
  $("#rec-reminder").textContent = s.reminder;
  if (!s.recording) {
    bar.hidden = true;
    if (lastRecording) { // it just ended (Stop, time limit, ...): show the session
      const done = lastRecording; lastRecording = null;
      location.hash = `#/session/${done.class}/${done.session}`;
      toast("Recording saved.");
    }
    return;
  }
  lastRecording = { class: s.class, session: s.session };
  bar.hidden = false;
  const paused = s.state !== "recording";
  bar.classList.toggle("paused", paused);
  $("#rec-label").textContent = paused ? "PAUSED" : "REC";
  $("#rec-time").textContent = hms(s.recorded_s);
  $("#rec-class").textContent = s.class_name ? `· ${s.class_name}` : "";
  $("#rec-reason").textContent = paused && s.reason ? `(${String(s.reason).replaceAll("_", " ")})` : "";
  $("#rec-pause").hidden = paused; $("#rec-resume").hidden = !paused;
  const level = $("#rec-level"); // loudest 100 ms block of the last second, -60..0 dBFS
  level.hidden = paused;
  level.value = Math.max(-60, Math.min(0, Number(s.level_dbfs ?? -120)));
  const app = s.source && s.source.kind === "app" ? s.source.app_name : null;
  $("#rec-source").textContent = app ? `· from ${app}${s.source.include_mic ? " + microphone" : ""}` : "";
  level.title = `${app ? `${app} audio` : "Microphone input"} level: ${Math.round(Number(s.level_dbfs ?? -120))} dBFS`;
  $("#rec-noaudio").textContent = app ? `No audio from ${app} — is it muted or turned down in Windows' Volume Mixer? (Silence in the meeting shows this too.)`
    : `No audio detected — check your microphone${micHint ? `. ${micHint}` : ""}`;
  $("#rec-noaudio").hidden = paused || !s.no_audio;
}
async function recCommand(cmd, msg) {
  try { await api(`/api/record/${cmd}`, { method: "POST" }); if (msg) toast(msg); } catch (e) { toast(e.message); }
  refreshRecording();
}
$("#rec-pause").addEventListener("click", () => recCommand("pause"));
$("#rec-resume").addEventListener("click", () => recCommand("resume"));
$("#rec-bookmark").addEventListener("click", () => recCommand("bookmark", "Bookmark added."));
$("#rec-stop").addEventListener("click", async (ev) => {
  ev.target.disabled = true; ev.target.textContent = "Saving…";
  await recCommand("stop");
  ev.target.disabled = false; ev.target.textContent = "Stop";
});
startPoll(refreshRecording, 2000);

// ------------------------------------------------------------------ quit
const JOB_LABEL = { transcribe: "Transcribing", summarize: "Summarizing", embed: "Indexing for search", exam: "Making exam material" };
async function quitApp() {
  let plan;
  try { plan = await api("/api/quit"); } catch (e) { toast(e.message); return; }
  const lines = [
    plan.recording && `• The recording "${plan.recording.title}" stops and is saved.`,
    ...plan.running.map((j) => `• ${JOB_LABEL[j.kind] || "Processing"} "${j.title}" stops. It starts again from the beginning the next time you process.`),
    plan.queued && `• ${plan.queued} queued job${plan.queued === 1 ? "" : "s"} wait until you open the app again.`].filter(Boolean);
  if (!confirm(lines.length ? `Quit Lecture Recorder?\n\n${lines.join("\n")}` : "Quit Lecture Recorder?")) return;
  let r;
  try { r = await api("/api/quit", { method: "POST", json: { confirm: plan } }); } catch (e) { toast(e.message); return; }
  for (const p of [...polls]) stopPoll(p);  // closed on purpose: nothing to wait for
  showClosed(r.error);
}
$("#quit").addEventListener("click", quitApp);
function showClosed(error) {
  for (const p of [...polls]) if (p.fn !== refreshRecording) stopPoll(p);
  document.body.replaceChildren(h("main", { id: "closed" }, h("h1", { text: "Lecture Recorder has closed." }),
    error ? h("p", { class: "warning", role: "alert", text: error }) : null,
    lastRecording ? h("p", { class: "warning", role: "alert", text: "A recording was running when the app stopped answering. It keeps recording on its own "
      + "and is saved when it stops (at the class's time limit); open the app again to stop it now." }) : null,
    h("p", { class: "muted", text: "Your recordings and notes are saved. To open it again, start run.bat." }),
    h("p", {}, h("a", { href: "/", text: "Reload this page" }), h("span", { class: "muted", text: " once it is running again." }))));
}

// ------------------------------------------------------------------ deletes elsewhere
// A delete, restore or permanent delete (another tab, the terminal) changes trash_version, which the
// status poll above already returns: no extra requests. Pages then re-render from fresh data.
let trashVersion = null;
let shown = { kind: "home" };  // what route() last showed
// The user is typing, or left unsaved input in the page (a half-typed name, a ticked unit, a chosen file).
const changed = (el) => el.matches("select") ? el.selectedIndex !== Math.max(0, [...el.options].findIndex((o) => o.defaultSelected))
  : el.matches("[type=checkbox]") ? el.checked !== el.defaultChecked : el.value !== el.defaultValue;
const editing = () => document.activeElement?.matches?.("input:not([type=checkbox]):not([type=file]), textarea, [contenteditable=true]")
  || [...root.querySelectorAll("input:not([data-transient]), textarea, select")].some(changed) || !!root.querySelector("[contenteditable=true]");
function trashChanged(v) {
  if (v === undefined || v === trashVersion) return;
  if (trashVersion === null) { trashVersion = v; return; }
  if (editing()) return;  // don't wipe what the user is typing; checked again on the next poll
  trashVersion = v;
  if (shown.kind !== "session" || root.querySelector("#gone")) { route(); return; }  // also: a deleted lecture was restored
  // A lecture page re-renders only if the lecture itself is gone (keeps playback and scroll otherwise);
  // a rename elsewhere just updates its heading.
  api(`/api/sessions/${encodeURIComponent(shown.slug)}/${encodeURIComponent(shown.name)}`)
    .then((s) => { const t = $("#lecture-title"); if (t) t.textContent = s.title; })
    .catch((e) => { if (e.status === 404) route(); });
}
// After this page's own delete/restore: take the new version so it doesn't re-render a second time.
const tookVersion = (r) => { if (r && r.trash_version !== undefined) trashVersion = r.trash_version; return r; };

// ------------------------------------------------------------------ processing progress
// Driven by the status poll above (s.processing): no timers of its own. A bar fills only from a real counter
// the job reports (seconds of audio, parts, passages, pyannote's chunk counts); otherwise it drifts
// (indeterminate). Between two polls a CSS transition moves the fill to the newest real value, so it trails
// the job by one poll and never runs ahead of it.
const STAGE = { prepare: "Checking the audio", gpu_wait: "Waiting for the GPU", load: "Loading the speech model",
  transcribe: "Transcribing", speakers: "Labeling speakers", finish: "Saving files", map: "Summarizing",
  condense: "Condensing notes", combine: "Combining notes", embed: "Indexing for search",
  exam: "Writing exam material", glossary: "Writing the glossary" };
const SUB = { segmentation: "finding speech turns", embeddings: "matching voices", clustering: "grouping voices" };
const WAIT = { "recording in progress": "while recording", "on battery": "on battery" };
const waitText = (w) => WAIT[w] || w;
let lastActive = new Map(), lastPollAt = 0, recentFailed = new Set(), viewStale = false, afterFade = null;
const jobKey = (j) => `${j.class}/${j.session}`;

function describe(j) {
  if (j.state === "queued") return { chip: j.waiting ? `Waiting · ${waitText(j.waiting)}` : j.position === 1 ? "Queued · next up" : `Queued · ${j.position - 1} ahead` };
  const p = j.progress || {}, f = p.fraction ?? null, name = STAGE[p.stage] || JOB_LABEL[j.kind] || "Processing";
  const pct = f === null ? null : Math.min(99, Math.floor(f * 100));  // 100% only once the job is done
  const count = p.total == null ? null : p.unit === "s" ? `${hms(p.done)} of ${hms(p.total)} audio`
    : p.unit === "part" ? `part ${Math.min(p.done + 1, p.total)} of ${p.total}`
    : p.unit === "passage" ? `${p.done} of ${p.total} passages` : p.unit === "unit" ? `unit ${Math.min(p.done + 1, p.total)} of ${p.total}` : null;
  const eta = p.eta_s == null ? null : p.eta_s < 60 ? "under a minute left in this step" : `about ${Math.ceil(p.eta_s / 60)} min left in this step`;
  return { name, pct, f, label: pct === null ? name : `${name} · ${pct}%`, meta: [p.sub && SUB[p.sub], count, eta].filter(Boolean).join(" · "),
    stage: `${j.kind}/${p.stage}/${p.sub}` };
}
const jobChip = (j) => h("span", { class: `chip ${j.waiting ? "waiting" : "queued"}`, text: describe(j).chip });

// Moves a bar to the job's latest real value: smoothly only within the same step, right after the previous
// poll (a tab coming back from hidden jumps), and never backwards.
function setFill(holder, bar, d) {
  const fill = bar.querySelector(".pbar-fill");
  bar.classList.toggle("indeterminate", d.f === null);
  const smooth = holder.dataset.stage === d.stage && Date.now() - lastPollAt < 4000 && d.f !== null && d.f >= Number(holder.dataset.f || 0);
  fill.style.setProperty("--dur", smooth ? "2s" : "0s");
  if (d.f !== null) fill.style.setProperty("--p", d.f);
  holder.dataset.f = d.f ?? 0;
}

let progressIds = 0;
function progressEl(j) {
  const id = `pl-${++progressIds}`;
  const el = h("div", { class: "progress", "data-progress": jobKey(j), "data-job": j.id },
    h("div", { class: "progress-head" }, h("span", { class: "progress-label", id }), h("span", { class: "progress-meta" })),
    h("div", { class: "pbar", role: "progressbar", "aria-labelledby": id, "aria-valuemin": "0", "aria-valuemax": "100" }, h("span", { class: "pbar-fill" })));
  updateProgress(el, j, false);
  return el;
}
function updateProgress(el, j, announce = true) {
  const d = describe(j), bar = el.querySelector(".pbar");
  el.querySelector(".progress-label").textContent = d.label;
  el.querySelector(".progress-meta").textContent = d.meta;
  setFill(el, bar, d);
  if (d.pct !== null) bar.setAttribute("aria-valuenow", d.pct); else bar.removeAttribute("aria-valuenow");
  bar.setAttribute("aria-valuetext", [d.label.replace(" · ", ", ").replace("%", " percent"), d.meta].filter(Boolean).join(", "));
  if (announce && el.dataset.stage !== d.stage) $("#progress-live").textContent = `${j.title || j.session}: ${d.name}`;
  el.dataset.stage = d.stage;
}
// The job left the queue: "Done" (or "Failed"), then the row fades; the page re-renders after that.
function finishProgress(el, failed) {
  const bar = el.querySelector(".pbar");
  bar.classList.remove("indeterminate");
  el.classList.add(failed ? "failed" : "done");
  el.querySelector(".progress-label").textContent = failed ? "Failed" : "Done";
  el.querySelector(".progress-meta").textContent = "";
  if (!failed) bar.setAttribute("aria-valuenow", "100");
  bar.setAttribute("aria-valuetext", failed ? "Failed" : "Done");
  requestAnimationFrame(() => el.classList.add("leaving"));
}

// Lecture page: which step of Transcribe -> Speakers -> Summary -> Search the lecture is on.
const STEPS = ["Transcribe", "Speakers", "Summary", "Search"];
const stepAt = (j) => j.kind === "summarize" ? 2 : j.kind === "embed" ? 3 : ["speakers", "finish"].includes(j.progress?.stage) ? 1 : 0;
function stepsEl(j, s) {
  const at = stepAt(j);
  const hasSummary = !["none", "placeholder"].includes(s.summary?.kind);  // a failed or missing summary gets no check mark
  return h("ol", { class: "steps", "aria-label": "Processing steps", "data-at": at }, STEPS.map((name, i) =>
    h("li", { class: `step${i === at ? " current" : i < at && (i !== 2 || hasSummary) ? " done" : ""}`, "aria-current": i === at ? "step" : null, text: name })));
}
function fillProcessingCard(card, j, s) {
  card.dataset.job = j ? j.id : "";
  card.dataset.kind = j ? j.kind : "";
  card.hidden = !j;
  if (!j) { card.replaceChildren(); return; }
  const row = h("div", { class: "row" });
  if (j.state === "running") row.append(progressEl(j));
  else {
    row.append(jobChip(j));
    const running = [...lastActive.values()].some((x) => x.state === "running");
    if (lastMode !== "auto" && !running) {
      const start = h("button", { type: "button", class: "primary", text: `Start the queue (${jobCount} job${jobCount === 1 ? "" : "s"})` });
      start.addEventListener("click", async () => {
        try { const r = await api(`/api/sessions/${s.class}/${s.name}/process`, { method: "POST" }); if (r.blocked) toast(`Still waiting: ${waitText(r.blocked)}.`); } catch (e) { toast(e.message); }
      });
      row.append(start, h("span", { class: "muted", text: "Runs the queued jobs in order, this lecture included." }));
    } else if (j.waiting) row.append(h("span", { class: "muted", text: lastMode === "auto" ? "It starts by itself when that changes." : "Press Start the queue when that changes." }));
  }
  card.replaceChildren(stepsEl(j, s), row);
}
let lastMode = null, jobCount = 0;
function processingCard(s) {
  const card = h("section", { class: "card processing", id: "processing", "aria-label": "Processing", "data-key": `${s.class}/${s.name}` });
  fillProcessingCard(card, lastActive.get(`${s.class}/${s.name}`), s);
  card.lecture = s;
  return card;
}

// "All caught up": once per run of activity, when the queue goes from busy to empty. The lectures seen in
// this run are counted; failures come from recent_failed. Fades after ~8 s by CSS (paused while hovered or
// focused; no timer); a failure stays until clicked. Hidden tabs don't poll, so it shows when the tab is back.
let runSeen = new Map(), caughtUp = null;
function trackRun(p) {
  for (const j of p.active) runSeen.set(jobKey(j), j.kind);
  if (p.active.length) { caughtUp = null; return; }
  if (!runSeen.size) return;
  const keys = [...runSeen.keys()], failed = keys.filter((k) => recentFailed.has(k));
  const what = [...runSeen.values()].every((k) => k !== "exam") ? "lecture" : "item", n = keys.length - failed.length;
  caughtUp = { failed: failed.length, href: failed.length ? `#/session/${failed[0]}` : null,
    text: failed.length ? `${n} processed · ${failed.length} failed` : `All caught up · ${n} ${what}${n === 1 ? "" : "s"} processed` };
  runSeen = new Map();
  toast(caughtUp.text);  // .toast is role=status: announced once (not also through #progress-live)
}
function resetPill(pill) {
  pill.classList.remove("caught-up", "failed");
  pill.querySelector(".pbar").hidden = false;
}
function renderPill(p) {
  const pill = $("#queue-pill"), list = p.active || [];
  if (!list.length) {
    if (caughtUp && !pill.classList.contains("caught-up")) {
      pill.classList.remove("waiting");
      pill.classList.add("caught-up");
      pill.classList.toggle("failed", !!caughtUp.failed);
      pill.querySelector(".pbar").hidden = true;
      $("#queue-pill-text").textContent = caughtUp.text;
      pill.href = caughtUp.href || location.hash || "#/";
      pill.title = caughtUp.failed ? "Open the first lecture that failed" : "Everything in the queue has been processed";
      pill.hidden = false;
    } else if (!caughtUp) pill.hidden = true;
    return;
  }
  resetPill(pill);
  pill.hidden = false;
  const run = list.find((j) => j.state === "running"), queued = list.filter((j) => j.state === "queued");
  const bar = pill.querySelector(".pbar");
  bar.hidden = !run;
  pill.classList.toggle("waiting", !run && !!p.waiting);
  let text, href, title;
  if (run) {
    const d = describe(run);
    setFill(pill, bar, d);
    pill.dataset.stage = d.stage;
    text = `${d.pct === null ? d.name : `${d.name} ${d.pct}%`}${queued.length ? ` · ${queued.length} more queued` : ""}`;
    href = run.kind === "exam" ? `#/exam/${run.class}/${run.session}` : `#/session/${run.class}/${run.session}`;
    title = `${run.title || run.session}: ${[d.label, d.meta].filter(Boolean).join(" · ")}`;
  } else {
    const first = queued[0];
    text = p.waiting ? `Waiting · ${waitText(p.waiting)} · ${queued.length} queued` : `${queued.length} queued`;
    href = first.kind === "exam" ? `#/exam/${first.class}/${first.session}` : `#/session/${first.class}/${first.session}`;
    title = "Open the next lecture in the queue";
  }
  $("#queue-pill-text").textContent = text;
  pill.href = href; pill.title = title;
}

{
  const pill = $("#queue-pill");
  const dismiss = () => { caughtUp = null; resetPill(pill); pill.hidden = true; };
  pill.addEventListener("animationend", (e) => { if (e.animationName === "caught-up-out") dismiss(); });
  pill.addEventListener("click", () => { if (caughtUp) dismiss(); });  // the link still opens
}
// Re-render the page, unless the user is typing (then on a later poll).
function rerender() {
  if (editing()) { viewStale = true; return; }
  viewStale = false;
  route();
}
// After a finished bar has faded out (its transitionend; no timers), or at the next poll if it never
// animates (reduced motion, or a hidden tab), whichever comes first.
function whenFaded(el, fn) {
  afterFade = fn;
  el.addEventListener("transitionend", (e) => { if (e.target === el && e.propertyName === "opacity" && afterFade === fn) { afterFade = null; fn(); } });
}
function onProcessing(p) {
  if (!p) return;
  const now = new Map();  // per lecture: its running job, else its first queued one
  for (const j of p.active) if (!now.has(jobKey(j)) || j.state === "running") now.set(jobKey(j), j);
  jobCount = p.active.length;
  recentFailed = new Set(p.recent_failed || []);
  lastMode = p.mode;
  trackRun(p);
  renderPill(p);
  if (!lastPollAt) {  // the first poll: the page renders from it (or, if it rendered before, re-renders now)
    lastActive = now; lastPollAt = Date.now();
    if (now.size && root.childElementCount) rerender();
    return;
  }
  const pending = afterFade;
  afterFade = null;
  const differs = (k) => { const a = lastActive.get(k), b = now.get(k); return !a !== !b || (a && (a.kind !== b.kind || a.state !== b.state || a.id !== b.id || a.position !== b.position || a.waiting !== b.waiting)); };
  const keys = new Set([...lastActive.keys(), ...now.keys()]);
  let finished = null;
  for (const el of root.querySelectorAll("[data-progress]")) {
    const j = now.get(el.dataset.progress);
    if (j && j.state === "running" && j.id === el.dataset.job) updateProgress(el, j);
    else if (!j && lastActive.has(el.dataset.progress)) {
      const failed = recentFailed.has(el.dataset.progress);
      finishProgress(el, failed);
      $("#progress-live").textContent = `${lastActive.get(el.dataset.progress).title || el.dataset.progress}: ${failed ? "processing failed" : "done"}`;
      finished = el;
    }
  }
  const card = $("#processing");
  let next = null;  // what this poll changes on the page
  if (shown.kind === "class") {
    if ([...keys].some((k) => k.startsWith(`${shown.slug}/`) && differs(k)) || viewStale) next = rerender;
  } else if (shown.kind === "session" && card) {
    const k = card.dataset.key, a = lastActive.get(k);
    // The transcript or the summary is new (its job ended): re-render the page. Otherwise only the card changes.
    if ((a && ["transcribe", "summarize"].includes(a.kind) && now.get(k)?.kind !== a.kind) || viewStale) next = rerender;
    else if (differs(k) || finished) next = () => { if (card.isConnected) fillProcessingCard(card, lastActive.get(k), card.lecture); };
    else if (now.get(k) && card.querySelector(".steps")?.dataset.at !== String(stepAt(now.get(k)))) {  // Transcribe -> Speakers
      card.querySelector(".steps").replaceWith(stepsEl(now.get(k), card.lecture));
    }
  }
  lastActive = now; lastPollAt = Date.now();
  if (next && finished) whenFaded(finished, next);  // "Done" first, then the page changes
  else if (next) next();
  else if (pending) pending();
}

// ------------------------------------------------------------------ views
async function homeView() {
  const view = stage();
  const [classes, devices, trash, caps] = await Promise.all([api("/api/classes"), api("/api/devices").catch(() => []),
    api("/api/trash").catch(() => ({ items: [] })), capabilities]);
  const named = classes.filter((c) => c.has_class_file && c.slug !== "unsorted");  // "Not sure yet" stands for Unsorted
  const classOptions = () => [h("option", { value: "", text: "Choose a class…" }), ...named.map((c) => h("option", { value: c.slug, text: c.name }))];

  const recClass = h("select", { id: "record-class", "aria-label": "Class" }, classOptions());
  const mic = h("select", { id: "record-device", "aria-label": "Microphone" },
    h("option", { value: "", text: "System default microphone" }),
    devices.map((d) => h("option", { value: String(d.index), text: d.name })));
  const start = h("button", { class: "primary", id: "record-start", disabled: true, type: "button", text: "Start recording" });
  // Source: a microphone, or one app's audio (e.g. Zoom). Apps are listed only when asked for.
  const srcMic = h("input", { type: "radio", name: "record-source", id: "src-mic", value: "mic", checked: true });
  const srcApp = h("input", { type: "radio", name: "record-source", id: "src-app", value: "app" });
  const appSel = h("select", { id: "record-app", "aria-label": "App" });
  const appRefresh = h("button", { type: "button", class: "ghost", text: "Refresh list" });
  const withMic = h("input", { type: "checkbox", id: "record-with-mic" });
  const micLabel = h("label", { for: "record-device", text: "Microphone" });
  const appBox = h("div", { class: "stack", id: "app-box", hidden: true },
    h("label", { for: "record-app", text: "App" }), h("div", { class: "row" }, appSel, appRefresh),
    h("label", { class: "check" }, withMic, " Also include my microphone (off: only the app's audio, so your own questions aren't recorded)"),
    h("p", { class: "muted", id: "app-volume-note", text: "Keep the app's volume up and unmuted in Windows' Volume Mixer and in the app: "
      + "lowering or muting them lowers the recording. The system volume doesn't matter. Recording stops when the app closes." }));
  const isApp = () => srcApp.checked;
  const syncStart = () => (start.disabled = !recClass.value || (isApp() && !appSel.value));
  async function loadApps() {
    appSel.replaceChildren(h("option", { value: "", text: "Loading…" })); syncStart();
    let r;
    try { r = await api("/api/apps"); } catch (e) { r = { supported: false, reason: e.message, apps: [] }; }
    if (!r.supported) { appSel.replaceChildren(h("option", { value: "", text: r.reason })); syncStart(); return; }
    appSel.replaceChildren(...(r.apps.length ? r.apps.map((a) => h("option", { value: String(a.pid),
      text: `${a.name}${a.active ? "" : " (not playing now)"}` })) : [h("option", { value: "", text: "No app has audio open. Start the meeting, then refresh." })]));
    syncStart();
  }
  const syncSource = () => {
    appBox.hidden = !isApp();
    mic.hidden = micLabel.hidden = isApp() && !withMic.checked;
    if (isApp() && !appSel.options.length) loadApps();
    syncStart();
  };
  for (const el of [srcMic, srcApp, withMic]) el.addEventListener("change", syncSource);
  appRefresh.addEventListener("click", loadApps);
  appSel.addEventListener("change", syncStart);
  recClass.addEventListener("change", syncStart);
  start.addEventListener("click", async () => {
    start.disabled = true;
    const json = { class: recClass.value, device: mic.value || null };
    if (isApp()) Object.assign(json, { app_pid: Number(appSel.value), include_mic: withMic.checked, device: withMic.checked ? json.device : null });
    try { await api("/api/record/start", { method: "POST", json }); toast("Recording started."); }
    catch (e) { toast(e.message); syncStart(); if (isApp()) loadApps(); }
    refreshRecording();
  });

  const impClass = h("select", { id: "import-class", "aria-label": "Class" }, classOptions(), h("option", { value: "unsorted", text: "Not sure yet" }));
  const impFile = h("input", { type: "file", id: "import-file", multiple: true, accept: ".wav,.mp3,.m4a,.opus,.ogg,.flac,.mp4,audio/*,video/mp4" });
  const impBtn = h("button", { type: "button", text: "Import", disabled: true });
  const syncImp = () => (impBtn.disabled = !(impClass.value && impFile.files.length));
  impClass.addEventListener("change", syncImp); impFile.addEventListener("change", syncImp);
  const importWarnings = h("div", { id: "import-warnings" });
  // Pre-select the class the file seems to belong to. Only a pre-selection: the user still presses Import,
  // and a class they chose themselves is never changed.
  const impHint = h("p", { class: "muted", id: "import-suggestion", "aria-live": "polite", hidden: true });
  let autoPicked = false, impAsked = 0;
  impClass.addEventListener("change", () => { autoPicked = false; impHint.hidden = true; });
  async function suggestClass(json) {
    const asked = ++impAsked;
    const r = await api("/api/import/suggest", { method: "POST", json }).catch(() => null);
    if (asked !== impAsked || (impClass.value && !autoPicked)) return;  // newer files picked, or the user chose
    const s = r?.suggestion;
    autoPicked = !!s;
    impClass.value = s ? s.class : "";
    impHint.hidden = !s;
    impHint.textContent = s ? `Suggested: ${s.class_name} · ${s.reason}` : "";
    syncImp(); syncZoom();
  }
  impFile.addEventListener("change", () => { if (impFile.files.length) suggestClass({ files: [...impFile.files].map((f) => ({ name: f.name })) }); });
  const uploadFile = (f, slug, anyway) => {
    const fd = new FormData();
    fd.append("file", f); fd.append("class_slug", slug); fd.append("last_modified_ms", String(f.lastModified));
    if (anyway) fd.append("anyway", "true");
    return api("/api/import", { method: "POST", body: fd });
  };
  impBtn.addEventListener("click", async () => {
    const files = [...impFile.files]; const done = []; const failed = []; const repeats = [];
    const slug = impClass.value;
    impBtn.disabled = true;
    importWarnings.replaceChildren();
    for (const [i, f] of files.entries()) { // one at a time: each upload is written and fsynced before the next
      impBtn.textContent = files.length > 1 ? `Uploading ${i + 1} of ${files.length}…` : "Uploading…";
      try { done.push(await uploadFile(f, slug)); }
      catch (e) {
        if (e.body?.already_imported) repeats.push([f, e.body.already_imported]);
        else failed.push(`${f.name}: ${e.message}`);
      }
    }
    impBtn.textContent = "Import"; syncImp();
    importWarnings.replaceChildren(...repeats.map(([f, seen]) => alreadyImported(f.name, seen, async () => { const r = await uploadFile(f, slug, true); toast("Imported."); return r; })));
    if (failed.length) toast(`Imported ${done.length} of ${files.length}. Not imported: ${failed.join("; ")}`);
    else if (done.length) toast(done.length === 1 ? "Imported." : `Imported ${done.length} files.`);
    if (repeats.length) return;  // stay here: the warnings need an answer
    if (done.length === 1 && !failed.length) location.hash = `#/session/${done[0].class}/${done[0].session}`;
    else if (done.length) location.hash = `#/class/${done[0].class}`;
  });

  const zoomDir = h("input", { type: "file", id: "import-zoom", webkitdirectory: true });
  const zoomBtn = h("button", { type: "button", text: "Import Zoom folder", disabled: true });
  const syncZoom = () => (zoomBtn.disabled = !(impClass.value && zoomDir.files.length));
  impClass.addEventListener("change", syncZoom); zoomDir.addEventListener("change", syncZoom);
  zoomDir.addEventListener("change", async () => {
    const list = [...zoomDir.files].map((f) => ({ name: f.webkitRelativePath || f.name, size: f.size }));
    if (!list.length) return;
    const plan = await api("/api/import/zoom-plan", { method: "POST", json: { files: list } }).catch(() => null);  // names and sizes only
    if (plan && !plan.error) suggestClass({ topic: plan.title, when: plan.when });
  });
  zoomBtn.addEventListener("click", async () => {
    const files = [...zoomDir.files];
    const byName = new Map(files.map((f) => [f.webkitRelativePath || f.name, f]));
    const slug = impClass.value;
    zoomBtn.disabled = true;
    importWarnings.replaceChildren();
    let upload, announce;
    try {
      // The server picks the files from names and sizes only; then only those are uploaded (not the video if an m4a exists).
      const plan = await api("/api/import/zoom-plan", { method: "POST", json: { files: files.map((f) => ({ name: f.webkitRelativePath || f.name, size: f.size })) } });
      if (plan.error) { toast(plan.error); return; }
      const audio = byName.get(plan.audio);
      upload = (anyway) => {
        const fd = new FormData();
        fd.append("class_slug", slug); fd.append("folder", plan.folder || "");
        fd.append("audio", audio); fd.append("last_modified_ms", String(audio.lastModified));
        if (plan.transcript) fd.append("transcript", byName.get(plan.transcript));
        if (plan.chat) fd.append("chat", byName.get(plan.chat));
        if (anyway) fd.append("anyway", "true");
        return api("/api/import/zoom", { method: "POST", body: fd });
      };
      announce = () => toast(`Imported ${audio.name}${plan.transcript ? " with Zoom's transcript" : ""}.`
        + (plan.participants ? ` ${plan.participants} per-participant file(s) were not imported.` : "")
        + (plan.others ? ` The folder has ${plan.others} more recording file(s) (e.g. the recording was restarted): only the largest was imported; import the others as files.` : ""));
      zoomBtn.textContent = "Uploading…";
      const r = await upload();
      announce();
      location.hash = `#/session/${r.class}/${r.session}`;
    } catch (e) {
      if (e.body?.already_imported) importWarnings.replaceChildren(alreadyImported(zoomDir.files[0]?.webkitRelativePath.split("/")[0] || "This recording", e.body.already_imported, async () => { const r = await upload(true); announce(); return r; }));
      else toast(e.message);
    }
    finally { zoomBtn.textContent = "Import Zoom folder"; syncZoom(); }
  });

  const newName = h("input", { id: "class-name", placeholder: "e.g. Physics 101", maxlength: "100" });
  const newMax = h("input", { id: "class-max", type: "number", min: "1", max: "600", placeholder: "120" });
  const create = h("button", { type: "button", text: "Add class" });
  create.addEventListener("click", async () => {
    try { await api("/api/classes", { method: "POST", json: { name: newName.value, max_length_minutes: newMax.value ? Number(newMax.value) : null } }); toast("Class added."); route(); }
    catch (e) { toast(e.message); }
  });

  view.replaceChildren(
    h("h1", { text: "Library" }),
    h("p", { class: "muted", text: "Recordings, transcripts and summaries are plain files in your data folder." }),
    h("div", { class: "grid" },
      h("section", { class: "card stack", "aria-labelledby": "rec-h" },
        h("h2", { id: "rec-h", text: "Record a lecture" }),
        h("label", { for: "record-class", text: "Class (required)" }), recClass,
        caps.app_capture_hidden ? h("p", { class: "muted", id: "no-app-capture", text: caps.app_capture_reason || "Recording from one app isn't available on this computer." })
          : h("fieldset", { class: "row" }, h("legend", { text: "Record from" }),
          h("label", { class: "check" }, srcMic, " A microphone"), h("label", { class: "check" }, srcApp, " An app (e.g. Zoom)")),
        appBox,
        micLabel, mic,
        h("div", { class: "row" }, start),
        h("p", { class: "reminder", text: "You are responsible for getting permission before recording anyone." })),
      h("section", { class: "card stack", "aria-labelledby": "imp-h" },
        h("h2", { id: "imp-h", text: "Import a recording" }),
        h("label", { for: "import-class", text: "Class" }), impClass, impHint,
        h("label", { for: "import-file", text: "Audio or video files (wav, mp3, m4a, opus, ogg, flac, mp4: only the audio is kept; several at once is fine)" }), impFile,
        h("div", { class: "row" }, impBtn),
        h("label", { for: "import-zoom", text: "Or a Zoom recording folder (Documents\\Zoom\\…, or a downloaded cloud recording). "
          + "The audio-only file is used, and Zoom's transcript and chat are kept. Files are only copied into your data folder, never uploaded anywhere." }), zoomDir,
        h("div", { class: "row" }, zoomBtn), importWarnings),
      h("section", { class: "card", "aria-labelledby": "cls-h" },
        h("h2", { id: "cls-h", text: "Classes" }),
        classes.length ? h("ul", { class: "list", id: "class-list" }, classes.map((c) =>
          h("li", {}, h("a", { class: "title", href: `#/class/${c.slug}`, text: c.name }),
            c.has_class_file ? null : h("span", { class: "chip failed", text: "No class settings", title: "A class folder without class.json (e.g. from an older terminal recording)" }), h("span", { class: "spacer" }),
            h("span", { class: "muted", text: `${c.sessions} session${c.sessions === 1 ? "" : "s"}` }))))
          : h("p", { class: "muted", text: "No classes yet. Add one below." }),
        h("p", {}, h("a", { href: "#/trash", id: "trash-link", text: `Recently deleted (${trash.items.length})` })),
        h("label", { for: "class-name", text: "New class name" }), newName,
        h("label", { for: "class-max", text: "Auto-stop after (minutes, optional)" }), newMax,
        h("div", { class: "row" }, create)),
      await settingsCard()));
}

// ------------------------------------------------------------------ processing and storage settings
async function settingsCard() {
  const box = h("section", { class: "card stack", id: "settings", "aria-labelledby": "set-h" });
  let p;
  try { p = await api("/api/processing"); } catch (e) { box.append(h("p", { text: e.message })); return box; }
  const save = async (body) => { try { await api("/api/settings", { method: "PUT", json: body }); toast("Saved."); } catch (e) { toast(e.message); } route(); };

  const mode = h("select", { id: "processing-mode", "aria-label": "Processing" },
    h("option", { value: "manual", text: "Manual (I press Process)", selected: p.mode === "manual" }),
    h("option", { value: "auto", text: "Automatic when plugged in", selected: p.mode === "auto" }));
  mode.addEventListener("change", () => save({ processing_mode: mode.value }));
  const why = p.waiting || p.blocked_now;
  const status = p.queued ? `${p.queued} job(s) in the queue${why ? `, waiting: ${why}` : p.running ? ", running" : ""}.`
    : `${p.unprocessed} recording(s) not processed.`;
  const all = h("button", { type: "button", class: "primary", id: "process-all",
    text: p.queued && !p.unprocessed ? "Start the queue" : `Process all pending (${p.unprocessed})`, disabled: !p.unprocessed && !p.queued });
  all.addEventListener("click", async () => {
    all.disabled = true;
    try { const r = await api("/api/process-pending", { method: "POST" }); toast(r.blocked ? `Queued ${r.queued}. Waiting: ${r.blocked}.` : `Queued ${r.queued}.`); } catch (e) { toast(e.message); }
    route();
  });

  const idleQuit = h("select", { id: "idle-quit", "aria-label": "Quit when no page is open" },
    [[0, "Never (default)"], [10, "After 10 minutes"], [30, "After 30 minutes"], [60, "After 1 hour"]].map(([v, t]) =>
      h("option", { value: String(v), text: t, selected: p.idle_quit_minutes === v })),
    [0, 10, 30, 60].includes(p.idle_quit_minutes) ? null : h("option", { value: String(p.idle_quit_minutes), text: `After ${p.idle_quit_minutes} minutes`, selected: true }));
  idleQuit.addEventListener("change", () => save({ idle_quit_minutes: Number(idleQuit.value) }));
  const awake = h("input", { type: "checkbox", id: "keep-awake", checked: p.keep_awake });
  awake.addEventListener("change", () => save({ keep_awake: awake.checked }));

  const del = h("input", { type: "checkbox", id: "delete-original", checked: p.delete_original_after_opus });
  del.addEventListener("change", () => {
    if (del.checked && !confirm("Delete the original recording (WAV, m4a, mp3, …) after its compressed Opus copy has been verified?\n\nDeletion is permanent: the original cannot be recovered. The Opus copy is kept and plays in the app. Your own files outside the app (e.g. a Zoom mp4) are never touched.")) { del.checked = false; return; }
    save({ delete_original_after_opus: del.checked, confirm_permanent_deletion: del.checked });
  });
  const purge = h("select", { id: "trash-days", "aria-label": "Empty Recently deleted" },
    [[0, "Never (default)"], [7, "After 7 days"], [30, "After 30 days"], [90, "After 90 days"]].map(([v, t]) =>
      h("option", { value: String(v), text: t, selected: p.trash_days === v })),
    [0, 7, 30, 90].includes(p.trash_days) ? null : h("option", { value: String(p.trash_days), text: `After ${p.trash_days} days`, selected: true }));
  purge.addEventListener("change", () => {
    const days = Number(purge.value);
    if (days && !confirm(`Permanently delete items that have been in Recently deleted for more than ${days} days?\n\nThis cannot be undone. It runs each time the app starts.`)) { purge.value = String(p.trash_days); return; }
    save({ trash_days: days, confirm_permanent_deletion: days > 0 });
  });
  const tpl = h("input", { type: "text", id: "naming-template", value: p.naming_template || "", maxlength: "200",
    placeholder: "e.g. {class} {date} Lecture {n}", "aria-describedby": "naming-help naming-preview" });
  const tplSave = h("button", { type: "button", id: "naming-save", text: "Save name template", disabled: true });
  const tplOut = h("div", { id: "naming-preview", class: "stack", "aria-live": "polite" });
  // At most one preview request in flight; typing meanwhile asks once more when it returns (no timers).
  let tplBusy = false, tplAgain = false, tplValid = true;
  const preview = async () => {
    if (tplBusy) { tplAgain = true; return; }
    tplBusy = true;
    const value = tpl.value;
    let r;
    try { r = await api(`/api/naming/preview?template=${encodeURIComponent(value)}`); } catch (e) { r = { valid: false, error: e.message, examples: [] }; }
    tplBusy = false;
    if (tplAgain || value !== tpl.value) { tplAgain = false; preview(); return; }  // the text changed meanwhile
    tplValid = r.valid;
    tplSave.disabled = !r.valid || value.trim() === (p.naming_template || "");
    tplOut.replaceChildren(...(!r.valid ? [h("p", { class: "warning", role: "alert", text: r.error })]
      : !value.trim() ? [h("p", { class: "muted", text: "No template: recordings are named by date and time, imports by their file name." })]
      : r.examples.map((x) => h("p", {}, h("span", { class: "muted", text: `${x.what}: ` }), h("strong", { text: x.title })))));
  };
  tpl.addEventListener("input", () => { tplSave.disabled = true; preview(); });
  tplSave.addEventListener("click", () => { if (tplValid) save({ naming_template: tpl.value }); });
  preview();
  const cleanupOut = h("div", { class: "stack", id: "cleanup" });
  const cleanup = h("button", { type: "button", id: "compact", text: "Delete originals that have an Opus copy…", hidden: !p.delete_original_after_opus });
  cleanup.addEventListener("click", async () => {
    let plan;
    try { plan = await api("/api/compact"); } catch (e) { toast(e.message); return; }
    const mb = (b) => `${(b / 1048576).toFixed(1)} MB`;
    const n = plan.items.length, freed = plan.bytes_before - plan.bytes_after;
    const go = h("button", { type: "button", class: "danger", text: `Delete ${n} original(s), free ${mb(freed)}`, disabled: !n });
    go.addEventListener("click", async () => {
      if (!confirm(`Permanently delete ${n} original recording(s)? Each Opus copy is checked again first.`)) return;
      go.disabled = true;
      try {
        const r = await api("/api/compact", { method: "POST", json: { token: plan.token, confirm: true } });
        toast(`Deleted ${r.done.length} original(s), freed ${mb(r.bytes_before - r.bytes_after)}${r.kept.length ? `; kept ${r.kept.length}` : ""}.`);
      } catch (e) { toast(e.message); }
      route();
    });
    cleanupOut.replaceChildren(...[
      h("p", { class: "muted", text: `Dry run (nothing deleted yet): ${n} original(s) can go, freeing ${mb(freed)}; ${plan.skipped.length} kept.` }),
      n ? h("ul", { class: "list" }, plan.items.map((x) => h("li", { text: `${x.session} · ${x.file} · ${mb(x.bytes_before)} → ${mb(x.bytes_after)}` }))) : null,
      plan.skipped.length ? h("ul", { class: "list" }, plan.skipped.map((x) => h("li", { class: "muted", text: `${x.session}: kept (${x.reason})` }))) : null,
      h("div", { class: "row" }, go)].filter(Boolean));
  });
  const m = p.machine || { picks: {}, overrides: {} };
  const describe = (t, own) => {
    const v = (k) => own[`transcription.${k}`] ?? t[k];
    return t.model ? `Speech-to-text: ${v("model")} on ${v("device") === "cuda" ? "the GPU" : "the CPU"} (${v("compute_type")}, `
      + `${Number(v("batch_size")) > 0 ? "batched" : "sequential"} decoding, ${v("cpu_threads")} CPU threads)`
      + (Object.keys(own).length ? ". Your config.toml overrides: " + Object.entries(own).map(([k, x]) => `${k} = ${x}`).join(", ") : "")
      : "Not configured for this computer yet.";
  };
  const machineText = h("p", { id: "machine-picks", text: describe(m.picks, m.overrides) });
  const machineNotes = h("ul", { id: "machine-notes", class: "muted" });
  const redetect = h("button", { type: "button", text: "Detect this computer again" });
  redetect.addEventListener("click", async () => {
    redetect.disabled = true;
    try {
      const r = await api("/api/configure", { method: "POST" });
      machineText.textContent = describe(r.picks, r.overrides);
      machineNotes.replaceChildren(...r.notes.map((n) => h("li", { text: n })), ...r.missing.map((x) => h("li", { text: `Missing: ${x}` })));
      toast("Settings for this computer updated. The next processing job uses them.");
    } catch (e) { toast(e.message); }
    redetect.disabled = false;
  });
  box.append(
    h("h2", { id: "set-h", text: "Processing and storage" }),
    machineText, machineNotes, h("div", { class: "row" }, redetect),
    h("label", { for: "processing-mode", text: "Processing" }), mode,
    h("p", { class: "muted", text: "Processing never starts while recording or on battery; queued work waits." }),
    h("p", { id: "processing-status", text: status }),
    h("div", { class: "row" }, all),
    h("label", { for: "naming-template", text: "Name new lectures" }), tpl,
    h("p", { class: "muted", id: "naming-help", text: "Placeholders: {class} the class, {date} and {time} when it was recorded (for imports, read from the file), "
      + "{n} its number in the class by date, {original} the imported file's name. Applies to new recordings and imports only; existing names never change. Empty: date and time." }),
    tplOut, h("div", { class: "row" }, tplSave),
    h("label", { class: "check" }, awake, " Keep the laptop awake while processing"),
    h("p", { class: "muted", text: "Prevents idle sleep only: closing the lid or choosing Sleep still follows Windows' own settings." }),
    h("label", { class: "check" }, del, " Delete the original after a verified Opus copy (permanent)"),
    h("p", { class: "muted", text: "Every recording and import gets a compact Opus copy (about 20 MB per hour) once it is transcribed. Older lectures without one: run lecture-recorder compact." }),
    h("div", { class: "row" }, cleanup), cleanupOut,
    h("label", { for: "trash-days", text: "Empty Recently deleted" }), purge,
    h("label", { for: "idle-quit", text: "Quit when no page is open" }), idleQuit,
    h("p", { class: "muted", text: "Never while recording, processing or with jobs queued. A page in a hidden or minimized tab doesn't check in, so it counts as closed." }));
  return box;
}

// Rename a lecture in place: the title element becomes a text box (Enter saves, Escape cancels).
// Only the title changes (meta.json); the folder and the original file name never do.
function renameButton(slug, s, titleEl, after) {
  const b = h("button", { type: "button", class: "ghost", text: "Rename", "aria-label": `Rename ${s.title}`,
    disabled: s.state === "recording" || s.state === "unfinished", title: "Change the lecture's name (its files stay as they are)" });
  b.addEventListener("click", () => {
    const input = h("input", { type: "text", class: "rename", value: s.title, maxlength: "200", "aria-label": "Lecture name" });
    const save = h("button", { type: "button", class: "primary", text: "Save" });
    const cancel = h("button", { type: "button", text: "Cancel" });
    const form = h("span", { class: "row rename-form" }, input, save, cancel);
    const close = () => { form.replaceWith(titleEl); b.hidden = false; };
    const submit = async () => {
      save.disabled = true;
      try {
        const r = tookVersion(await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(s.name)}/rename`, { method: "POST", json: { title: input.value } }));
        s.title = r.title; titleEl.textContent = r.title; b.setAttribute("aria-label", `Rename ${r.title}`); close(); toast("Renamed."); after?.();
      } catch (e) { toast(e.message); save.disabled = false; input.focus(); }
    };
    save.addEventListener("click", submit);
    cancel.addEventListener("click", close);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); else if (e.key === "Escape") close(); });
    titleEl.replaceWith(form); b.hidden = true;
    input.focus(); input.select();
  });
  return b;
}

async function deleteLecture(slug, s, after) {
  if (!confirm(`Move "${s.title}" to Recently deleted?\n\nYou can restore it from there. Its queued processing is cancelled.`)) return;
  try { tookVersion(await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(s.name)}/delete`, { method: "POST" })); toast("Moved to Recently deleted."); after(); }
  catch (e) { toast(e.message); }
}

const sessionFilter = new Map();  // class slug -> filter text on its lecture list

// A class row's status: the progress bar of the lecture's running job (transcribe, then summarize, then
// embed), a chip for a queued one, or the lecture's state (failed: with Try again).
function lectureStatus(slug, s) {
  const j = lastActive.get(`${slug}/${s.name}`);
  if (j) return j.state === "running" ? progressEl(j) : jobChip(j);
  if (s.state !== "failed") return chip(s.state);
  const again = h("button", { type: "button", class: "ghost", text: "Try again", "aria-label": `Try processing ${s.title} again` });
  again.addEventListener("click", async () => {
    again.disabled = true;
    try { const r = await api(`/api/sessions/${slug}/${s.name}/process`, { method: "POST" }); if (r.blocked) toast(`Queued. Waiting: ${waitText(r.blocked)}.`); route(); } catch (e) { toast(e.message); again.disabled = false; }
  });
  return h("span", { class: "row" }, chip(s.state), again);
}

async function classView(slug) {
  const view = stage();
  const c = await api(`/api/classes/${encodeURIComponent(slug)}`);
  const n = c.sessions.length;
  const del = h("button", { type: "button", class: "danger", id: "delete-class", text: "Delete class…" });
  del.addEventListener("click", async () => {
    const msg = n ? `Delete the class "${c.name}" and its ${n} lecture${n === 1 ? "" : "s"}?\n\nThe whole class goes to Recently deleted, where you can restore it.`
      : `Delete the empty class "${c.name}"?`;
    if (!confirm(msg)) return;
    try { tookVersion(await api(`/api/classes/${encodeURIComponent(slug)}/delete`, { method: "POST", json: { sessions: n } })); toast("Class moved to Recently deleted."); location.hash = "#/"; }
    catch (e) { toast(e.message); route(); }
  });
  let folderOnly = "";
  if (!c.has_class_file) {
    const reg = h("button", { type: "button", text: "Make it a class" });
    reg.addEventListener("click", async () => { try { await api(`/api/classes/${encodeURIComponent(slug)}/register`, { method: "POST" }); toast("Class settings created."); route(); } catch (e) { toast(e.message); } });
    folderOnly = h("div", { class: "warning", id: "folder-only" },
      h("strong", { text: "Folder only: " }), "this class folder has no class settings (class.json), for example from an older terminal recording. Its recordings are shown here and kept.",
      h("div", { class: "row" }, reg, h("span", { class: "muted", text: "Creates the settings file, named after the folder, so you can record into it and use units." })));
  }
  const list = h("ul", { class: "list", id: "session-list" }, c.sessions.map((s) => {
    const d = h("button", { type: "button", class: "ghost", text: "Delete", "aria-label": `Delete ${s.title}`,
      disabled: s.state === "recording" || s.state === "running", title: "Move to Recently deleted" });
    d.addEventListener("click", () => deleteLecture(slug, s, route));
    const name = h("a", { class: "title", href: `#/session/${slug}/${s.name}`, text: s.title });
    const li = h("li", { "data-session": s.name }, name,
      h("span", { class: "muted", text: `${fmtDate(s.created)}${s.duration_s ? " · " + hms(s.duration_s) : ""}${s.source === "import" ? " · imported" : ""}` }),
      h("span", { class: "spacer" }), s.duplicate ? h("span", { class: "chip duplicate", text: "Possible duplicate", title: "Open the lecture to compare the two" }) : null,
      s.suggested ? h("span", { class: "chip suggested", text: `Suggested: ${s.suggested.class_name}`, title: `${s.suggested.reason}. Open the lecture to move it.` }) : null,
      lectureStatus(slug, s), renameButton(slug, s, name, route), d);  // route: units and exams show the new name
    li.match = (q) => [s.title, s.original || ""].some((x) => x.toLocaleLowerCase().includes(q));
    return li;
  }));
  // Filter by name or original file name. Kept across re-renders (a rename or delete elsewhere), and
  // not counted as unsaved input: it changes nothing on disk.
  const filter = h("input", { type: "search", id: "session-filter", "data-transient": "", value: sessionFilter.get(slug) || "",
    placeholder: "Filter by name or original file", "aria-label": "Filter lectures by name or original file" });
  const applyFilter = () => {
    const q = filter.value.trim().toLocaleLowerCase();
    if (q) sessionFilter.set(slug, filter.value); else sessionFilter.delete(slug);
    for (const li of list.children) li.hidden = !!q && !li.match(q);
  };
  filter.addEventListener("input", applyFilter);
  applyFilter();
  view.replaceChildren(
    h("p", {}, h("a", { href: "#/", text: "← Library" })),
    h("h1", { text: c.name }),
    h("p", { class: "muted", text: c.max_length_minutes ? `Auto-stop after ${c.max_length_minutes} min` : "Auto-stop after 120 min (default)" }),
    folderOnly,
    h("section", { class: "card" },
      h("h2", { text: "Sessions" }),
      n > 1 ? filter : null,
      n ? list : h("p", { class: "muted", text: "No sessions yet." })),
    ...(c.has_class_file ? await unitAndExamCards(slug) : []),
    h("div", { class: "row" }, del));
}

async function trashView() {
  const view = stage();
  const t = await api("/api/trash");
  const restoreBtn = (x) => {
    const b = h("button", { type: "button", class: "primary", text: "Restore", disabled: x.occupied === true,
      title: x.occupied ? "Something is already at its original place" : "Put it back where it was" });
    b.addEventListener("click", async () => {
      b.disabled = true;
      try { const r = tookVersion(await api(`/api/trash/${x.id}/restore`, { method: "POST" })); toast(r.requeued ? `Restored; ${r.requeued} job(s) queued again.` : "Restored."); route(); }
      catch (e) { toast(e.message); b.disabled = false; }
    });
    return b;
  };
  const purgeBtn = (x) => {
    const b = h("button", { type: "button", class: "danger", text: "Delete permanently…" });
    b.addEventListener("click", async () => {
      const typed = prompt(`Permanently delete "${x.title}"${x.kind === "class" ? ` and its ${x.sessions} lecture(s)` : ""}? This cannot be undone.\n\nType the name to confirm: ${x.title}`, "");
      if (typed === null) return;
      try { tookVersion(await api(`/api/trash/${x.id}/delete`, { method: "POST", json: { confirm: typed } })); toast("Deleted permanently."); route(); }
      catch (e) { toast(e.message); }
    });
    return b;
  };
  view.replaceChildren(
    h("p", {}, h("a", { href: "#/", text: "← Library" })),
    h("h1", { text: "Recently deleted" }),
    h("p", { class: "muted", text: t.trash_days ? `Items older than ${t.trash_days} days are deleted permanently when the app starts (Settings).`
      : "Nothing here is deleted until you choose Delete permanently (or turn on emptying in Settings)." }),
    h("section", { class: "card" }, t.items.length ? h("ul", { class: "list", id: "trash-list" }, t.items.map((x) =>
      h("li", { "data-id": x.id },
        h("div", { class: "stack grow" }, h("div", {}, h("strong", { text: x.title })),
          h("div", { class: "muted", text: (x.kind === "class" ? `Class · ${x.sessions} lecture${x.sessions === 1 ? "" : "s"}` : `Lecture · ${x.class_name}${x.created ? " · " + fmtDate(x.created) : ""}`)
            + ` · deleted ${fmtDate(x.deleted)}` }),
          x.occupied ? h("div", { class: "warning", text: "Something is already at its original place, so it can't be restored until that is moved." }) : ""),
        restoreBtn(x), purgeBtn(x))))
      : h("p", { class: "muted", text: "Nothing here." })));
}

const fetchUnitsExams = (slug) => Promise.all([api(`/api/classes/${encodeURIComponent(slug)}/units`), api(`/api/classes/${encodeURIComponent(slug)}/exams`)]);
async function unitAndExamCards(slug) {
  const [u, exams] = await fetchUnitsExams(slug);
  return [unitsCard(slug, u), examsCard(slug, u, exams)];
}

// ------------------------------------------------------------------ units (time spans) and exams
const spanText = (sp, titles) => `${titles[sp.session] || sp.session} ${hms(sp.start)}–${sp.end === null ? "end" : hms(sp.end)}`;
const sessionUrl = (slug, session, t) => `#/session/${encodeURIComponent(slug)}/${encodeURIComponent(session)}${t ? `?t=${t}` : ""}`;
const clock = (s) => { const t = hms(s).replace(/^00:/, ""); return t.length === 5 && t[0] === "0" ? t.slice(1) : t; };  // 0:00, 32:10, 01:05:00

// One vertical timeline: the class's lectures in date order under unit headers. Between two lectures of the
// same unit is a "+" gap button that starts a new unit at the next lecture (t = 0); a header's "×" removes
// its boundary. Both show an Undo toast. Suggestions are shown in place (a lit gap, or a chip in the row).
function unitsCard(slug, u) {
  const box = h("section", { class: "card stack", id: "units", "aria-labelledby": "units-h" });
  let sugg = null, busy = false;  // the last "Suggest boundaries" answer, until cleared
  const enc = encodeURIComponent(slug), base = `/api/classes/${enc}/units`;
  const timeline = h("ol", { class: "list timeline", id: "unit-timeline", "aria-label": "Units and lectures" });
  const note = h("div", { class: "muted", id: "unit-suggest-note", "aria-live": "polite" });
  const after = h("div", { class: "stack" });

  // The exam form lists the units: rebuild it, keeping what the user typed or ticked (ticks of removed units drop).
  function syncExams(exams) {
    const old = $("#exams");
    if (!old) return;
    const card = examsCard(slug, u, exams);
    for (const el of old.querySelectorAll("input:not([type=checkbox])")) {
      const twin = card.querySelector(el.id ? `#${el.id}` : `input[aria-label="${el.getAttribute("aria-label")}"]`);
      if (twin) twin.value = el.value;
    }
    for (const el of old.querySelectorAll("input[type=checkbox]:checked")) {
      const twin = card.querySelector(`input[type=checkbox][value="${CSS.escape(el.value)}"]`);
      if (twin) twin.checked = true;
    }
    old.replaceWith(card);
  }
  // focus: { id, select } = a unit's name field; { gap } = the "+" before that lecture
  async function reload(focus) {
    let exams;
    try { [u, exams] = await fetchUnitsExams(slug); } catch (e) { toast(e.message); return; }
    if (!box.isConnected) return;
    draw();
    syncExams(exams);
    const el = focus && (focus.id ? timeline.querySelector(`[data-unit="${CSS.escape(focus.id)}"] input`)
      : timeline.querySelector(`[data-session="${CSS.escape(focus.gap)}"] button`));
    if (el) { el.focus(); if (focus.select) el.select(); }
  }
  // Every write goes through here: one at a time, then the card re-renders from the server's answer.
  // Returns false if another write is still running (nothing started).
  function act(fn) {
    if (busy) return false;
    busy = true;
    (async () => { try { await fn(); } catch (e) { toast(e.message); } finally { busy = false; } })();
    return true;
  }
  const insert = (session, t) => act(async () => {
    const nu = await api(base, { method: "POST", json: { session, t } });
    if (sugg) sugg = Object.assign(sugg.filter((x) => !(x.session === session && Math.abs((x.at_lecture_start ? 0 : x.t) - t) < 1)), { notIndexed: sugg.notIndexed });
    await reload({ id: nu.id, select: true });
    toast("New unit started.", { label: "Undo", run: () => act(async () => {
      await api(`${base}/${encodeURIComponent(nu.id)}`, { method: "DELETE" }); toast("New unit removed."); await reload(t ? null : { gap: session });
    }) });
  });
  const remove = (unit, prevName) => act(async () => {
    const at = unit.spans[0];  // where it starts now (for a moved unit: the lecture after the deleted one)
    await api(`${base}/${encodeURIComponent(unit.id)}`, { method: "DELETE" });
    await reload();
    const undo = at ? { label: "Undo", run: () => act(async () => {
      await api(base, { method: "POST", json: { session: at.session, t: at.start, name: unit.name, id: unit.id } });
      toast(`${unit.name} restored.`); await reload({ id: unit.id });
    }) } : null;
    toast(`${unit.name} merged into ${prevName}.`, undo);
    if (undo) $("#toast .toast-action").focus();  // the focused × is gone: keep the keyboard user next to Undo
  });

  const lectures = () => Object.fromEntries(u.sessions.map((s) => [s.name, s]));
  function header(unit, i, prevName, by) {
    const name = h("input", { class: "unit-name", value: unit.name, "aria-label": `Name of unit ${i + 1}`, maxlength: "100" });
    name.addEventListener("change", async () => {
      try {
        await api(`${base}/${encodeURIComponent(unit.id)}`, { method: "PATCH", json: { name: name.value } });
        name.defaultValue = name.value; unit.name = name.value.trim(); toast("Renamed.");
        syncExams(await api(`/api/classes/${enc}/exams`));
      } catch (e) { toast(e.message); name.value = name.defaultValue; }
    });
    const x = i ? h("button", { type: "button", class: "unit-remove", text: "×", "aria-label": "Remove this boundary (merge into previous unit)",
      title: `Remove this boundary (${unit.name} joins ${prevName})` }) : null;
    x?.addEventListener("click", () => remove(unit, prevName));
    const first = unit.spans[0] && by[unit.spans[0].session];
    return h("li", { class: "unit-head", "data-unit": unit.id }, name, h("span", { class: "unit-rule", "aria-hidden": "true" }), x,
      unit.moved ? h("p", { class: "warning unit-warn", text: `The lecture where this unit started was moved or deleted, so it now starts at ${first ? first.title : "the next lecture"}. Check this boundary.` }) : null,
      unit.spans.length ? null : h("p", { class: "muted unit-warn", text: "No lectures in this unit." }));
  }
  function lectureRow(sp, s) {
    const whole = sp.start === 0 && sp.end === null;
    const range = whole ? [fmtDay(s.created), s.duration_s ? clock(s.duration_s) : ""].filter(Boolean).join(" · ")
      : `${clock(sp.start)}–${sp.end === null ? "end" : clock(sp.end)}`;
    const chips = (sugg || []).filter((x) => !x.at_lecture_start && x.session === sp.session && x.t >= sp.start && (sp.end === null || x.t < sp.end)).map((x) => {
      const b = h("button", { type: "button", class: "chip suggest", text: `Suggested split at ${clock(x.t)} · Add`,
        "aria-label": `Add the suggested boundary at ${clock(x.t)} in ${s.title}` });
      b.addEventListener("click", () => insert(x.session, x.t));
      return b;
    });
    return h("li", { class: whole ? "lecture" : "lecture part" },
      h("a", { class: "title", href: sessionUrl(slug, sp.session, sp.start), text: whole ? s.title : `${s.title} · ${range}` }),
      whole ? h("span", { class: "muted", text: range }) : null, h("span", { class: "spacer" }), chips);
  }
  function gap(s) {
    const shift = (sugg || []).some((x) => x.at_lecture_start && x.session === s.name);
    const b = h("button", { type: "button", class: shift ? "gap-btn suggested" : "gap-btn", title: "Start a new unit here",
      "aria-label": `Start a new unit before ${s.title}${shift ? " (suggested: topic shift)" : ""}` },
      h("span", { class: "gap-plus", "aria-hidden": "true", text: "+" }),
      h("span", { class: "gap-label", "aria-hidden": "true", text: shift ? "Suggested: topic shift" : "Start a new unit here" }));
    b.addEventListener("click", () => insert(s.name, 0));
    return h("li", { class: "gap", "data-session": s.name }, b);
  }
  function draw() {
    const by = lectures(), rows = [];
    const lec = (name) => by[name] || { name, title: name };
    u.units.forEach((unit, i) => {
      rows.push(header(unit, i, i ? u.units[i - 1].name : "", by));
      unit.spans.forEach((sp, k) => {
        if (k && sp.start === 0) rows.push(gap(lec(sp.session)));  // no gap where a header already is
        rows.push(lectureRow(sp, lec(sp.session)));
      });
    });
    timeline.replaceChildren(...rows);
    const clear = h("button", { type: "button", class: "link", text: "Clear suggestions" });
    clear.addEventListener("click", () => { sugg = null; draw(); });
    const shown = timeline.querySelectorAll(".gap-btn.suggested, .chip.suggest").length;  // e.g. not where a header already is
    note.replaceChildren(...(sugg ? [shown ? `${shown} suggested boundar${shown === 1 ? "y is" : "ies are"} marked below; nothing changes until you add one. ` : "No clear topic shift found. ",
      sugg.notIndexed ? `${sugg.notIndexed} lecture(s) are not indexed for search by meaning yet, so they were not considered. ` : "", clear] : []));
    after.replaceChildren(...[
      u.orphans.length ? h("p", { class: "warning", text: `${u.orphans.join(", ")} started in a moved or deleted lecture after the last one, so it has no lectures now.` }) : null,
      h("p", { class: "muted", text: u.sessions.length ? "To split inside a lecture, open it and use Start a new unit here." : "No lectures yet." })].filter(Boolean));
  }

  const suggest = h("button", { type: "button", text: "Suggest boundaries" });
  suggest.addEventListener("click", async () => {
    suggest.disabled = true;
    try { const s = await api(`/api/classes/${enc}/unit-suggestions`); sugg = Object.assign(s.suggestions, { notIndexed: s.not_indexed.length }); draw(); }
    catch (e) { toast(e.message); }
    suggest.disabled = false;
  });
  draw();
  box.append(h("div", { class: "row" }, h("h2", { id: "units-h", text: "Units" }), h("span", { class: "spacer" }), suggest),
    h("p", { class: "muted", text: "Units run from one boundary to the next." }), note, timeline, after);
  return box;
}

function examsCard(slug, u, exams) {
  const box = h("section", { class: "card stack", id: "exams", "aria-labelledby": "exams-h" });
  const name = h("input", { id: "exam-name", placeholder: "e.g. Midterm 1", maxlength: "100" });
  const checks = u.units.map((unit) => ({ unit, box: h("input", { type: "checkbox", value: unit.id }) }));
  const from = h("input", { type: "date", "aria-label": "From" }), to = h("input", { type: "date", "aria-label": "To" });
  const save = h("button", { type: "button", class: "primary", text: "Save exam" });
  save.addEventListener("click", async () => {
    const ids = checks.filter((c) => c.box.checked).map((c) => c.unit.id);
    const body = ids.length ? { name: name.value, units: ids } : { name: name.value, from: from.value, to: to.value };
    try { const e = await api(`/api/classes/${slug}/exams`, { method: "POST", json: body }); location.hash = `#/exam/${slug}/${e.id}`; } catch (e) { toast(e.message); }
  });
  box.append(h("h2", { id: "exams-h", text: "Exams" }),
    exams.length ? h("ul", { class: "list" }, exams.map((e) => h("li", {}, h("a", { class: "title", href: `#/exam/${slug}/${e.id}`, text: e.name }),
      h("span", { class: "muted", text: e.units ? e.scope_units.map((x) => x.name).join(", ") : `${e.from} to ${e.to}` })))) : h("p", { class: "muted", text: "No exams yet." }),
    h("label", { for: "exam-name", text: "New exam" }), name,
    h("p", { class: "muted", text: "Scope: tick units, or leave them unticked and give a date range." }),
    h("div", { class: "row" }, checks.map((c) => h("label", { class: "check" }, c.box, ` ${c.unit.name}`))),
    h("div", { class: "row" }, "From", from, "to", to),
    h("div", { class: "row" }, save));
  return box;
}

async function examView(slug, id) {
  const view = stage();
  const e = await api(`/api/exams/${encodeURIComponent(slug)}/${encodeURIComponent(id)}`);
  const titles = e.titles;
  const q = h("input", { type: "search", placeholder: "Search inside this exam", "aria-label": "Search inside this exam", maxlength: "500" });
  const mode = h("select", { "aria-label": "Search type" }, h("option", { value: "exact", text: "Exact words" }), h("option", { value: "concept", text: "By meaning" }));
  const go = h("button", { type: "button", text: "Search" });
  const search = () => { if (q.value.trim()) location.hash = `#/search?${new URLSearchParams({ q: q.value.trim(), mode: mode.value, class: slug, exam: id })}`; };
  go.addEventListener("click", search); q.addEventListener("keydown", (ev) => { if (ev.key === "Enter") search(); });

  const busy = e.job === "queued" || e.job === "running";
  const gen = h("button", { type: "button", class: "primary", text: e.material === "none" ? "Generate study material" : "Generate again", disabled: busy || e.material === "user" });
  gen.addEventListener("click", async () => { try { const r = await api(`/api/exams/${slug}/${id}/generate`, { method: "POST" }); if (r.blocked) toast(`Queued. Waiting: ${r.blocked}.`); route(); } catch (err) { toast(err.message); } });
  if (busy && view.live()) pollTimer = startPoll(async () => { const n = await api(`/api/exams/${slug}/${id}`).catch((x) => x); if (n.status === 404 || n.job && n.job !== e.job) route(); }, 3000);
  const material = h("div", { class: "md", id: "exam-material" });
  if (e.material !== "none") {
    try { material.append(...mdBlocks(String(await api(`/files-exam/${slug}/${id}/exam.md`)).split(/\r?\n/).filter((l) => !/^#\s/.test(l)))); }
    catch (err) { material.append(h("p", { text: err.message })); }
  }
  const audience = h("input", { type: "checkbox", id: "bundle-audience", checked: false });
  const bundle = h("a", { class: "button", id: "bundle-link", href: `/api/exams/${slug}/${id}/bundle`, download: "", text: "Download bundle (.md)" });
  audience.addEventListener("change", () => { bundle.href = `/api/exams/${slug}/${id}/bundle?include_audience=${audience.checked}`; });

  view.replaceChildren(
    h("p", {}, h("a", { href: `#/class/${slug}`, text: `← ${slug}` })),
    h("h1", { text: e.name }),
    e.moved.length ? h("p", { class: "warning", text: `The lecture where ${e.moved.join(", ")} started was moved or deleted; it now starts at the next lecture. Check the class's units.` }) : "",
    h("section", { class: "card stack" }, h("h2", { text: "Scope" }),
      h("ul", { class: "list" }, e.scope_units.map((x) => h("li", {}, h("div", { class: "stack" }, h("strong", { text: x.name }),
        h("div", { class: "muted" }, x.spans.map((sp, k) => [k ? " · " : "", h("a", { href: sessionUrl(slug, sp.session, sp.start), text: spanText(sp, titles) })])))))),
      h("div", { class: "row" }, q, mode, go)),
    h("section", { class: "card stack" }, h("h2", { text: "Study material" }),
      h("p", { class: "muted", text: busy ? (e.blocked ? `Queued, waiting: ${e.blocked}.` : "Being written by the local model… this page updates when done.")
        : e.stale ? "Out of date: the exam's units changed since this was generated, so it may not match the scope. It is left out of the bundle."
        : e.material === "user" ? "You edited this material, so the app will not replace it." : e.error ? `Could not generate: ${e.error}` : "Combined unit summaries and a key-terms glossary, from the lectures in scope only." }),
      h("div", { class: "row" }, gen), material),
    h("section", { class: "card stack" }, h("h2", { text: "Bundle for another AI (optional)" }),
      h("p", { class: "muted", text: "A Markdown file you download and may give to an AI service yourself. The app never sends it anywhere." }),
      h("label", { class: "check" }, audience, " Include student questions"),
      h("p", { class: "muted", text: "Off by default: classmates did not agree to have their words shared with an outside service." }),
      h("div", { class: "row" }, bundle)));
}

// ------------------------------------------------------------------ search
const searchForm = $("#search-form"), searchQ = $("#search-q"), searchMode = $("#search-mode");
searchForm.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = searchQ.value.trim();
  if (!q) return;
  const next = `#/search?${new URLSearchParams({ q, mode: searchMode.value })}`;
  if (location.hash === next) route(); else location.hash = next;
});
const jumpUrl = (r) => `#/session/${encodeURIComponent(r.class)}/${encodeURIComponent(r.session)}?t=${r.start}`;

async function searchView(params) {
  const view = stage();
  const q = params.get("q") || "", mode = params.get("mode") === "concept" ? "concept" : "exact";
  searchQ.value = q; searchMode.value = mode;
  view.replaceChildren(h("h1", { text: "Search" }), h("p", { class: "muted", text: mode === "concept" ? "Finding passages about this idea…" : "Searching…" }));
  let r;
  const scope = {};
  if (params.get("class")) scope.class_slug = params.get("class");
  if (params.get("exam")) scope.exam = params.get("exam");
  try { r = await api(`/api/search?${new URLSearchParams({ q, mode, ...scope })}`); }
  catch (e) { view.replaceChildren(h("h1", { text: "Search" }), h("p", { class: "warning", role: "alert", text: e.message })); return; }
  const parts = [h("h1", { text: "Search" })];
  if (scope.exam) parts.push(h("p", {}, "Inside the exam ", h("a", { href: `#/exam/${scope.class_slug}/${scope.exam}`, text: scope.exam }),
    h("span", { class: "muted", text: " (nothing from outside its units or dates)." })));
  const where = (x) => [h("a", { class: "title", href: jumpUrl(x), text: x.title }),
    h("span", { class: "muted", text: ` · ${x.class_name}${x.unit ? " · " + x.unit : ""} · ${fmtDate(x.created)}` })];
  if (mode === "exact") {
    parts.push(h("p", { class: "muted", text: r.message || `${r.total} match${r.total === 1 ? "" : "es"} for “${q}” in ${r.sessions} lecture${r.sessions === 1 ? "" : "s"} (exact words; case and punctuation ignored).` }));
    if (r.results.length) parts.push(h("ul", { class: "list results", id: "search-results" }, r.results.map((x) =>
      h("li", { class: "result" },
        h("a", { class: "time mono", href: jumpUrl(x), title: "Play from here", text: hms(x.start) }),
        h("div", {}, h("div", {}, where(x)),
          h("p", { class: "context" }, x.context.before, h("mark", { text: x.context.match }), x.context.after))))));
  } else {
    const st = r.status || {};
    if (r.message) parts.push(h("p", { class: "muted", text: r.message }));
    else parts.push(h("p", { class: "muted", text: `Closest passages to “${q}” by meaning (they may not use your words).` }));
    if (st.pending) {
      const b = h("button", { type: "button", text: "Index them now" });
      b.addEventListener("click", async () => { b.disabled = true; try { const x = await api("/api/search/index-missing", { method: "POST" }); toast(`Queued ${x.queued} lecture(s) for indexing.`); } catch (e) { toast(e.message); b.disabled = false; } });
      parts.push(h("div", { class: "row warning" }, h("span", { text: `${st.pending} of ${st.transcribed} transcribed lecture(s) are not indexed for search by meaning yet.` }), b));
    }
    if (r.results.length) parts.push(h("ul", { class: "list results", id: "search-results" }, r.results.map((x) =>
      h("li", { class: "result" },
        h("a", { class: "time mono", href: jumpUrl(x), title: "Play from here", text: hms(x.start) }),
        h("div", {}, h("div", {}, where(x), h("span", { class: "muted", text: ` · ${hms(x.start)}–${hms(x.end)} · match ${x.score.toFixed(2)}` })),
          h("p", { class: "context", text: x.text }))))));
  }
  view.replaceChildren(...parts);
}

const sessionHash = (x) => `#/session/${encodeURIComponent(x.class)}/${encodeURIComponent(x.session)}`;

// Import page: the file was not imported (nothing was kept). `again` uploads it once more with Import anyway and says so.
function alreadyImported(what, seen, again) {
  const any = h("button", { type: "button", text: "Import anyway" });
  const box = h("div", { class: "warning", role: "alert" },
    h("strong", { text: `${what}: already imported as ` }), h("a", { href: sessionHash(seen), text: seen.title }), ` (${fmtDate(seen.created)}).`,
    h("div", { class: "row" }, h("a", { class: "button", href: sessionHash(seen), text: "Open it" }), any));
  any.addEventListener("click", async () => {
    any.disabled = true;
    try {
      const r = await again();
      const others = box.parentElement.childElementCount > 1;  // other files still wait for an answer
      box.remove();
      if (!others) location.hash = sessionHash({ class: r.class, session: r.session });
    } catch (e) { toast(e.message); any.disabled = false; }
  });
  return box;
}

// Lecture page: one banner per possible duplicate. Nothing moves until the user presses the button.
async function duplicateBanners(slug, name) {
  const r = await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(name)}/duplicates`).catch(() => null);
  return (r?.matches || []).map((m, i) => duplicateBanner(m, i));
}

function duplicateBanner(m, i) {
  const { this: mine, other } = m;
  const line = (label, fn) => h("tr", {}, h("th", { scope: "row", text: label }), [mine, other].map((x) => h("td", { text: fn(x) })));
  const compare = h("table", { class: "dup-compare" },
    h("thead", {}, h("tr", {}, h("td"), h("th", { scope: "col", text: "This lecture" }), h("th", { scope: "col" }, h("a", { href: sessionHash(other), text: "The other one" })))),
    h("tbody", {},
      line("Date", (x) => fmtDate(x.created) || "unknown"),
      line("Duration", (x) => x.duration_s ? hms(x.duration_s) : "unknown"),
      line("Source", (x) => x.source),
      line("Processed", (x) => x.processed ? "Yes" : "Not yet")));
  const pick = (value, x, label) => h("label", { class: "check" },
    h("input", { type: "radio", name: `dup-move-${i}`, value, checked: m.move === value }), ` ${label}: ${x.title}`);
  const which = h("fieldset", { class: "dup-which" }, h("legend", { text: "Which one to move?" }), pick("this", mine, "This lecture"), pick("other", other, "The other one"));
  const keep = h("button", { type: "button", text: "Keep both" });
  const move = h("button", { type: "button", class: "danger", text: "Move to Recently deleted" });
  const box = h("section", { class: "warning dup-banner", "aria-label": `Possible duplicate of ${other.title}` },
    h("strong", {}, "Possible duplicate of ", h("a", { href: sessionHash(other), text: other.title })),
    ` (${m.labels.join(", ")})`, compare, which, h("div", { class: "row" }, keep, move));
  keep.addEventListener("click", async () => {
    keep.disabled = true;
    try {
      await api(`/api/sessions/${encodeURIComponent(mine.class)}/${encodeURIComponent(mine.session)}/duplicates/dismiss`, { method: "POST", json: { class: other.class, session: other.session } });
      box.remove(); toast("Kept both. This pair won't be flagged again.");
    } catch (e) { toast(e.message); keep.disabled = false; }
  });
  move.addEventListener("click", async () => {
    const thisOne = box.querySelector("input:checked").value === "this";
    const target = thisOne ? mine : other;
    const back = sessionHash(mine);
    move.disabled = true;
    try {
      const r = tookVersion(await api(`/api/sessions/${encodeURIComponent(target.class)}/${encodeURIComponent(target.session)}/delete`, { method: "POST" }));
      toast(`Moved “${target.title}” to Recently deleted.`, { label: "Undo", run: async () => {
        try { tookVersion(await api(`/api/trash/${r.id}/restore`, { method: "POST" })); toast(`Restored “${target.title}”.`); if (location.hash === back) route(); else location.hash = back; }
        catch (e) { toast(e.message); }
      } });
      $("#toast .toast-action").focus();  // the focused button is gone: keep the keyboard user next to Undo
      if (thisOne) location.hash = sessionHash(other); else route();
    } catch (e) { toast(e.message); move.disabled = false; }
  });
  return box;
}

// The player holds the audio file open, and Windows won't move a folder with an open file: let go of it first.
// Returns a function that puts the player back (the move didn't happen).
function releasePlayer() {
  const a = $("audio"), src = a?.getAttribute("src");
  if (!src) return () => {};
  const at = a.currentTime, playing = !a.paused;
  a.pause(); a.removeAttribute("src"); a.load();
  return () => {
    a.setAttribute("src", src);
    a.addEventListener("loadedmetadata", () => { a.currentTime = at; if (playing) a.play().catch(() => {}); }, { once: true });
    a.load();
  };
}
const moveLecture = (c, s, json) => api(`/api/sessions/${encodeURIComponent(c)}/${encodeURIComponent(s)}/move`, { method: "POST", json }).then(tookVersion);

// One move from the lecture page, with its Undo toast. False if it was refused (the toast says why).
let relocating = false;  // the banner's Move and "Move to…" are both on the page: one move at a time
async function relocate(slug, name, to) {
  if (relocating) return false;
  relocating = true;
  const restore = releasePlayer();
  try {
    const m = await moveLecture(slug, name, { to });
    toast(`Moved to ${m.class_name}.`, { label: "Undo", run: async () => {
      const again = releasePlayer();
      try {
        const b = await moveLecture(m.class, m.session, { to: m.from_class, name: m.from_session, undo: true });
        toast(b.session === m.from_session ? "Moved back." : "Moved back. A newer lecture has its old folder name, so it got the next free one.");
        if (location.hash === sessionHash(b)) route(); else location.hash = sessionHash(b);  // the same after Back: no hashchange
      }
      catch (e) { again(); toast(e.message); }
    } });
    $("#toast .toast-action").focus();  // the focused button is gone: keep the keyboard user next to Undo
    location.hash = sessionHash(m);
    return true;
  } catch (e) { restore(); toast(e.message); return false; }
  finally { relocating = false; }
}

// Lecture page: "Move to…" for any lecture, with or without a suggestion. The class list is read when it opens.
function moveControl(slug, name, state) {
  const go = h("button", { type: "button", class: "primary", id: "move-go", text: "Move", disabled: true });
  const cancel = h("button", { type: "button", text: "Cancel" });
  const panel = h("div", { class: "row", id: "move-panel" });  // the picker is only in the page while open: a select counts as unsaved input
  panel.hidden = true;
  const open = h("button", { type: "button", class: "ghost", id: "move-session", text: "Move to…", title: "Move to another class",
    "aria-expanded": "false", disabled: state === "recording" || state === "running" });
  const close = () => { panel.hidden = true; panel.replaceChildren(); open.setAttribute("aria-expanded", "false"); };
  let picker = null;
  open.addEventListener("click", async () => {
    if (!panel.hidden) return close();
    try {
      const targets = (await api("/api/classes")).filter((c) => c.has_class_file && c.slug !== slug && c.slug !== "unsorted");
      if (!targets.length && slug === "unsorted") return toast("There is no other class yet. Add one first.");
      picker = h("select", { id: "move-target", "aria-label": "Move this lecture to", onchange: () => { go.disabled = !picker.value; } },
        h("option", { value: "", text: "Choose a class…" }), targets.map((c) => h("option", { value: c.slug, text: c.name })),
        slug === "unsorted" ? null : h("option", { value: "unsorted", text: "Unsorted" }));
      go.disabled = true;
      panel.replaceChildren(picker, go, cancel);
      panel.hidden = false;
      open.setAttribute("aria-expanded", "true");
      picker.focus();
    } catch (e) { toast(e.message); }
  });
  cancel.addEventListener("click", () => { close(); open.focus(); });
  go.addEventListener("click", async () => {
    go.disabled = true;
    if (!await relocate(slug, name, picker.value)) go.disabled = false;
  });
  return { open, panel };
}

// Lecture page: "Looks like <Class>". Nothing moves until the user presses Move; Keep here is for good.
async function suggestionBanner(slug, name) {
  const r = await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(name)}/suggestion`).catch(() => null);
  const s = r?.suggestion;
  if (!s) return null;
  const move = h("button", { type: "button", class: "primary", text: `Move to ${s.class_name}` });
  const keep = h("button", { type: "button", text: "Keep here" });
  const box = h("section", { class: "warning dup-banner", id: "class-suggestion", "aria-label": `Looks like ${s.class_name}` },
    h("strong", { text: `Looks like ${s.class_name}` }), ` · ${s.reason}`, h("div", { class: "row" }, move, keep));
  keep.addEventListener("click", async () => {
    keep.disabled = true;
    try { await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(name)}/keep-here`, { method: "POST" }); box.remove(); toast("Kept here. This lecture won't be asked about again."); }
    catch (e) { toast(e.message); keep.disabled = false; }
  });
  move.addEventListener("click", async () => {
    move.disabled = true;
    if (!await relocate(slug, name, s.class)) move.disabled = false;
  });
  return box;
}

async function sessionView(slug, name, startAt = null) {
  const view = stage();
  const s = await api(`/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(name)}`);
  const audio = s.audio ? h("audio", { controls: true, preload: "metadata", src: fileUrl(slug, name, s.audio) }) : null;
  // lead-in for bookmarks and warnings; transcript lines seek to their exact start
  const seek = (t, lead = 0.3) => { if (audio) { audio.currentTime = Math.max(0, t - lead); audio.play().catch(() => {}); } };
  const transcriptBox = h("section", { class: "transcript", id: "transcript", "aria-label": "Transcript" });
  const titleEl = h("h1", { id: "lecture-title", text: s.title });

  const move = moveControl(slug, name, s.state);
  const parts = [
    h("p", {}, h("a", { href: `#/class/${slug}`, text: `← ${slug}` })),
    h("div", { class: "row" }, titleEl, renameButton(slug, s, titleEl), h("span", { class: "spacer" }), chip(s.state),
      move.open,
      h("button", { type: "button", class: "ghost", id: "delete-session", text: "Delete", title: "Move to Recently deleted",
        disabled: s.state === "recording" || s.state === "running", onclick: () => deleteLecture(slug, s, () => { location.hash = `#/class/${slug}`; }) })),
    h("p", { class: "muted", text: `${fmtDate(s.created)}${s.duration_s ? " · " + hms(s.duration_s) : ""}` }),
    s.original ? h("p", { class: "muted", id: "original-file" }, "Original file: ", h("span", { class: "mono", text: s.original })) : null,
    move.panel,
  ];
  if (audio) parts.push(h("div", { class: "player" }, audio));
  const units = await api(`/api/classes/${encodeURIComponent(slug)}/units`).catch(() => null);
  if (units) {
    const here = units.units.flatMap((u) => u.spans.filter((sp) => sp.session === name).map((sp) => ({ u, sp })));
    const mark = h("button", { type: "button", id: "new-unit-here", text: "Start a new unit here",
      title: "A new unit begins at the current playback time" });
    mark.addEventListener("click", async () => {
      const t = audio ? audio.currentTime : 0;
      const label = prompt(`Name of the unit that starts at ${hms(t)} (optional):`, "");
      if (label === null) return;
      try { await api(`/api/classes/${slug}/units`, { method: "POST", json: { session: name, t, name: label } }); toast("New unit added."); route(); } catch (e) { toast(e.message); }
    });
    parts.push(h("div", { class: "row units-here" }, h("span", { class: "muted", text: "Units:" }),
      here.map(({ u, sp }) => h("button", { type: "button", class: "ghost", onclick: () => seek(sp.start, 0), text: `${u.name} (${hms(sp.start)}–${sp.end === null ? "end" : hms(sp.end)})` })),
      h("span", { class: "spacer" }), mark));
  }
  if (s.bookmarks.length) parts.push(h("div", { class: "bookmarks row" }, h("span", { class: "muted", text: "Bookmarks:" }),
    s.bookmarks.map((b) => h("button", { type: "button", class: "ghost", onclick: () => seek(b.audio_s), text: `${hms(b.audio_s)}${b.label ? " " + b.label : ""}` }))));
  const check = (s.meta && s.meta.audio_check) || {};
  if (check.warning && s.state !== "no audio") parts.push(h("p", { class: "warning", id: "audio-warning", text: check.warning }));
  const ds = s.dropped_speech;
  if (ds && ds.total_s > 0) parts.push(h("div", ds.reliable === false ? { class: "muted", id: "dropped-speech" } : { class: "warning", role: "alert", id: "dropped-speech" },
    h("strong", { text: ds.reliable === false
      ? `Approximate check (batched decoding): up to ${Math.round(ds.total_s)} s may be missing. `
      : `About ${Math.round(ds.total_s)} s of speech may be missing from this transcript. ` }),
    "Listen to: ", ds.stretches.map((g, i) => [i ? ", " : "", h("button", { type: "button", class: "link", onclick: () => seek(g.start), text: `${hms(g.start)}–${hms(g.end)}` })])));
  const duplicates = h("div", { id: "duplicates" });  // filled below: a hint never holds up the page
  duplicates.hidden = true;
  const suggestion = h("div", { id: "suggestion" });  // the same
  suggestion.hidden = true;
  parts.push(suggestion, duplicates);
  // Progress of this lecture's jobs, updated by the status poll (no poll of its own).
  parts.push(processingCard(s));
  if (s.state === "no audio") {
    const b = h("button", { type: "button", text: "Transcribe anyway" });
    b.addEventListener("click", async () => {
      try { const r = await api(`/api/sessions/${slug}/${name}/process`, { method: "POST", json: { anyway: true } }); if (r.blocked) toast(`Queued. Waiting: ${r.blocked}.`); route(); } catch (e) { toast(e.message); }
    });
    parts.push(h("div", { class: "warning", role: "alert", id: "no-audio" },
      h("strong", { text: "No usable audio. " }), check.reason || "This recording has no speech the app can transcribe.",
      h("div", { class: "row" }, b, h("span", { class: "muted", text: "Nothing was sent to the GPU. The audio file is kept unchanged." }))));
  } else if (s.state === "failed" || s.state === "not processed") {
    const b = h("button", { type: "button", class: "primary", text: s.state === "failed" ? "Try again" : "Process now" });
    b.addEventListener("click", async () => {
      try { const r = await api(`/api/sessions/${slug}/${name}/process`, { method: "POST" }); if (r.blocked) toast(`Queued. Waiting: ${r.blocked}.`); route(); } catch (e) { toast(e.message); }
    });
    parts.push(h("div", { class: "row" }, b, h("span", { class: "muted", text: s.state === "failed" ? "Processing failed. See the queue file for details."
      : "Transcribe, label speakers, summarize and index for search (uses the GPU; best when plugged in)." })));
  }
  const summaryBox = h("section", { class: "card summary", id: "summary", "aria-label": "Summary" });
  if (s.files.includes("transcript.json")) parts.push(summaryBox);
  parts.push(transcriptBox);
  parts.push(h("section", { class: "card stack files" }, h("h2", { text: "Files" }),
    h("div", {}, s.files.filter((f) => f !== "meta.json" && f !== "bookmarks.json").map((f) => h("a", { href: fileUrl(slug, name, f), target: "_blank", rel: "noopener", text: f }))),
    h("p", { class: "muted" }, "Folder: ", h("code", { class: "path", text: s.folder }))));
  view.replaceChildren(...parts);
  duplicateBanners(slug, name).then((b) => { duplicates.hidden = !b.length; duplicates.replaceChildren(...b); });
  suggestionBanner(slug, name).then((b) => { suggestion.hidden = !b; if (b) suggestion.replaceChildren(b); });
  if (s.files.includes("transcript.json")) {
    await renderSummary(slug, name, s.summary, summaryBox);
    await renderTranscript(slug, name, transcriptBox, audio, seek);
  }
  if (startAt !== null && Number.isFinite(startAt)) {
    jumpTo(startAt, audio, transcriptBox);
    history.replaceState(null, "", `#/session/${encodeURIComponent(slug)}/${encodeURIComponent(name)}`);  // a later re-render must not jump again
  }
}

// From a search result: cue the player a moment before the match, mark and show its transcript line.
function jumpTo(t, audio, box) {
  const rows = [...box.querySelectorAll(".seg")];
  let row = null;
  for (const r of rows) { if (Number(r.dataset.start) <= t + 0.01) row = r; else break; }
  if (row) { row.classList.add("jump"); row.scrollIntoView({ block: "center" }); }
  if (!audio) return;
  const go = () => { audio.currentTime = Math.max(0, t - 1); audio.play().catch(() => {}); };
  if (audio.readyState >= 1) go(); else audio.addEventListener("loadedmetadata", go, { once: true });
}

// Minimal, safe Markdown subset (headings, bullets, **bold**, paragraphs). Text only via textContent.
function mdInline(text) {
  const out = []; const re = /\*\*(.+?)\*\*/g; let last = 0, m;
  while ((m = re.exec(text))) { out.push(text.slice(last, m.index), h("strong", { text: m[1] })); last = re.lastIndex; }
  out.push(text.slice(last)); return out;
}
function mdBlocks(lines) {
  const out = []; let list = null; let para = [];
  const flush = () => { if (para.length) { out.push(h("p", {}, mdInline(para.join(" ")))); para = []; } list = null; };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\s*<!--.*-->\s*$/.test(line)) continue;
    const hm = line.match(/^(#{1,6})\s+(.*)$/); const li = line.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/);
    if (hm) { flush(); out.push(h(`h${Math.min(6, hm[1].length + 1)}`, {}, mdInline(hm[2]))); }
    else if (li) { if (para.length) { const l = list; flush(); list = l; } if (!list) { list = h("ul"); out.push(list); } list.append(h("li", {}, mdInline(li[1]))); }
    else if (!line.trim()) flush();
    else if (/^_[^_].*_$/.test(line.trim())) { flush(); out.push(h("p", { class: "muted" }, h("em", { text: line.trim().slice(1, -1) }))); }
    else { list = null; para.push(line.trim()); }
  }
  flush(); return out;
}

async function renderSummary(slug, name, sum, box) {
  const head = h("div", { class: "row" }, h("h2", { text: "Summary" }), h("span", { class: "spacer" }));
  const summarizeBtn = (label) => {
    const b = h("button", { type: "button", class: "primary", text: label });
    b.addEventListener("click", async () => { b.disabled = true; try { await api(`/api/sessions/${slug}/${name}/summarize`, { method: "POST" }); route(); } catch (e) { toast(e.message); b.disabled = false; } });
    return b;
  };
  if (sum.state === "queued" || sum.state === "running") {
    box.replaceChildren(head, h("p", { class: "muted", text: sum.state === "queued" ? "Waiting to be summarized; the progress is shown above." : "Being written with the local model (progress above); it appears here when done." }));
    return;
  }
  if (sum.kind === "none" || sum.kind === "placeholder") {
    box.replaceChildren(head,
      sum.state === "failed" ? h("p", { class: "warning", text: `The summary could not be generated: ${sum.error || "unknown error"}` }) : h("p", { class: "muted", text: "No summary yet." }),
      h("div", { class: "row" }, summarizeBtn(sum.state === "failed" ? "Try again" : "Summarize")));
    return;
  }
  let text;
  try { text = await api(fileUrl(slug, name, "summary.md")); } catch (e) { box.replaceChildren(head, h("p", { text: e.message })); return; }
  const lines = String(text).split(/\r?\n/);
  const cut = lines.findIndex((l) => /^##\s+Detailed notes\s*$/i.test(l));
  const glance = cut >= 0 ? lines.slice(0, cut) : lines;
  const body = glance.filter((l) => !/^#\s/.test(l));  // the file's own "# Summary: …" title
  const parts = [head];
  if (sum.warning) parts.push(h("p", { class: "muted", text: sum.warning }));
  if (sum.kind === "user") parts.push(h("p", { class: "muted", text: "This summary was written or edited by you; the app will not replace it." }));
  parts.push(h("div", { class: "md glance" }, mdBlocks(body)));
  if (cut >= 0) parts.push(h("details", { class: "md notes" }, h("summary", { text: "Detailed notes" }), mdBlocks(lines.slice(cut + 1))));
  box.replaceChildren(...parts);
}

async function renderTranscript(slug, name, box, audio, seek) {
  const doc = await api(`/api/sessions/${slug}/${name}/transcript`);
  let editing = false;
  const rows = doc.segments.map((seg) => {
    const txt = h("span", { class: "txt", text: seg.text });
    const who = h("span", { class: `who ${seg.role}`, text: `${seg.speaker} · ${seg.role}` });
    const row = h("div", { class: "seg", "data-start": seg.start, "data-id": seg.id }, h("span", { class: "t mono", text: hms(seg.start) }), who, txt);
    row.addEventListener("click", () => { if (!editing) seek(seg.start, 0); });
    return { seg, row, txt, who };
  });
  const editBtn = h("button", { type: "button", text: "Edit transcript" });
  const saveBtn = h("button", { type: "button", class: "primary", text: "Save changes", hidden: true });
  editBtn.addEventListener("click", () => {
    editing = !editing; editBtn.textContent = editing ? "Cancel" : "Edit transcript"; saveBtn.hidden = !editing;
    for (const r of rows) {
      r.txt.contentEditable = editing ? "true" : "false";
      if (editing) {
        const sel = h("select", { "aria-label": "Speaker" }, doc.speakers.map((sp) => h("option", { value: sp.id, text: `${sp.id} (${sp.role})`, selected: sp.id === r.seg.speaker })));
        r.who.replaceChildren(sel); r.sel = sel;
      } else { r.who.textContent = `${r.seg.speaker} · ${r.seg.role}`; r.txt.textContent = r.seg.text; r.sel = null; }
    }
  });
  saveBtn.addEventListener("click", async () => {
    const changes = rows.filter((r) => r.txt.textContent.trim() !== r.seg.text || (r.sel && r.sel.value !== r.seg.speaker))
      .map((r) => ({ id: r.seg.id, text: r.txt.textContent.trim(), speaker: r.sel ? r.sel.value : r.seg.speaker }));
    try {
      const r = await api(`/api/sessions/${slug}/${name}/transcript`, { method: "PUT", json: { segments: changes } });
      toast(r.kept_hand_edited.length ? `Saved. ${r.kept_hand_edited.join(" and ")} kept as you edited it on disk.` : `Saved ${changes.length} change(s).`);
      route();
    }
    catch (e) { toast(e.message); }
  });
  const zoomNames = (doc.zoom_speakers || []).length ? h("p", { class: "muted", id: "zoom-speakers", text: "Names from Zoom's transcript (a hint; labels unchanged): "
    + doc.zoom_speakers.map((z) => `${z.speaker} ≈ ${z.name} (${Math.round(z.share * 100)}%)`).join(", ") }) : null;
  box.replaceChildren(h("div", { class: "row" }, h("h2", { text: "Transcript" }), h("span", { class: "spacer" }), editBtn, saveBtn), zoomNames || "",
    h("p", { class: "muted", text: doc.speakers.map((sp) => `${sp.id} (${sp.role}): ${(sp.speech_s / 60).toFixed(1)} min`).join(" · ") }),
    ...rows.map((r) => r.row));
  if (audio) {
    let current = null;
    audio.addEventListener("timeupdate", () => {
      const t = audio.currentTime; let lo = 0, hi = rows.length - 1, idx = -1;
      while (lo <= hi) { const mid = (lo + hi) >> 1; if (rows[mid].seg.start <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
      const r = idx >= 0 ? rows[idx] : null;
      if (r !== current) { current?.row.classList.remove("current"); r?.row.classList.add("current"); current = r; }
    });
  }
}

// ------------------------------------------------------------------ router
async function route() {
  stopPoll(pollTimer); pollTimer = null;
  const seq = ++routeSeq;
  const [path, query = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").map(decodeURIComponent);
  const params = new URLSearchParams(query);
  shown = { kind: parts[0] || "home", slug: parts[1], name: parts[2] };
  try {
    if (parts[0] === "class" && parts[1]) await classView(parts[1]);
    else if (parts[0] === "session" && parts[2]) await sessionView(parts[1], parts[2], params.has("t") ? Number(params.get("t")) : null);
    else if (parts[0] === "search") await searchView(params);
    else if (parts[0] === "exam" && parts[2]) await examView(parts[1], parts[2]);
    else if (parts[0] === "trash") await trashView();
    else await homeView();
  } catch (e) {
    if (e.status === 404 && await goneView(seq, parts)) return;
    stage(seq).replaceChildren(h("h1", { text: "Something went wrong" }), h("p", { text: e.message }), h("a", { href: "#/", text: "Back to the library" }));
  }
}

// A lecture, class or exam page whose item no longer exists (deleted here, in another tab or in the
// terminal; or an old link). False if the item still exists (the 404 was about something else).
async function goneView(seq, [kind, slug, name]) {
  if (!["session", "class", "exam"].includes(kind) || !slug) return false;
  const own = kind === "session" ? `/api/sessions/${encodeURIComponent(slug)}/${encodeURIComponent(name)}` : `/api/classes/${encodeURIComponent(slug)}`;
  try { await api(own); return false; } catch (e) { if (e.status !== 404) return false; }
  const items = (await api("/api/trash").catch(() => ({ items: [] }))).items;
  const cls = items.find((x) => x.kind === "class" && x.class === slug);
  const lec = kind === "session" && items.find((x) => x.kind === "session" && x.class === slug && x.session === name);
  const what = kind === "session" ? "lecture" : kind;
  const [title, text] = lec ? ["This lecture was deleted", `“${lec.title}” is in Recently deleted, where you can restore it.`]
    : cls ? [kind === "class" ? "This class was deleted" : `This ${what}'s class was deleted`, `“${cls.title}” is in Recently deleted, where you can restore it with all its lectures.`]
    : [`This ${what} no longer exists`, "It was deleted permanently, or the link is out of date."];
  // (for a class or an exam, `own` was the class itself: gone)
  const classExists = kind === "session" && !cls && await api(`/api/classes/${encodeURIComponent(slug)}`).then(() => true, () => false);
  const back = classExists ? h("a", { href: `#/class/${encodeURIComponent(slug)}`, text: "Back to the class" }) : h("a", { href: "#/", text: "Back to the library" });
  stage(seq).replaceChildren(h("h1", { text: title }),
    h("p", { class: "muted", id: "gone", text }),
    h("div", { class: "row" }, h("a", { href: "#/trash", text: "Recently deleted" }), back));
  return true;
}
window.addEventListener("hashchange", route);
refreshRecording().finally(route);  // the first status poll gives the bars their state
