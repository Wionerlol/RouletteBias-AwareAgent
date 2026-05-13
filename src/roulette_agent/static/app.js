// ---- App state ----
const S = {
  apiKey:    localStorage.getItem('apiKey')    || '',
  sessionId: localStorage.getItem('sessionId') || null,
  history:   [],  // [{spin, result, pnl}] newest-first, max 10
};

// ---- Boot ----
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('api-key').value = S.apiKey;

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js').catch(() => {});
  }

  document.getElementById('api-key').addEventListener('change', e => {
    S.apiKey = e.target.value.trim();
    localStorage.setItem('apiKey', S.apiKey);
  });
  document.getElementById('btn-new-session').addEventListener('click', submitNewSession);
  document.getElementById('btn-spin').addEventListener('click', submitSpin);
  document.getElementById('btn-reset').addEventListener('click', resetToNew);
  document.getElementById('result-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') submitSpin();
  });

  if (S.sessionId) {
    loadState();
  } else {
    showView('new');
  }
});

// ---- View switching ----
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  document.getElementById(`view-${name}`).classList.remove('hidden');
}

function resetToNew() {
  localStorage.removeItem('sessionId');
  S.sessionId = null;
  S.history = [];
  showView('new');
}

// ---- Parse a comma-separated list of roulette numbers (shared by history + wheel order) ----
function parseNumberList(raw, label) {
  const tokens = raw.trim().split(/[\s,]+/).filter(Boolean);
  if (!tokens.length) return null;
  return tokens.map(t => {
    if (t === '00') return 37;
    const n = parseInt(t, 10);
    if (isNaN(n) || n < 0 || n > 36) throw new Error(`${label}: 无效数字 "${t}"（范围 0-36 或 "00"）`);
    return n;
  });
}

function parseHistory(raw) {
  const result = parseNumberList(raw, 'Recent draws');
  if (!result || !result.length) throw new Error('请输入至少一个开奖数字');
  return result;
}

// ---- New Session ----
async function submitNewSession() {
  const btn = document.getElementById('btn-new-session');
  const errEl = document.getElementById('error-new');
  errEl.classList.add('hidden');

  S.apiKey = document.getElementById('api-key').value.trim();
  localStorage.setItem('apiKey', S.apiKey);

  let recentHistory;
  try {
    recentHistory = parseHistory(document.getElementById('recent-history').value);
  } catch (e) {
    return showError(errEl, e.message);
  }

  const exclVal = document.querySelector('input[name="excl"]:checked').value;
  const excludedDozens = exclVal === 'none' ? [] : [parseInt(exclVal, 10)];

  // External stats (all optional)
  const extStats = {};
  [['black-pct','black_pct'],['red-pct','red_pct'],['odd-pct','odd_pct'],['even-pct','even_pct']].forEach(([id, key]) => {
    const v = document.getElementById(id).value.trim();
    if (v !== '') extStats[key] = parseFloat(v);
  });
  const extNRaw = document.getElementById('ext-n').value.trim();

  // Custom wheel order (optional)
  let wheelOrder = null;
  const wheelOrderRaw = document.getElementById('wheel-order').value.trim();
  if (wheelOrderRaw) {
    try {
      wheelOrder = parseNumberList(wheelOrderRaw, 'Wheel order');
    } catch (e) {
      return showError(errEl, e.message);
    }
    if (!wheelOrder || wheelOrder.length < 37) {
      return showError(errEl, 'Wheel order 至少需要 37 个数字（欧式）或 38 个（美式）');
    }
    const uniq = new Set(wheelOrder);
    if (uniq.size !== wheelOrder.length) {
      return showError(errEl, 'Wheel order 中有重复数字');
    }
  }

  // Custom payouts (optional — only include bet types the user filled in)
  const PAYOUT_IDS = [
    ['pay-straight', 'straight'],
    ['pay-split',    'split'],
    ['pay-street',   'street'],
    ['pay-corner',   'corner'],
    ['pay-six_line', 'six_line'],
    ['pay-dozen',    'dozen'],
    ['pay-red',      'red'],
  ];
  // Outside bets share the same payout; if user sets red, mirror to all outside bets
  const OUTSIDE_BETS = ['red', 'black', 'odd', 'even', 'low', 'high'];
  const customPayouts = {};
  for (const [id, key] of PAYOUT_IDS) {
    const v = document.getElementById(id).value.trim();
    if (v !== '') {
      const payout = parseInt(v, 10);
      if (isNaN(payout) || payout < 1) return showError(errEl, `Custom payout for ${key} 必须是正整数`);
      if (OUTSIDE_BETS.includes(key)) {
        // Mirror to all outside bets
        OUTSIDE_BETS.forEach(t => { customPayouts[t] = payout; });
      } else {
        customPayouts[key] = payout;
        // Dozen and column share the same payout input
        if (key === 'dozen') customPayouts['column'] = payout;
      }
    }
  }

  const body = {
    wheel_type:      document.getElementById('wheel-type').value,
    bankroll:        parseFloat(document.getElementById('bankroll').value),
    bet_unit:        parseFloat(document.getElementById('bet-unit').value),
    excluded_dozens: excludedDozens,
    recent_history:  recentHistory,
    ...(Object.keys(extStats).length ? { external_stats: extStats } : {}),
    ...(extNRaw ? { external_stats_n_estimate: parseInt(extNRaw, 10) } : {}),
    ...(wheelOrder ? { wheel_order: wheelOrder } : {}),
    ...(Object.keys(customPayouts).length ? { custom_payouts: customPayouts } : {}),
  };

  setBtn(btn, true, 'Analyzing…');
  try {
    const data = await apiPost('/session/new', body);
    S.sessionId = data.session_id;
    localStorage.setItem('sessionId', S.sessionId);
    S.history = [];
    renderSpinView({
      spin_index:    data.spin_index,
      bankroll_now:  data.bankroll_now,
      pnl_last_spin: null,
      bias_report:   data.bias_report,
      next_strategy: data.next_strategy,
      rationale:     data.rationale,
    });
    showView('spin');
  } catch (e) {
    showError(errEl, e.message);
  } finally {
    setBtn(btn, false, 'Start Session →');
  }
}

// ---- Spin ----
async function submitSpin() {
  const btn = document.getElementById('btn-spin');
  const errEl = document.getElementById('error-spin');
  errEl.classList.add('hidden');

  const raw = document.getElementById('result-input').value.trim();
  const resultNumber = parseInt(raw, 10);
  if (isNaN(resultNumber) || resultNumber < 0 || resultNumber > 37) {
    return showError(errEl, '请输入有效数字（0-36）或按 00 按钮');
  }

  setBtn(btn, true, 'Loading…');
  try {
    const data = await apiPost(`/session/${S.sessionId}/spin`, { result_number: resultNumber });
    S.history.unshift({ spin: data.spin_index, result: resultNumber, pnl: data.pnl_last_spin });
    if (S.history.length > 10) S.history.pop();
    renderSpinView(data);
    document.getElementById('result-input').value = '';
  } catch (e) {
    showError(errEl, e.message);
  } finally {
    setBtn(btn, false, 'Spin →');
  }
}

// ---- Load existing session state ----
async function loadState() {
  try {
    const data = await apiFetch(`/session/${S.sessionId}/state?last_k=10`);
    const spins = (data.recent_k_spins || []).slice().reverse(); // newest first
    S.history = spins.map(s => ({ spin: s.spin_index, result: s.result_number, pnl: s.pnl }));
    const last = data.recent_k_spins?.[data.recent_k_spins.length - 1];
    renderSpinView({
      spin_index:    data.spin_count,
      bankroll_now:  data.bankroll_now,
      pnl_last_spin: last?.pnl ?? null,
      bias_report:   data.last_bias_report,
      next_strategy: last?.bets ?? [],
      rationale:     last?.rationale ?? '',
    });
    showView('spin');
  } catch {
    localStorage.removeItem('sessionId');
    S.sessionId = null;
    showView('new');
  }
}

// ---- Render spin view ----
function renderSpinView({ spin_index, bankroll_now, pnl_last_spin, bias_report, next_strategy, rationale }) {
  // Stats bar
  document.getElementById('bankroll-display').textContent = `$${fmt(bankroll_now)}`;
  document.getElementById('spin-index-display').textContent = spin_index;
  const pnlEl = document.getElementById('pnl-display');
  if (pnl_last_spin === null || pnl_last_spin === undefined) {
    pnlEl.textContent = '—';
    pnlEl.className = 'stat-value';
  } else {
    pnlEl.textContent = `${pnl_last_spin >= 0 ? '+' : ''}$${fmt(pnl_last_spin)}`;
    pnlEl.className = `stat-value ${pnl_last_spin >= 0 ? 'positive' : 'negative'}`;
  }

  // Bias card
  if (bias_report) {
    const verdict = bias_report.verdict || 'no_evidence';
    const cssClass = verdict.replace('_', '-'); // no_evidence → no-evidence
    document.getElementById('bias-card').className = `card bias-card ${cssClass}`;
    document.getElementById('bias-verdict').textContent = verdict.replace('_', ' ').toUpperCase();
    document.getElementById('bias-weight').textContent = `w=${(bias_report.weight || 0).toFixed(2)}`;
    document.getElementById('bias-summary').textContent = bias_report.summary || '';

    const extEl = document.getElementById('bias-external');
    const ext = bias_report.external_check;
    if (ext && ext.status && ext.status !== 'unknown') {
      extEl.textContent = `[外部数据 · 仅供参考] ${ext.status}${ext.note ? ' — ' + ext.note : ''}`;
      extEl.classList.remove('hidden');
    } else {
      extEl.classList.add('hidden');
    }
  }

  // Strategy list
  const list = document.getElementById('strategy-list');
  if (!next_strategy || !next_strategy.length) {
    list.innerHTML = '<p class="no-bets">No bets recommended this spin.</p>';
  } else {
    list.innerHTML = next_strategy.map(bet => {
      const type = bet.type || bet.bet_type || '?';
      const nums = formatNumbers(bet.numbers);
      return `<div class="bet-row">
        <span class="bet-type">${capitalize(type)}</span>
        <span class="bet-numbers">${nums}</span>
        <span class="bet-amount">$${fmt(bet.amount)}</span>
      </div>`;
    }).join('');
  }

  document.getElementById('rationale-text').textContent = rationale || '';

  renderHistory();
}

function renderHistory() {
  const tbody = document.getElementById('history-body');
  if (!S.history.length) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#666;padding:12px">暂无记录</td></tr>';
    return;
  }
  tbody.innerHTML = S.history.map(h => {
    const pnlText = h.pnl !== null && h.pnl !== undefined
      ? `${h.pnl >= 0 ? '+' : ''}$${fmt(h.pnl)}`
      : '—';
    const cls = h.pnl > 0 ? 'positive' : h.pnl < 0 ? 'negative' : '';
    return `<tr>
      <td>#${h.spin}</td>
      <td>${h.result === 37 ? '00' : h.result}</td>
      <td class="${cls}">${pnlText}</td>
    </tr>`;
  }).join('');
}

// ---- API helpers ----
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': S.apiKey },
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

async function apiFetch(path) {
  const res = await fetch(path, { headers: { 'X-API-Key': S.apiKey } });
  return handleResponse(res);
}

async function handleResponse(res) {
  if (res.ok) return res.json();
  let detail = `HTTP ${res.status}`;
  try { const j = await res.json(); detail = j.detail || JSON.stringify(j); } catch {}
  if (res.status === 401) throw new Error('API Key 无效或未设置');
  if (res.status === 404) throw new Error('Session 不存在（已过期？）');
  if (res.status === 422) throw new Error(`请求参数错误: ${detail}`);
  throw new Error(`服务器错误: ${detail}`);
}

// ---- Utility ----
function setResult(n) { document.getElementById('result-input').value = n; }

function showError(el, msg) { el.textContent = msg; el.classList.remove('hidden'); }

function setBtn(btn, disabled, text) { btn.disabled = disabled; btn.textContent = text; }

function fmt(n) {
  if (n === null || n === undefined) return '—';
  return parseFloat(n).toFixed(2);
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function formatNumbers(nums) {
  if (!nums || !nums.length) return '';
  return nums.map(n => n === 37 ? '00' : n).join(', ');
}
