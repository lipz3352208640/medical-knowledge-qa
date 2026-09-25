const form = document.querySelector('#query-form');
const question = document.querySelector('#question');
const drugFilter = document.querySelector('#drug-filter');
const submitButton = document.querySelector('#submit-button');
const emptyState = document.querySelector('#empty-state');
const loadingState = document.querySelector('#loading-state');
const errorState = document.querySelector('#error-state');
const answerArea = document.querySelector('#answer-area');
const sourceDialog = document.querySelector('#source-dialog');

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败（${response.status}）`);
  return data;
}

function showLoading() {
  submitButton.disabled = true;
  emptyState.classList.add('hidden');
  answerArea.classList.add('hidden');
  errorState.classList.add('hidden');
  loadingState.classList.remove('hidden');
}

function showError(message) {
  errorState.textContent = message;
  errorState.classList.remove('hidden');
}

async function openSource(chunkId) {
  const body = document.querySelector('#source-body');
  body.replaceChildren(node('p', 'source-details', '正在加载原文…'));
  sourceDialog.showModal();
  try {
    const source = await api(`/api/v1/evidence/${encodeURIComponent(chunkId)}`);
    const details = node('div', 'source-details');
    const labels = [source.drug_name, source.source_file, source.section, `第 ${source.page_start}${source.page_end !== source.page_start ? `–${source.page_end}` : ''} 页`].filter(Boolean);
    details.textContent = labels.join(' · ');
    body.replaceChildren(details, node('div', 'source-raw', source.content), node('p', 'source-id', `chunk ID: ${source.chunk_id} · parent ID: ${source.parent_id || '无'}`));
  } catch (error) {
    body.replaceChildren(node('p', 'error-state', error.message));
  }
}

function renderAnswer(data, elapsed) {
  const confidence = document.querySelector('#confidence');
  confidence.className = `confidence ${data.confidence || 'low'}`;
  confidence.textContent = {high: '高可信度', medium: '需核对', low: '证据不足'}[data.confidence] || '未评级';
  document.querySelector('#elapsed').textContent = `${(elapsed / 1000).toFixed(1)} 秒`;
  document.querySelector('#answer-text').textContent = data.answer || '';
  const warnings = document.querySelector('#warning-area');
  warnings.replaceChildren(...(data.warnings || []).map(item => node('p', '', item)));
  warnings.classList.toggle('hidden', !data.warnings?.length);
  const evidence = document.querySelector('#evidence-list');
  evidence.replaceChildren();
  const rows = data.evidence || [];
  document.querySelector('#evidence-count').textContent = `(${rows.length})`;
  if (!rows.length) evidence.append(node('p', 'no-evidence', '当前回答没有可展示的原文引用。'));
  for (const item of rows) {
    const card = node('button', 'evidence-card');
    card.type = 'button';
    card.append(node('strong', '', item.drug_name || '说明书'));
    card.append(node('span', 'source-meta', `${item.source_file || '来源文件'} · ${item.section || '章节未标注'} · 第 ${item.page}${item.page_end ? `–${item.page_end}` : ''} 页`));
    card.append(node('blockquote', '', item.quote || ''));
    card.addEventListener('click', () => openSource(item.chunk_id));
    evidence.append(card);
  }
  const debugPanel = document.querySelector('#debug-panel');
  const debugData = {
    query_rewrite: data.query_rewrite || null,
    compression: data.compression || [],
  };
  debugPanel.classList.toggle('hidden', !data.compression?.length && !data.query_rewrite);
  document.querySelector('#debug-text').textContent = JSON.stringify(debugData, null, 2);
  answerArea.classList.remove('hidden');
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const query = question.value.trim();
  if (query.length < 2) return;
  const drugNames = drugFilter.value.split(/[，,、]/).map(value => value.trim()).filter(Boolean);
  showLoading();
  const started = performance.now();
  try {
    const result = await api('/api/v1/chat/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, filters: {drug_names: drugNames, document_versions: ['current']}, debug: true}),
    });
    renderAnswer(result, performance.now() - started);
  } catch (error) {
    showError(error.message);
  } finally {
    loadingState.classList.add('hidden');
    submitButton.disabled = false;
  }
});

document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  question.value = button.dataset.question;
  question.focus();
}));
document.querySelector('#dialog-close').addEventListener('click', () => sourceDialog.close());
sourceDialog.addEventListener('click', event => { if (event.target === sourceDialog) sourceDialog.close(); });

api('/api/v1/health').then(() => {
  document.querySelector('.status-dot').classList.add('ready');
  document.querySelector('#service-status').textContent = '页面服务已连接';
}).catch(() => { document.querySelector('#service-status').textContent = '页面服务不可用'; });
