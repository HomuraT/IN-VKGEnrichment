let currentFile = null;
let currentIndex = 0;
let total = 0;
let currentVersion = null; // File version (backend md5)

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || (res.status + ''));
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

async function idExists(id) {
  if (!currentFile) return false;
  try {
    await api(`/api/samples/by-id?filename=${encodeURIComponent(currentFile)}&sample_id=${encodeURIComponent(id)}`);
    return true;
  } catch (_) {
    return false;
  }
}

async function generateExampleId() {
  for (let i = 1; i < 1000000; i++) {
    const candidate = `example-${i}`;
    // 404 -> does not exist, available
    // other exceptions are propagated to the caller
    const exists = await idExists(candidate);
    if (!exists) return candidate;
  }
  throw new Error('Unable to generate available example-* ID');
}

async function loadFiles() {
  const data = await api('/api/files');
  const sel = document.getElementById('fileSelect');
  sel.innerHTML = '';
  data.files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f; sel.appendChild(opt);
  });
  if (data.files.length) {
    currentFile = data.files[0];
    sel.value = currentFile;
    try {
      await loadSample(0);
    } catch (e) {
      // Target file is empty or read failed, reset to empty state
      total = 0; currentIndex = 0; currentVersion = null;
      updatePosition();
      clearForm();
    }
  } else {
    currentFile = null; currentIndex = 0; total = 0;
    updatePosition();
  clearForm();
  }
}

function updatePosition() {
  document.getElementById('position').textContent = `${total ? (currentIndex + 1) : 0} / ${total}`;
}

async function loadSample(idx) {
  if (!currentFile) return;
  const data = await api(`/api/samples?filename=${encodeURIComponent(currentFile)}&index=${idx}`);
  currentIndex = data.index; total = data.total;
  currentVersion = data.version || null;
  updatePosition();
  fillForm(data.sample);
}

async function loadSettings() {
  // Read all from browser localStorage
  document.getElementById('annotator').value = (localStorage.getItem('dataset_annotator') || '');
  document.getElementById('endpoint').value = (localStorage.getItem('dataset_endpoint_url') || '');
}

async function saveSettings() {
  const annotator = document.getElementById('annotator').value.trim();
  localStorage.setItem('dataset_annotator', annotator);
  const endpoint = document.getElementById('endpoint').value.trim();
  localStorage.setItem('dataset_endpoint_url', endpoint);
  document.getElementById('settingsInfo').textContent = `Saved locally: ${endpoint || '(empty)'}`;
}

async function createFile() {
  const name = document.getElementById('newFileName').value.trim();
  if (!name) { alert('Please enter a new filename, e.g. new_file.jsonl'); return; }
  try {
    await api(`/api/files?filename=${encodeURIComponent(name)}`, { method: 'POST' });
    await loadFiles();
    alert('File created successfully');
  } catch (e) {
    alert(parseErr(e));
  }
}

async function saveSample(update = true) {
  if (!currentFile) { alert('No file selected'); return; }
  // Basic validation
  const annotator = document.getElementById('annotator').value.trim();
  if (!annotator) { alert('Please fill in your name in "Global Settings" first and click "Save Settings"'); return; }
  let obj;
  try {
    obj = collectForm();
  } catch (e) {
    alert(parseErr(e));
    return;
  }
  obj.annotator = annotator; // Store in sample, backend validates non-empty
  const missing = [];
  if (!obj.id) missing.push('id');
  if (!obj.vkg) missing.push('vkg');
  if (!obj.sample_type) missing.push('sample_type');
  if (!obj.question) missing.push('question');
  if (!obj.sparql) missing.push('sparql');
  if (missing.length) { alert('Please complete required fields: ' + missing.join(', ')); return; }

  // When creating new, if ID already exists, append _2
  if (!update) {
    try {
      const exists = await idExists(obj.id);
      if (exists) obj.id = `${obj.id}_2`;
    } catch (_) {}
  }
  const url = `/api/samples?filename=${encodeURIComponent(currentFile)}${update ? `&index=${currentIndex}` : ''}`;
  const method = update ? 'PUT' : 'POST';
  try {
    const payload = { ...obj, expected_version: currentVersion };
    const res = await api(url, { method, body: JSON.stringify(payload) });
    currentVersion = res.version || currentVersion;
    if (!update) {
      await loadSample(res.index);
    } else {
      await loadSample(currentIndex);
    }
    alert('Saved successfully');
  } catch (e) {
    alert(parseErr(e));
  }
}

async function deleteSample() {
  if (!currentFile) return;
  if (!total) { alert('Current file is empty'); return; }
  try {
    const url = currentVersion
      ? `/api/samples?filename=${encodeURIComponent(currentFile)}&index=${currentIndex}&expected_version=${encodeURIComponent(currentVersion)}`
      : `/api/samples?filename=${encodeURIComponent(currentFile)}&index=${currentIndex}`;
    await api(url, { method: 'DELETE' });
    const idx = Math.max(0, currentIndex - 1);
    try {
      await loadSample(idx);
    } catch {
      // empty file now
      total = 0; currentIndex = 0; updatePosition();
      clearForm();
    }
  } catch (e) {
    alert(parseErr(e));
  }
}

async function goPrev() {
  if (currentIndex > 0) await loadSample(currentIndex - 1);
}
async function goNext() {
  if (currentIndex + 1 < total) await loadSample(currentIndex + 1);
}

async function goById() {
  const id = document.getElementById('searchId').value.trim();
  if (!id) return;
  try {
    const data = await api(`/api/samples/by-id?filename=${encodeURIComponent(currentFile)}&sample_id=${encodeURIComponent(id)}`);
    await loadSample(data.index);
  } catch (e) {
    alert(parseErr(e));
  }
}

async function runSparql() {
  const sparql = document.getElementById('field_sparql').value;
  if (!sparql || typeof sparql !== 'string') { alert('SPARQL query is required'); return; }
  try {
    const endpoint = (localStorage.getItem('dataset_endpoint_url') || '').trim();
    const res = await api('/api/run-sparql', { method: 'POST', body: JSON.stringify({ sparql, endpoint_url: endpoint }) });
    const out = document.getElementById('result');
    if (res.type === 'table') {
      out.textContent = JSON.stringify(res.data, null, 2);
    } else {
      out.textContent = res.data;
    }
  } catch (e) {
    alert(parseErr(e));
  }
}

// -------- Import/Export --------
async function importJsonSample() {
  if (!currentFile) { alert('Please select a dataset file first'); return; }
  // Validate annotator (required by backend for saving)
  const annotator = document.getElementById('annotator').value.trim();
  if (!annotator) { alert('Please fill in your name in "Global Settings" first and click "Save Settings"'); return; }
  const txt = document.getElementById('import_json').value.trim();
  if (!txt) { alert('Please enter JSON'); return; }
  let obj;
  try {
    obj = JSON.parse(txt);
  } catch (e) {
    alert('JSON parsing failed');
    return;
  }
  if (typeof obj !== 'object' || Array.isArray(obj) || obj === null) {
    alert('JSON must be an object');
    return;
  }
  // Extract valid structure only
  const minimal = {
    vkg: String(obj.vkg || '').trim(),
    sample_type: String(obj.sample_type || 'ONTOLOGY_ONLY').trim(),
    question: String(obj.question || '').toString(),
    sparql: String(obj.sparql || '').toString()
  };
  if (Array.isArray(obj.references)) {
    minimal.references = obj.references.map(r => {
      const item = {
        type: String(r.type || 'text'),
        purpose: String(r.purpose || ''),
        content: String(r.content || '')
      };
      if (Array.isArray(r.triples)) item.triples = r.triples;
      return item;
    });
  }
  if (!minimal.vkg || !minimal.sample_type || !minimal.question || !minimal.sparql) {
    alert('Missing required fields: vkg / sample_type / question / sparql');
    return;
  }

  // Auto-generate example-* ID; if conflict in extreme cases, append _2
  try {
    minimal.id = await generateExampleId();
  } catch (e) {
    alert(parseErr(e));
    return;
  }
  try {
    const exists = await idExists(minimal.id);
    if (exists) minimal.id = `${minimal.id}_2`;
  } catch (_) {}

  // Submit new sample
  const payload = { ...minimal, annotator, expected_version: currentVersion };
  try {
    const res = await api(`/api/samples?filename=${encodeURIComponent(currentFile)}`, { method: 'POST', body: JSON.stringify(payload) });
    currentVersion = res.version || currentVersion;
    await loadSample(res.index);
    alert('Imported and appended as new sample');
  } catch (e) {
    alert(parseErr(e));
  }
}

function exportMinimalJson() {
  // Collect from current form and trim to minimal structure
  let obj;
  try {
    obj = collectForm();
  } catch (e) {
    alert(parseErr(e));
    return;
  }
  const minimal = {
    id: obj.id,
    vkg: obj.vkg,
    sample_type: obj.sample_type,
    question: obj.question,
    sparql: obj.sparql
  };
  if (Array.isArray(obj.references) && obj.references.length) {
    minimal.references = obj.references;
  }
  const text = JSON.stringify(minimal, null, 2);
  const out = document.getElementById('result');
  // Prefer async Clipboard API (may not be available in http/insecure contexts or remote access)
  (async () => {
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(text);
        alert('Copied as minimal JSON');
        return;
      }
    } catch (_) {
      // Ignore, try fallback
    }
    // Fallback: use hidden textarea + execCommand('copy')
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (ok) {
        alert('Copied as minimal JSON');
        return;
      }
    } catch (_) {
      // Ignore, enter final fallback
    }
    // Final fallback: display in result area
    if (out) out.textContent = text;
    alert('Cannot access clipboard, displayed in result area below');
  })();
}

document.getElementById('refreshFiles').addEventListener('click', loadFiles);
document.getElementById('createFile').addEventListener('click', createFile);
document.getElementById('saveSettings').addEventListener('click', saveSettings);
document.getElementById('fileSelect').addEventListener('change', async (e) => {
  currentFile = e.target.value;
  try {
    await loadSample(0);
  } catch (e2) {
    // Switching to empty file, clear form and reset position
    total = 0; currentIndex = 0; currentVersion = null;
    updatePosition();
    clearForm();
  }
});
document.getElementById('prev').addEventListener('click', goPrev);
document.getElementById('next').addEventListener('click', goNext);
document.getElementById('goId').addEventListener('click', goById);
document.getElementById('saveSample').addEventListener('click', () => saveSample(true));
document.getElementById('newSample').addEventListener('click', () => saveSample(false));
document.getElementById('deleteSample').addEventListener('click', deleteSample);
document.getElementById('runSparql').addEventListener('click', runSparql);
document.getElementById('importJsonBtn').addEventListener('click', importJsonSample);
document.getElementById('exportMinimalBtn').addEventListener('click', exportMinimalJson);
document.getElementById('addRef').addEventListener('click', () => addRef());

(async function init(){
  await loadSettings();
  await loadFiles();
  // Version polling: check every 5 seconds, prompt refresh if changed
  setInterval(async () => {
    if (!currentFile) return;
    try {
      const v = await api(`/api/version?filename=${encodeURIComponent(currentFile)}`);
      if (currentVersion && v.version && v.version !== currentVersion) {
        if (confirm('The current file has been updated by another session. Refresh now?')) {
          await loadSample(currentIndex);
        } else {
          // Mark version to ensure next save conflicts, prompting user to refresh first
          currentVersion = v.version;
        }
      }
    } catch {}
  }, 5000);
})();

// ---------- Form Logic ----------
function clearForm() {
  document.getElementById('field_id').value = '';
  document.getElementById('field_vkg').value = '';
  document.getElementById('field_type').value = 'ONTOLOGY_ONLY';
  document.getElementById('field_question').value = '';
  document.getElementById('field_sparql').value = '';
  const refs = document.getElementById('refs_container');
  if (refs) refs.innerHTML = '';
  const out = document.getElementById('result');
  if (out) out.textContent = '';
}

function fillForm(sample) {
  clearForm();
  document.getElementById('field_id').value = sample.id || '';
  document.getElementById('field_vkg').value = sample.vkg || '';
  document.getElementById('field_type').value = sample.sample_type || 'ONTOLOGY_ONLY';
  document.getElementById('field_question').value = sample.question || '';
  document.getElementById('field_sparql').value = sample.sparql || '';
  if (Array.isArray(sample.references)) {
    sample.references.forEach(r => addRef(r));
  }
}

function collectForm() {
  const id = document.getElementById('field_id').value.trim();
  const vkg = document.getElementById('field_vkg').value.trim();
  const sample_type = document.getElementById('field_type').value;
  const question = document.getElementById('field_question').value;
  const sparql = document.getElementById('field_sparql').value;
  const res = { id, vkg, question, sample_type, sparql };
  const refs = readReferencesFromUI();
  if (refs.length) res.references = refs;
  return res;
}

// ---- References panel logic ----
function buildRefItem(ref) {
  const tpl = document.getElementById('ref_item_template');
  const node = tpl.content.firstElementChild.cloneNode(true);
  const typeEl = node.querySelector('.ref-type');
  const purposeEl = node.querySelector('.ref-purpose');
  const contentEl = node.querySelector('.ref-content');
  const triplesEl = node.querySelector('.ref-triples');
  const removeBtn = node.querySelector('.ref-remove');
  if (ref) {
    if (ref.type) typeEl.value = String(ref.type);
    if (ref.purpose) purposeEl.value = String(ref.purpose);
    if (ref.content) contentEl.value = String(ref.content);
    if (Array.isArray(ref.triples)) {
      const nt = ref.triples.map(t => {
        if (Array.isArray(t) && t.length >= 3) {
          return `${t[0]} ${t[1]} ${t[2]} .`;
        }
        return '';
      }).filter(Boolean).join('\n');
      triplesEl.value = nt;
    }
  }
  removeBtn.addEventListener('click', () => {
    node.remove();
  });
  return node;
}

function addRef(prefill) {
  const container = document.getElementById('refs_container');
  container.appendChild(buildRefItem(prefill));
}

function parseNtToTriples(ntText) {
  const lines = ntText.split('\n').map(l => l.trim()).filter(Boolean);
  const triples = [];
  for (const line of lines) {
    const noDot = line.replace(/\s*\.$/, '');
    const m = noDot.match(/^(.*?)\s+(.*?)\s+([\s\S]+)$/);
    if (m) {
      const s = m[1];
      const p = m[2];
      const o = m[3];
      triples.push([s, p, o]);
    }
  }
  return triples;
}

function readReferencesFromUI() {
  const container = document.getElementById('refs_container');
  const items = Array.from(container.querySelectorAll('.ref-item'));
  const refs = [];
  for (const item of items) {
    const type = item.querySelector('.ref-type').value;
    const purpose = item.querySelector('.ref-purpose').value.trim();
    const content = item.querySelector('.ref-content').value;
    const nt = item.querySelector('.ref-triples').value.trim();
    const ref = { type, purpose, content };
    if (nt) {
      const triples = parseNtToTriples(nt);
      if (triples.length) ref.triples = triples;
    }
    if (ref.content || ref.triples) refs.push(ref);
  }
  return refs;
}

function parseErr(e) {
  const s = String(e && e.message ? e.message : e);
  try {
    const j = JSON.parse(s);
    if (j && j.detail) return j.detail;
  } catch {}
  return s;
}
