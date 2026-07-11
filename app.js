/* Early Signals — runs the Random Forest entirely in the browser. */

// ---------- model math ----------

function walkTree(t, x) {
  let i = 0;
  while (t.left[i] !== -1) {
    i = x[t.feature[i]] <= t.threshold[i] ? t.left[i] : t.right[i];
  }
  return t.prob[i];
}

function predict(model, row) {
  const x = model.columns.map(c => row[c] ?? 0);
  let sum = 0;
  for (const t of model.trees) sum += walkTree(t, x);
  return sum / model.trees.length;
}

// Build the one-hot encoded row from raw form values.
function encodeRow(meta, values) {
  const row = {};
  for (const f of meta.features) {
    if (f.type === "numeric") {
      row[f.name] = Number(values[f.name]);
    } else {
      for (const o of f.options) {
        row[`${f.name}_${o}`] = values[f.name] === o ? 1 : 0;
      }
    }
  }
  return row;
}

// Allow node-based testing: `node --input-type=module -e "import(...)"`.
if (typeof module !== "undefined") {
  module.exports = { walkTree, predict, encodeRow };
}

// ---------- everything below only runs in the browser ----------

if (typeof document !== "undefined") {

  const LABELS = {
    school:     ["School", "GP = Gabriel Pereira, MS = Mousinho da Silveira"],
    sex:        ["Sex", "F = female, M = male"],
    age:        ["Age", "years"],
    address:    ["Home area", "U = urban, R = rural"],
    famsize:    ["Family size", "LE3 = 3 or fewer, GT3 = more than 3"],
    Pstatus:    ["Parents' cohabitation", "T = together, A = apart"],
    Medu:       ["Mother's education", "0 none … 4 higher education"],
    Fedu:       ["Father's education", "0 none … 4 higher education"],
    Mjob:       ["Mother's job", null],
    Fjob:       ["Father's job", null],
    reason:     ["Reason for choosing school", null],
    guardian:   ["Guardian", null],
    traveltime: ["Travel time to school", "1: <15 min … 4: >1 hour"],
    studytime:  ["Weekly study time", "1: <2 h, 2: 2–5 h, 3: 5–10 h, 4: >10 h"],
    failures:   ["Past class failures", "number of times"],
    schoolsup:  ["Extra school support", null],
    famsup:     ["Family educational support", null],
    paid:       ["Extra paid classes", null],
    activities: ["Extracurricular activities", null],
    nursery:    ["Attended nursery school", null],
    higher:     ["Wants higher education", null],
    internet:   ["Internet access at home", null],
    romantic:   ["In a romantic relationship", null],
    famrel:     ["Family relationship quality", "1 very bad … 5 excellent"],
    freetime:   ["Free time after school", "1 very low … 5 very high"],
    goout:      ["Going out with friends", "1 very low … 5 very high"],
    Dalc:       ["Workday alcohol use", "1 very low … 5 very high"],
    Walc:       ["Weekend alcohol use", "1 very low … 5 very high"],
    health:     ["Current health", "1 very bad … 5 very good"],
    absences:   ["School absences", "number of days"],
  };

  const GROUPS = [
    ["About the student", ["sex", "age", "address", "health", "romantic"]],
    ["Home & family", ["famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
                       "guardian", "famrel", "famsup", "internet"]],
    ["School", ["school", "reason", "traveltime", "studytime", "failures",
                "schoolsup", "paid", "nursery", "higher", "absences"]],
    ["Lifestyle", ["activities", "freetime", "goout", "Dalc", "Walc"]],
  ];

  let MODEL = null, META = null;

  function fieldHTML(f) {
    const [label, help] = LABELS[f.name] || [f.name, null];
    let control;
    if (f.type === "categorical") {
      const opts = f.options.map(o =>
        `<option value="${o}"${o === f.default ? " selected" : ""}>${o}</option>`
      ).join("");
      control = `<select id="f-${f.name}" name="${f.name}">${opts}</select>`;
    } else {
      control = `<input type="number" id="f-${f.name}" name="${f.name}"
        min="${Math.floor(f.min)}" max="${Math.ceil(f.max)}" step="1"
        value="${Math.round(f.default)}">`;
    }
    return `<div class="field">
      <label for="f-${f.name}">${label}</label>${control}
      ${help ? `<span class="hint">${help}</span>` : ""}</div>`;
  }

  function buildForm() {
    const byName = Object.fromEntries(META.features.map(f => [f.name, f]));
    const used = new Set();
    let html = "";
    for (const [title, names] of GROUPS) {
      const fields = names.filter(n => byName[n]);
      if (!fields.length) continue;
      fields.forEach(n => used.add(n));
      html += `<fieldset><legend>${title}</legend><div class="grid">` +
        fields.map(n => fieldHTML(byName[n])).join("") + `</div></fieldset>`;
    }
    const leftovers = META.features.filter(f => !used.has(f.name));
    if (leftovers.length) {
      html += `<fieldset><legend>Other</legend><div class="grid">` +
        leftovers.map(fieldHTML).join("") + `</div></fieldset>`;
    }
    html += `<button type="submit" class="go">Estimate support need</button>`;
    document.getElementById("form").innerHTML = html;
  }

  function showResult(prob) {
    const pct = Math.round(prob * 100);
    const band = prob < 0.30 ? ["low", "Low risk"]
               : prob < 0.60 ? ["moderate", "Moderate risk"]
               : ["high", "High risk"];
    document.getElementById("result").innerHTML = `
      <p class="slip-eyebrow">Screening result</p>
      <p class="verdict ${band[0]}">${band[1]}</p>
      <div class="meter" role="img"
           aria-label="Estimated probability ${pct} percent">
        <div class="meter-fill ${band[0]}" style="width:${pct}%"></div>
      </div>
      <p class="pct"><span>${pct}%</span> estimated chance this student
      finishes below the passing grade (${META.passing_grade}/20)</p>
      <p class="note">This is a screening estimate from a statistical model,
      not a judgment about the student. Use it as one prompt to check in and
      offer support — never as a label. Nothing you enter leaves this page.</p>`;
  }

  async function init() {
    try {
      const [meta, model] = await Promise.all([
        fetch("feature_meta.json").then(r => { if (!r.ok) throw 0; return r.json(); }),
        fetch("model.json").then(r => { if (!r.ok) throw 0; return r.json(); }),
      ]);
      META = meta; MODEL = model;
      buildForm();
      document.getElementById("form").addEventListener("submit", e => {
        e.preventDefault();
        const values = Object.fromEntries(new FormData(e.target));
        showResult(predict(MODEL, encodeRow(META, values)));
      });
    } catch (err) {
      document.getElementById("form").innerHTML =
        `<p class="note">Couldn't load the model files. If you opened this
        file directly from your computer, browsers block it — run
        <code>python -m http.server</code> in this folder and open
        <code>http://localhost:8000</code> instead. On GitHub Pages it
        works as-is.</p>`;
    }
  }

  init();
}
