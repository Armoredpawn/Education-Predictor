# Early Signals — student support screener

A website that estimates which students may be struggling, using a Random
Forest trained on the UCI Student Performance dataset. The model runs
entirely in the visitor's browser — no server, no data sent anywhere.

## Files

- `index.html` — the page
- `app.js` — builds the form and runs the model
- `style.css` — styling
- `model.json` — the trained forest (200 trees)
- `feature_meta.json` — the input form schema

## Put it live on GitHub Pages

1. Create a new repository on github.com (Public), e.g. `student-predictor`.
2. Click **uploading an existing file** (or Add file → Upload files) and
   drag all five files in. Commit.
3. Go to **Settings → Pages**. Under "Branch", pick `main`, folder `/ (root)`,
   and Save.
4. Wait a minute or two. Your site is live at
   `https://YOUR-USERNAME.github.io/student-predictor/`

## Updating the model later

Re-run the Colab notebook, then replace `model.json` and
`feature_meta.json` in the repo. The form and predictions update
automatically — no code changes.

## Testing on your computer (optional)

Browsers block `fetch()` for files opened directly from disk, so double-
clicking `index.html` shows a loading error. Instead run
`python -m http.server` in this folder and open http://localhost:8000

## Use responsibly

Estimates are screening prompts, not judgments. The training data is from
two Portuguese secondary schools (2005–2006) and may not reflect your
students.
