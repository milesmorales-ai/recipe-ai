# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:


## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.

# Recipe Finder

A React recipe search app backed by a 200,000-row RecipeNLG CSV downloaded from Hugging Face.

## Start the app

From `recipe-ai/app`:

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173/`.

## Database deployment

The deployed app reads recipes through a Netlify Function backed by Supabase. The CSV is only an import source and should not be committed to Git or deployed with the frontend.

1. Create a Supabase project and run [supabase/schema.sql](supabase/schema.sql) in its SQL Editor.
2. From the `app` directory, set the project URL and service-role key in PowerShell. Keep the service-role key private.

```powershell
$env:SUPABASE_URL = 'https://your-project.supabase.co'
$env:SUPABASE_SERVICE_ROLE_KEY = 'your-service-role-key'
node scripts/import-recipes.mjs
```

3. In Netlify, set the same two environment variables under Site configuration > Environment variables.
4. Deploy with base directory `app`, build command `npm run build`, and publish directory `dist`.

The Function uses the service-role key server-side. It is never included in the React bundle.

## Use it

1. Wait for the recipe count in the top-right to change to `200,000`.
2. Search for a title or ingredient such as `chicken`, `pasta`, or `garlic`.
3. Optionally choose a source from the filter.
4. Select a recipe card to view its ingredients and directions.
5. Close the detail view with the `X` button or by clicking outside the panel.

To refresh the data, run `python download_dataset.py` from the project root and re-run the Supabase import. The app no longer downloads the CSV directly in the browser.

## Browser-local AI model

The `Tell it what you have` box runs a lightweight machine-learning ranker directly in the browser. When the CSV loads, the app fits an ingredient vocabulary and inverse-document-frequency weights from the recipe collection. Each request is converted into a vector and compared with recipe vectors using cosine similarity.

This is free and unlimited because there is no external API, account, model download, or server call. Users download the app and dataset through the normal website load; all matching happens on their device. Results are always exact recipes from the CSV, so the ranker cannot invent a recipe.

The recommendation flow now has two ML layers. Normal searches use an explainable TF-IDF ranker trained from the loaded collection. The ingredient assistant additionally runs the pretrained `Xenova/all-MiniLM-L6-v2` sentence-embedding model in the browser, compares query and recipe vectors with cosine similarity, and blends that score with substitution-aware ingredient coverage. It can therefore treat terms such as `sea salt`/`salt`, `Roma tomatoes`/`tomato`, and `chicken breast`/`chicken` as related instead of requiring exact phrases. No recipe text is sent to a server.

The assistant embeds the top lexical candidates rather than all 200,000 recipes, keeping the browser workload practical. Recipe cards show the matched ingredient coverage used by the ranker.

## Ask Qwen about a recipe

Open a recipe, select `Ask AI about this recipe`, and ask a question about its ingredients or directions. The popup calls Ollama at `http://127.0.0.1:11434` using `qwen2.5:0.5b`.

Install Ollama from [ollama.com/download](https://ollama.com/download), then run:

```powershell
ollama pull qwen2.5:0.5b
ollama serve
```

The app reports an explicit error if Ollama is not running or the model is unavailable. It does not generate a fake answer or silently substitute another model.

The same recipe popup also has `Ask browser Qwen 2.5 0.5B`. That option runs the public ONNX-native `onnx-community/Qwen2.5-0.5B-Instruct` checkpoint inside `src/recipeAi.worker.js` with Transformers.js. The originally suggested `Xenova/LaMini-Flan-T5-78M` repository currently requires Hugging Face authorization, and the earlier Xenova Qwen repository did not contain the required ONNX file, so the app uses this verified ONNX-native model instead. The worker keeps generation off the React main thread and uses WebAssembly/CPU inference, so it does not require Ollama, WebGPU, or a dedicated GPU. The model is downloaded from Hugging Face on first use and cached by the browser. It is free for the site owner because every visitor downloads and runs their own copy, but it uses more memory and bandwidth than the previous micro-model.

The local vocabulary also normalizes common food language: `sodium`, `salty`, and `savory` map to `salt`; plural ingredient names are normalized; and modifiers such as `heavy`, `extra`, `lots`, `light`, and `less` change the ranking toward recipes with stronger or lighter quantity context in the ingredient text.

## Test the app

```powershell
npm run lint
npm run build
```

For a manual smoke test, verify that the initial six cards load, searching `chicken` changes the results, and selecting a card opens the ingredient and directions panel.
