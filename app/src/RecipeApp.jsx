import { useEffect, useMemo, useRef, useState } from 'react'
import Papa from 'papaparse'
import { pipeline } from '@huggingface/transformers'
import './RecipeApp.css'

const DATA_URL = '/recipes_200000.csv'
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2'
const STOP_WORDS = new Set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'have', 'i', 'in', 'is', 'it', 'me', 'my', 'of', 'on', 'or', 'some', 'the', 'to', 'with', 'without'])
const INGREDIENT_ALIASES = new Map([
  ['breast', 'chicken'], ['thigh', 'chicken'], ['poultry', 'chicken'],
  ['tomatoes', 'tomato'], ['roma', 'tomato'], ['bell', 'pepper'],
  ['salts', 'salt'], ['sea', 'salt'], ['kosher', 'salt'], ['sodium', 'salt'],
  ['scallion', 'onion'], ['scallions', 'onion'], ['shallot', 'onion'],
  ['coriander', 'cilantro'], ['margarine', 'butter'], ['lime', 'lemon'],
  ['garbanzo', 'chickpea'], ['garbanzos', 'chickpea'], ['aubergine', 'eggplant'],
])

let embeddingPipeline

function loadEmbeddingModel() {
  if (!embeddingPipeline) embeddingPipeline = pipeline('feature-extraction', EMBEDDING_MODEL, { dtype: 'q8' })
  return embeddingPipeline
}

function tokenize(value) {
  return [...new Set(String(value || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).map((word) => word.endsWith('s') && word.length > 3 ? word.slice(0, -1) : word).filter((word) => word.length > 2 && !STOP_WORDS.has(word)))]
}

function canonicalIngredient(term) {
  return INGREDIENT_ALIASES.get(term) || term
}

function ingredientTerms(value) {
  return new Set(tokenize(value).map(canonicalIngredient))
}

function cosineSimilarity(first, second) {
  let dot = 0
  let firstNorm = 0
  let secondNorm = 0
  for (let index = 0; index < first.length; index += 1) {
    dot += first[index] * second[index]
    firstNorm += first[index] ** 2
    secondNorm += second[index] ** 2
  }
  return firstNorm && secondNorm ? dot / Math.sqrt(firstNorm * secondNorm) : 0
}

async function embed(text, model) {
  const output = await model(text, { pooling: 'mean', normalize: true })
  return output.data
}

function parseList(value) {
  if (!value) return []
  const matches = String(value).match(/["']([^"']+)["']/g)
  if (matches) return matches.map((item) => item.slice(1, -1))
  return String(value).split(',').map((item) => item.trim()).filter(Boolean)
}

function RecipeApp() {
  const [recipes, setRecipes] = useState([])
  const [query, setQuery] = useState('')
  const [source, setSource] = useState('All sources')
  const [selectedRecipe, setSelectedRecipe] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [aiPrompt, setAiPrompt] = useState('')
  const [aiAnswer, setAiAnswer] = useState('')
  const [semanticResults, setSemanticResults] = useState([])
  const [modelStatus, setModelStatus] = useState('ready')
  const embeddingCache = useRef(new Map())

  useEffect(() => {
    Papa.parse(DATA_URL, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: ({ data }) => {
        setRecipes(data.filter((recipe) => recipe.title))
        setLoading(false)
      },
      error: () => {
        setError('The recipe file could not be loaded. Check that it exists in public/.')
        setLoading(false)
      },
    })
  }, [])

  const sources = useMemo(() => {
    const counts = recipes.reduce((all, recipe) => {
      const name = recipe.source?.trim()
      if (name) all[name] = (all[name] || 0) + 1
      return all
    }, {})
    return Object.entries(counts).sort(([, first], [, second]) => second - first).slice(0, 12).map(([name]) => name)
  }, [recipes])

  const localModel = useMemo(() => {
    const documentFrequency = new Map()
    const recipeTerms = recipes.map((recipe) => {
      const terms = tokenize(`${recipe.title} ${recipe.NER || recipe.ingredients}`)
      terms.forEach((term) => documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1))
      return terms
    })
    const idf = new Map([...documentFrequency].map(([term, frequency]) => [term, Math.log((recipes.length + 1) / (frequency + 1)) + 1]))
    return { idf, recipeTerms, vocabularySize: idf.size }
  }, [recipes])

  const lexicalResults = useMemo(() => {
    const queryTerms = ingredientTerms(query)
    if (!queryTerms.length) return recipes.filter((recipe) => source === 'All sources' || recipe.source === source).slice(0, 60)
    const forbiddenTerms = [...query.toLowerCase().matchAll(/\b(?:no|without)\s+([a-z]+)/g)].map((match) => match[1].endsWith('s') ? match[1].slice(0, -1) : match[1])
    const queryWeights = new Map(queryTerms.map((term) => [term, localModel.idf.get(term) || 0]))
    const queryNorm = Math.sqrt([...queryWeights.values()].reduce((sum, weight) => sum + weight ** 2, 0))
    return recipes.map((recipe, index) => {
      const weights = localModel.recipeTerms[index].map((term) => [term, localModel.idf.get(term) || 0])
      const weightMap = new Map(weights)
      const dot = queryTerms.reduce((sum, term) => sum + (queryWeights.get(term) || 0) * (weightMap.get(term) || 0), 0)
      const documentNorm = Math.sqrt([...weightMap.values()].reduce((sum, weight) => sum + weight ** 2, 0))
      const recipeWords = ingredientTerms(`${recipe.title} ${recipe.NER || recipe.ingredients}`)
      const matchedTerms = [...queryTerms].filter((term) => recipeWords.has(term))
      const coverage = matchedTerms.length / queryTerms.size
      return { recipe, score: queryNorm && documentNorm ? dot / (queryNorm * documentNorm) : 0, matchedTerms, coverage }
    }).filter(({ recipe, score }) => {
      const recipeWords = tokenize(`${recipe.title} ${recipe.NER || recipe.ingredients}`)
      const violatesConstraint = forbiddenTerms.some((term) => recipeWords.includes(term))
      return (source === 'All sources' || recipe.source === source) && score > 0 && !violatesConstraint
    }).sort((first, second) => second.score - first.score).slice(0, 120)
  }, [recipes, query, source, localModel])

  const results = semanticResults.length && semanticResults[0].query === query ? semanticResults : lexicalResults
  const featured = results.slice(0, 6)

  function handleQueryChange(event) {
    setQuery(event.target.value)
    setSemanticResults([])
  }

  async function askAssistant(event) {
    event.preventDefault()
    if (!aiPrompt.trim()) return
    const prompt = aiPrompt.trim()
    setQuery(prompt)
    setModelStatus('loading')
    setAiAnswer('Loading the pretrained MiniLM embedding model and comparing ingredient meaning...')
    try {
      const model = await loadEmbeddingModel()
      const queryEmbedding = await embed(prompt, model)
      const promptTerms = ingredientTerms(prompt)
      const candidates = query === prompt ? lexicalResults : recipes.filter((recipe) => source === 'All sources' || recipe.source === source).map((recipe) => {
        const recipeTerms = ingredientTerms(`${recipe.title} ${recipe.NER || recipe.ingredients}`)
        const matchedTerms = [...promptTerms].filter((term) => recipeTerms.has(term))
        return { recipe, matchedTerms, coverage: promptTerms.size ? matchedTerms.length / promptTerms.size : 0 }
      }).sort((first, second) => second.coverage - first.coverage).slice(0, 120)
      const ranked = await Promise.all(candidates.slice(0, 80).map(async (candidate) => {
        const cacheKey = candidate.recipe.title
        let recipeEmbedding = embeddingCache.current.get(cacheKey)
        if (!recipeEmbedding) {
          recipeEmbedding = await embed(`${candidate.recipe.title}. Ingredients: ${candidate.recipe.NER || candidate.recipe.ingredients}`, model)
          embeddingCache.current.set(cacheKey, recipeEmbedding)
        }
        const semanticScore = cosineSimilarity(queryEmbedding, recipeEmbedding)
        return { ...candidate, query: prompt, semanticScore, score: candidate.coverage * .45 + semanticScore * .55 }
      }))
      setSemanticResults(ranked.sort((first, second) => second.score - first.score))
      setModelStatus('ready')
      setAiAnswer(`MiniLM compared the meaning of your request with ${ranked.length} candidate recipes. Results combine semantic similarity, ingredient coverage, and substitution-aware matching.`)
    } catch {
      setModelStatus('error')
      setSemanticResults([])
      setAiAnswer('The semantic model could not load. Keyword and ingredient-coverage recommendations are still available.')
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Recipe Finder home"><span>RF</span> Recipe Finder</a>
        <div className="dataset-status"><span className="status-dot" /> {modelStatus === 'loading' ? 'Loading semantic ML' : 'RecipeNLG + MiniLM'} <strong>{recipes.length.toLocaleString()}</strong> recipes</div>
      </header>

      <section className="intro">
        <p className="eyebrow">YOUR NEXT GOOD MEAL</p>
        <h1>Find the recipe<br /><em>worth making.</em></h1>
        <p className="lede">Search 200,000 RecipeNLG recipes by what you have, what you crave, or the dish you cannot stop thinking about.</p>
        <div className="search-row">
          <label className="search-box"><span aria-hidden="true">⌕</span><input value={query} onChange={handleQueryChange} placeholder="Try chicken, pasta, or garlic" aria-label="Search recipes" />{query && <button type="button" className="clear-button" onClick={() => { setQuery(''); setSemanticResults([]) }} aria-label="Clear search">×</button>}</label>
          <select value={source} onChange={(event) => setSource(event.target.value)} aria-label="Filter by source"><option>All sources</option>{sources.map((name) => <option key={name}>{name}</option>)}</select>
        </div>
      </section>

      <section className="assistant-section">
        <div className="assistant-heading"><p className="eyebrow">BROWSER-LOCAL AI</p><h2>Tell it what you have.</h2><p>A pretrained MiniLM embedding model understands similar ingredients. Coverage scoring rewards recipes that match more of what you selected.</p></div>
        <form className="assistant-form" onSubmit={askAssistant}><textarea value={aiPrompt} onChange={(event) => setAiPrompt(event.target.value)} placeholder="I have eggs, tomato, onion, and cheese. I don't have an oven." aria-label="Ask the local recipe model" /><button type="submit" disabled={!aiPrompt.trim()}>Match ingredients <span aria-hidden="true">↗</span></button></form>
        {aiAnswer && <div className="assistant-answer"><span className="answer-mark">AI</span><p>{aiAnswer}</p></div>}
      </section>

      <section className="results-section" aria-live="polite">
        <div className="section-heading"><div><p className="eyebrow">{semanticResults.length && semanticResults[0].query === query ? 'SEMANTIC ML RESULTS' : query ? 'SEARCH RESULTS' : 'A LITTLE INSPIRATION'}</p><h2>{query ? `${results.length}${results.length === 60 ? '+' : ''} recipes found` : 'Start with something delicious'}</h2></div>{!loading && <span className="result-count">Showing {featured.length} of {results.length}</span>}</div>
        {loading && <div className="loading-state"><span className="loader" /> Reading your recipe collection...</div>}
        {error && <p className="error-state">{error}</p>}
        {!loading && !error && featured.length === 0 && <div className="empty-state"><strong>No recipes matched that search.</strong><span>Try a broader ingredient or clear the filters.</span></div>}
        <div className="recipe-grid">{featured.map((recipe, index) => <RecipeCard key={`${recipe.title}-${index}`} recipe={recipe} onSelect={setSelectedRecipe} />)}</div>
      </section>

      {selectedRecipe && <RecipeDetail recipe={selectedRecipe} onClose={() => setSelectedRecipe(null)} />}
    </main>
  )
}

function RecipeCard({ recipe: result, onSelect }) {
  const recipe = result.recipe || result
  const ingredients = parseList(recipe.NER || recipe.ingredients)
  return <article className="recipe-card" onClick={() => onSelect(recipe)}><div className="card-top"><span>{String(ingredients.length).padStart(2, '0')} INGREDIENTS</span><span aria-hidden="true">↗</span></div><h3>{recipe.title}</h3><p className="ingredient-preview">{ingredients.slice(0, 4).join(' · ') || 'Open for ingredients and directions'}</p>{result.coverage > 0 && <p className="match-score">{Math.round(result.coverage * 100)}% ingredient match{result.matchedTerms?.length ? ` · ${result.matchedTerms.join(', ')}` : ''}</p>}<button type="button" onClick={() => onSelect(recipe)}>View recipe <span aria-hidden="true">→</span></button></article>
}

function RecipeDetail({ recipe, onClose }) {
  const ingredients = parseList(recipe.ingredients || recipe.NER)
  const directions = parseList(recipe.directions)
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="detail-panel" role="dialog" aria-modal="true" aria-label={recipe.title}><button type="button" className="close-button" onClick={onClose} aria-label="Close recipe">×</button><p className="eyebrow">{recipe.source || 'RECIPE NLG'}</p><h2>{recipe.title}</h2><div className="detail-columns"><div><h4>Ingredients</h4><ul>{ingredients.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul></div><div><h4>Directions</h4><ol>{directions.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}</ol></div></div></section></div>
}

export default RecipeApp