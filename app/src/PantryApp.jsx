import { useEffect, useMemo, useRef, useState } from 'react'
import { pipeline } from '@huggingface/transformers'
import './PantryApp.css'

const RECIPES_API = '/api/recipes'
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2'
const STORAGE_KEY = 'recipe-finder-pantry'
const STOP_WORDS = new Set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'have', 'i', 'in', 'is', 'it', 'me', 'my', 'of', 'on', 'or', 'some', 'the', 'to', 'with', 'without', 'want', 'food', 'meal', 'dish', 'recipe', 'recipes', 'give', 'please', 'something', 'using', 'uses', 'involved', 'amount', 'need', 'make', 'making', 'cook', 'cooking', 'dinner', 'lunch', 'breakfast', 'snack', 'hungry', 'tonight', 'today', 'can', 'could', 'would', 'should', 'looking', 'find', 'option', 'options', 'kind', 'type', 'what'])
const FOOD_SYNONYMS = {
  sodium: 'salt', salty: 'salt', seasoned: 'salt', savoury: 'salt', savory: 'salt', seasoning: 'salt',
  sea: 'salt', kosher: 'salt', roma: 'tomato', breast: 'chicken', breasts: 'chicken', thigh: 'chicken', thighs: 'chicken',
  tomatoe: 'tomato', tomatoes: 'tomato', chickens: 'chicken', peppers: 'pepper', potatoes: 'potato', onions: 'onion', carrots: 'carrot', mushrooms: 'mushroom', beans: 'bean', peas: 'pea', apples: 'apple', bananas: 'banana', lemons: 'lemon', limes: 'lime', berries: 'berry', strawberries: 'strawberry',
  poultry: 'chicken', hen: 'chicken', fowl: 'chicken', beefsteak: 'beef', hamburger: 'beef', mince: 'beef', minced: 'beef', porkchop: 'pork', bacon: 'pork', seafood: 'fish', shellfish: 'fish', prawn: 'shrimp', prawns: 'shrimp', tuna: 'fish', salmon: 'fish',
  greens: 'vegetable', veggie: 'vegetable', veggies: 'vegetable', veg: 'vegetable', produce: 'vegetable', sweetcorn: 'corn', maize: 'corn', aubergine: 'eggplant', courgette: 'zucchini', coriander: 'cilantro', rocket: 'arugula',
  dairy: 'milk', cream: 'milk', cheddar: 'cheese', mozzarella: 'cheese', parmesan: 'cheese', margarine: 'butter', shortening: 'butter',
  pasta: 'noodle', noodles: 'noodle', spaghetti: 'noodle', macaroni: 'noodle', rice: 'rice', grains: 'grain', oats: 'oat', oatmeal: 'oat', flour: 'flour', bread: 'bread', toast: 'bread',
  sweetener: 'sugar', sugars: 'sugar', confectioners: 'sugar', icing: 'sugar', oil: 'oil', olive: 'oil', vinaigrette: 'vinegar',
  hot: 'spicy', heat: 'spicy', fiery: 'spicy', spicy: 'spicy', chile: 'pepper', chilli: 'pepper', chili: 'pepper', paprika: 'pepper',
  herbs: 'herb', spices: 'spice', garlics: 'garlic', gingerroot: 'ginger',
}
const INTENSITY_WORDS = {
  heavy: 2, heavily: 2, lots: 2, lot: 2, plenty: 2, very: 2, extra: 2, strong: 2,
  loaded: 2, generous: 2, abundant: 2, big: 2, more: 2, double: 2,
  light: 0.5, lightly: 0.5, little: 0.5, mild: 0.5, less: 0.5, low: 0.5, subtle: 0.5, small: 0.5,
}
const NAV_ITEMS = [
  ['home', 'Home', '⌂'],
  ['pantry', 'Pantry', '▦'],
  ['add', 'Add', '+'],
  ['recipes', 'Recipes', '✦'],
  ['saved', 'Saved', '♡'],
]
const THEME_KEY = 'recipe-finder-theme'
const TEXT_SIZE_KEY = 'recipe-finder-text-size'
const COMPACT_MODE_KEY = 'recipe-finder-compact-mode'
const HIGH_CONTRAST_KEY = 'recipe-finder-high-contrast'
const SHOW_TIPS_KEY = 'recipe-finder-show-tips'
const NOTIFICATION_KEY = 'recipe-finder-notification-reminders'
const TEXT_SCALE = {
  small: 0.9,
  medium: 1,
  large: 1.12,
}
let embeddingPipeline

function loadEmbeddingModel() {
  if (!embeddingPipeline) embeddingPipeline = pipeline('feature-extraction', EMBEDDING_MODEL, { dtype: 'q8' })
  return embeddingPipeline
}

async function embed(text, model) {
  const output = await model(text, { pooling: 'mean', normalize: true })
  return output.data
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

function tokenize(value) {
  return [...new Set(String(value || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).map((word) => word.endsWith('s') && word.length > 3 ? word.slice(0, -1) : word).map((word) => FOOD_SYNONYMS[word] || word).filter((word) => word.length > 2 && !STOP_WORDS.has(word)))]
}

function getFoodIntent(query) {
  const words = String(query || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)
  const intensity = words.reduce((value, word) => INTENSITY_WORDS[word] || value, 1)
  const targets = words.map((word) => FOOD_SYNONYMS[word] || (word.endsWith('s') && word.length > 3 ? word.slice(0, -1) : word)).filter((word) => word.length > 2 && !STOP_WORDS.has(word) && !INTENSITY_WORDS[word])
  return { targets: [...new Set(targets)], intensity }
}

function ingredientIntensity(recipe, target) {
  const rawIngredients = String(recipe.ingredients || '').toLowerCase()
  if (!rawIngredients.includes(target)) return 0
  const targetIndex = rawIngredients.indexOf(target)
  const nearbyText = rawIngredients.slice(Math.max(0, targetIndex - 35), targetIndex + target.length + 35)
  if (/\b(2|3|4|5|6|7|8|9|10|one|two|three|four|five|six|several|heaping|packed|heavy)\b/.test(nearbyText)) return 2
  if (/\b(pinche?s?|dash|sprinkle|small|little|light)\b/.test(nearbyText)) return 0.5
  return 1
}

function parseList(value) {
  if (!value) return []
  const matches = String(value).match(/["']([^"']+)["']/g)
  if (matches) return matches.map((item) => item.slice(1, -1))
  return String(value).split(',').map((item) => item.trim()).filter(Boolean)
}

function missingIngredientAnswer(question, ingredients) {
  const normalizedQuestion = question.toLowerCase()
  const ingredientText = ingredients.join(' ').toLowerCase()
  const substitutions = [
    ['milk', 'Use an equal amount of unsweetened plant milk. Water can work in a pinch, but the result will be less creamy.'],
    ['butter', 'Use an equal amount of margarine or neutral oil. Oil makes the result a little less rich.'],
    ['egg', 'For baking, try 1 tablespoon ground flaxseed mixed with 3 tablespoons water per egg and let it sit for 5 minutes. For binding, mashed banana or applesauce may work depending on the recipe.'],
    ['flour', 'Use an equal amount of all-purpose flour if the recipe calls for a different wheat flour. Gluten-free flour blends may work, but texture can change.'],
    ['sugar', 'Use an equal amount of granulated sugar, or reduce liquid sweetener slightly because it adds moisture.'],
  ]
  const missingMatch = normalizedQuestion.match(/(?:don't|do not|dont|no|without|missing)\s+(?:have\s+)?(?:any\s+)?([a-z]+)/)
  const missingIngredient = missingMatch?.[1]
  const substitution = substitutions.find(([name]) => missingIngredient?.startsWith(name) && ingredientText.includes(name))
  return substitution ? `Usually, yes. This recipe uses ${substitution[0]}; ${substitution[1]} The best choice depends on whether it is used for creaminess, baking structure, or a sauce.` : ''
}

function isQuestionEcho(answer, question) {
  const cleanAnswer = answer.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim()
  const cleanQuestion = question.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim()
  return cleanAnswer === cleanQuestion || cleanAnswer.includes(cleanQuestion) || cleanAnswer.length < 20
}

function cleanAssistantAnswer(answer, question) {
  const questionText = question.trim()
  return answer
    .replace(/^\s*(?:answer|response)\s*:\s*/i, '')
    .replace(new RegExp(`^\\s*(?:question|user question)\\s*:\\s*${questionText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*`, 'i'), '')
    .replace(/^\s*(?:answer|response)\s*:\s*/i, '')
    .trim()
}

function recipeIngredients(recipe) {
  return tokenize(recipe.NER || recipe.ingredients)
}

function dateStatus(expiration) {
  const days = Math.ceil((new Date(`${expiration}T23:59:59`) - new Date()) / 86400000)
  if (days < 0) return { label: 'Expired', className: 'expired', days }
  if (days <= 3) return { label: days === 0 ? 'Expires today' : `${days}d left`, className: 'soon', days }
  return { label: `${days}d fresh`, className: 'fresh', days }
}

function PantryApp({ user, onSignOut }) {
  const userInitials = user?.email ? user.email.split('@')[0].slice(0, 2).toUpperCase() : 'JD'
  const [activeTab, setActiveTab] = useState('home')
  const [pantry, setPantry] = useState(() => JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'))
  const [recipes, setRecipes] = useState([])
  const [query, setQuery] = useState('')
  const [selectedRecipe, setSelectedRecipe] = useState(null)
  const [saved, setSaved] = useState(() => JSON.parse(localStorage.getItem('recipe-finder-saved') || '[]'))
  const [showAdd, setShowAdd] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [form, setForm] = useState({ name: '', amount: '', expiration: '' })
  const [semanticResults, setSemanticResults] = useState([])
  const [semanticQuery, setSemanticQuery] = useState('')
  const [semanticStatus, setSemanticStatus] = useState('ready')
  const [semanticMessage, setSemanticMessage] = useState('')
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || 'light')
  const [textSize, setTextSize] = useState(() => localStorage.getItem(TEXT_SIZE_KEY) || 'medium')
  const [compactMode, setCompactMode] = useState(() => localStorage.getItem(COMPACT_MODE_KEY) === 'true')
  const [highContrast, setHighContrast] = useState(() => localStorage.getItem(HIGH_CONTRAST_KEY) === 'true')
  const [showTips, setShowTips] = useState(() => localStorage.getItem(SHOW_TIPS_KEY) !== 'false')
  const [notificationReminders, setNotificationReminders] = useState(() => localStorage.getItem(NOTIFICATION_KEY) !== 'false')

  useEffect(() => {
    const controller = new AbortController()
    const search = query.trim() ? `?q=${encodeURIComponent(query.trim())}&limit=60` : '?limit=60'
    setLoading(true)
    fetch(`${RECIPES_API}${search}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Recipe API returned HTTP ${response.status}.`)
        return response.json()
      })
      .then(({ recipes: loadedRecipes }) => {
        setRecipes(loadedRecipes || [])
        setError('')
      })
      .catch((requestError) => {
        if (requestError.name !== 'AbortError') {
          setRecipes([])
          setError('The recipe database could not be reached.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [query])

  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(pantry)) }, [pantry])
  useEffect(() => { localStorage.setItem('recipe-finder-saved', JSON.stringify(saved)) }, [saved])
  useEffect(() => { localStorage.setItem(THEME_KEY, theme) }, [theme])
  useEffect(() => { localStorage.setItem(TEXT_SIZE_KEY, textSize) }, [textSize])
  useEffect(() => { localStorage.setItem(COMPACT_MODE_KEY, String(compactMode)) }, [compactMode])
  useEffect(() => { localStorage.setItem(HIGH_CONTRAST_KEY, String(highContrast)) }, [highContrast])
  useEffect(() => { localStorage.setItem(SHOW_TIPS_KEY, String(showTips)) }, [showTips])
  useEffect(() => { localStorage.setItem(NOTIFICATION_KEY, String(notificationReminders)) }, [notificationReminders])

  const model = useMemo(() => {
    const frequency = new Map()
      const terms = recipes.map((recipe) => { const words = recipeIngredients(recipe); words.forEach((word) => frequency.set(word, (frequency.get(word) || 0) + 1)); return words })
    const idf = new Map([...frequency].map(([word, count]) => [word, Math.log((recipes.length + 1) / (count + 1)) + 1]))
    return { terms, idf, vocabulary: idf.size }
  }, [recipes])

  const results = useMemo(() => {
    const intent = getFoodIntent(query)
    const words = intent.targets
    if (!words.length) return recipes.slice(0, 60)
    const queryWeights = new Map(words.map((word) => [word, model.idf.get(word) || 0]))
    const queryNorm = Math.sqrt([...queryWeights.values()].reduce((sum, weight) => sum + weight ** 2, 0))
    return recipes.map((recipe, index) => {
      const weights = new Map(model.terms[index].map((word) => [word, model.idf.get(word) || 0]))
      const dot = words.reduce((sum, word) => sum + (queryWeights.get(word) || 0) * (weights.get(word) || 0), 0)
      const norm = Math.sqrt([...weights.values()].reduce((sum, weight) => sum + weight ** 2, 0))
      const intensityScore = words.reduce((sum, word) => sum + (weights.has(word) ? ingredientIntensity(recipe, word) : 0), 0)
      const intensityFit = intent.intensity >= 1 ? intensityScore * intent.intensity : intensityScore * (2 - intent.intensity)
      return { recipe, score: (queryNorm && norm ? dot / (queryNorm * norm) : 0) + intensityFit * 0.08 }
    }).filter(({ score }) => score > 0).sort((a, b) => b.score - a.score).map(({ recipe }) => recipe).slice(0, 60)
  }, [recipes, query, model])

  const visibleResults = semanticQuery === query && semanticResults.length ? semanticResults : results

  async function runSemanticMatch() {
    if (!query.trim() || !results.length) return
    setSemanticStatus('loading')
    setSemanticMessage('Loading MiniLM and comparing recipe meanings in your browser...')
    try {
      const model = await loadEmbeddingModel()
      const queryEmbedding = await embed(query, model)
      const queryWords = new Set(getFoodIntent(query).targets)
      const ranked = await Promise.all(results.slice(0, 80).map(async (recipe) => {
        const recipeWords = new Set(recipeIngredients(recipe))
        const matchedWords = [...queryWords].filter((word) => recipeWords.has(word))
        const coverage = queryWords.size ? matchedWords.length / queryWords.size : 0
        const recipeEmbedding = await embed(`${recipe.title}. Ingredients: ${recipe.NER || recipe.ingredients}`, model)
        const semanticScore = cosineSimilarity(queryEmbedding, recipeEmbedding)
        return { ...recipe, _coverage: coverage, _matchedWords: matchedWords, _semanticScore: semanticScore, _rankScore: coverage * .45 + semanticScore * .55 }
      }))
      setSemanticResults(ranked.sort((first, second) => second._rankScore - first._rankScore))
      setSemanticQuery(query)
      setSemanticStatus('ready')
      setSemanticMessage(`MiniLM compared meaning across ${ranked.length} candidate recipes. Ranking combines semantic similarity with ${Math.round((ranked[0]?._coverage || 0) * 100)}% ingredient coverage.`)
    } catch {
      setSemanticStatus('error')
      setSemanticMessage('The semantic model could not load. The explainable ingredient ranker is still active.')
    }
  }

  const freshCount = pantry.filter((item) => dateStatus(item.expiration).className === 'fresh').length
  const soonCount = pantry.filter((item) => dateStatus(item.expiration).className === 'soon').length
  const expiredCount = pantry.filter((item) => dateStatus(item.expiration).className === 'expired').length

  function addItem(event) {
    event.preventDefault()
    if (!form.name || !form.expiration) return
    setPantry((items) => [{ ...form, id: crypto.randomUUID() }, ...items])
    setForm({ name: '', amount: '', expiration: '' })
    setShowAdd(false)
    setActiveTab('pantry')
  }

  function removeItem(id) { setPantry((items) => items.filter((item) => item.id !== id)) }
  function toggleSaved(title) { setSaved((items) => items.includes(title) ? items.filter((item) => item !== title) : [...items, title]) }

  return <main className={`new-shell theme-${theme} size-${textSize} ${compactMode ? 'compact-mode' : ''} ${highContrast ? 'high-contrast' : ''}`} style={{ '--text-scale': TEXT_SCALE[textSize] }}>
    <header className="app-header">
      <div className="header-brand">
        <span className="brand-mark">rf</span>
        <div>
          <strong>recipe finder</strong>
          <small>your everyday kitchen companion</small>
        </div>
      </div>
      <div className="header-actions">
        {onSignOut && <button type="button" className="header-signout" onClick={onSignOut}>Sign out</button>}
        <div className="header-profile" aria-label="Profile">{userInitials}</div>
      </div>
    </header>
    <div className="page-body">
      {activeTab === 'home' && <HomeView pantry={pantry} freshCount={freshCount} soonCount={soonCount} expiredCount={expiredCount} onAdd={() => setShowAdd(true)} onPantry={() => setActiveTab('pantry')} onRecipes={() => setActiveTab('recipes')} onRemove={removeItem} onOpenSettings={() => setActiveTab('settings')} />}
      {activeTab === 'pantry' && <PantryView pantry={pantry} onAdd={() => setShowAdd(true)} onRemove={removeItem} onFindRecipes={() => { setQuery(pantry.map((item) => item.name).join(' ')); setActiveTab('recipes') }} />}
      {activeTab === 'recipes' && <RecipesView recipes={visibleResults} query={query} pantry={pantry} setQuery={(value) => { setQuery(value); setSemanticResults([]); setSemanticQuery('') }} loading={loading} error={error} saved={saved} onSave={toggleSaved} onSelect={setSelectedRecipe} onSemanticSearch={runSemanticMatch} semanticStatus={semanticStatus} semanticMessage={semanticMessage} semanticActive={semanticQuery === query && semanticResults.length > 0} />}
      {activeTab === 'saved' && <SavedView recipes={recipes.filter((recipe) => saved.includes(recipe.title))} pantry={pantry} saved={saved} onSave={toggleSaved} onSelect={setSelectedRecipe} />}
      {activeTab === 'settings' && <SettingsView theme={theme} textSize={textSize} compactMode={compactMode} highContrast={highContrast} showTips={showTips} notificationReminders={notificationReminders} onThemeChange={setTheme} onTextSizeChange={setTextSize} onCompactModeChange={setCompactMode} onHighContrastChange={setHighContrast} onShowTipsChange={setShowTips} onNotificationRemindersChange={setNotificationReminders} />}
    </div>
    <nav className="bottom-nav" aria-label="Main navigation">{NAV_ITEMS.map(([id, label, icon]) => id === 'add' ? <button key={id} className="nav-add" onClick={() => setShowAdd(true)} aria-label="Add pantry item"><span>{icon}</span><small>{label}</small></button> : <button key={id} className={activeTab === id ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab(id)}><span>{icon}</span><small>{label}</small></button>)}</nav>
    {showAdd && <AddModal form={form} setForm={setForm} onSubmit={addItem} onClose={() => setShowAdd(false)} />}
    {selectedRecipe && <RecipeDetail recipe={selectedRecipe} pantry={pantry} saved={saved.includes(selectedRecipe.title)} onSave={() => toggleSaved(selectedRecipe.title)} onClose={() => setSelectedRecipe(null)} />}
  </main>
}

function HomeView({ pantry, freshCount, soonCount, expiredCount, onAdd, onPantry, onRecipes, onRemove, onOpenSettings }) {
  return <section className="view home-view"><div className="welcome"><div><p className="kicker">TUESDAY, AUGUST 19</p><h1>Good morning, Jamie.</h1><p>Let's make something delicious today.</p></div><div className="home-actions"><button type="button" className="settings-button" onClick={onOpenSettings} aria-label="Open settings">⚙</button><span className="sun">☼</span></div></div><div className="home-grid"><div className="overview-panel"><div className="panel-title"><div><p className="kicker">YOUR PANTRY</p><h2>Keep it fresh.</h2></div><button className="text-button" onClick={onPantry}>See all →</button></div><div className="status-row"><StatusStat color="green" number={freshCount} label="Fresh" /><StatusStat color="yellow" number={soonCount} label="Use soon" /><StatusStat color="red" number={expiredCount} label="Expired" /></div>{pantry.length === 0 ? <div className="empty-pantry"><span>✚</span><p>Your pantry is waiting for its first ingredient.</p><button className="primary-button" onClick={onAdd}>Add an item</button></div> : <div className="mini-list">{pantry.slice(0, 3).map((item) => <PantryItem key={item.id} item={item} onRemove={onRemove} />)}</div>}</div><div className="inspire-panel"><p className="kicker">RECIPE MATCHING</p><h2>What's in your kitchen?</h2><p>Tell our local model what you have and discover recipes from your collection.</p><button className="primary-button" onClick={onRecipes}>Find a recipe <span>↗</span></button></div></div></section>
}

function PantryView({ pantry, onAdd, onRemove, onFindRecipes }) { return <section className="view"><PageHeading eyebrow="YOUR KITCHEN" title="Pantry" action={<button className="primary-button small" onClick={onAdd}>+ Add item</button>} /><div className="legend"><span><i className="light green" /> Fresh</span><span><i className="light yellow" /> Use soon</span><span><i className="light red" /> Expired</span></div>{pantry.length ? <><div className="pantry-actions"><p>{pantry.length} item{pantry.length === 1 ? '' : 's'} ready to match.</p><button className="primary-button" onClick={onFindRecipes}>Find recipes with these items <span>↗</span></button></div><div className="pantry-grid">{pantry.map((item) => <PantryItem key={item.id} item={item} onRemove={onRemove} />)}</div></> : <div className="blank-slate"><span>▦</span><h2>A calm pantry starts here.</h2><p>Add ingredients with their expiration dates and we will help you use them at the right time.</p><button className="primary-button" onClick={onAdd}>Add your first item</button></div>}</section> }

function RecipesView({ recipes, query, pantry, setQuery, loading, error, saved, onSave, onSelect, onSemanticSearch, semanticStatus, semanticMessage, semanticActive }) { const intent = getFoodIntent(query); return <section className="view"><PageHeading eyebrow="FROM YOUR RECIPE COLLECTION" title="Recipes" /><div className="recipe-search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search ingredients or describe a meal" aria-label="Search recipes" /></div><div className="ai-note"><span>✦</span><p><strong>Hybrid ML recommender</strong> first uses explainable ingredient coverage, then MiniLM embeddings understand related phrases such as “chicken breast”, “Roma tomato”, and “sea salt”.</p><button type="button" className="semantic-button" onClick={onSemanticSearch} disabled={!query.trim() || semanticStatus === 'loading'}>{semanticStatus === 'loading' ? 'Loading model...' : semanticActive ? 'Semantic match active' : 'Run semantic match'} <span>↗</span></button></div>{semanticStatus === 'error' && <div className="semantic-error" role="alert"><strong>AI semantic matching is unavailable.</strong><p>The MiniLM model did not load, so these are fallback keyword and ingredient-coverage results. No semantic AI score was applied.</p><button type="button" onClick={onSemanticSearch} disabled={!query.trim()}>Retry AI model</button></div>}{semanticMessage && semanticStatus !== 'error' && <p className="semantic-message">{semanticMessage}</p>}{query && intent.targets.length > 0 && <p className="interpreted-intent">Understood as: <strong>{intent.targets.join(', ')}</strong> · {intent.intensity > 1 ? 'heavier amount' : intent.intensity < 1 ? 'lighter amount' : 'regular amount'}</p>}{loading ? <div className="loading-state">Reading your recipe collection...</div> : error ? <p className="semantic-error" role="alert">{error}</p> : <div className="recipe-list">{recipes.slice(0, 12).map((recipe, index) => <RecipeCard key={`${recipe.title}-${index}`} recipe={recipe} pantry={pantry} saved={saved.includes(recipe.title)} onSave={onSave} onSelect={onSelect} />)}</div>}</section> }

function SavedView({ recipes, pantry, saved, onSave, onSelect }) { return <section className="view"><PageHeading eyebrow="YOUR SHORTLIST" title="Saved recipes" />{recipes.length ? <div className="recipe-list">{recipes.map((recipe, index) => <RecipeCard key={`${recipe.title}-${index}`} recipe={recipe} pantry={pantry} saved={saved.includes(recipe.title)} onSave={onSave} onSelect={onSelect} />)}</div> : <div className="blank-slate"><span>♡</span><h2>Save recipes for later.</h2><p>Tap the heart on any recipe that sounds like you.</p></div>}</section> }

function SettingsView({
  theme,
  textSize,
  compactMode,
  highContrast,
  showTips,
  notificationReminders,
  onThemeChange,
  onTextSizeChange,
  onCompactModeChange,
  onHighContrastChange,
  onShowTipsChange,
  onNotificationRemindersChange,
}) {
  return <section className="view settings-view">
    <PageHeading eyebrow="APP SETTINGS" title="Preferences" />
    <div className="settings-card">
      <div className="setting-block">
        <p className="kicker">THEME</p>
        <div className="option-row" role="radiogroup" aria-label="Theme preference">
          <button type="button" className={theme === 'light' ? 'choice-button active' : 'choice-button'} onClick={() => onThemeChange('light')}>Light</button>
          <button type="button" className={theme === 'dark' ? 'choice-button active' : 'choice-button'} onClick={() => onThemeChange('dark')}>Dark</button>
        </div>
      </div>

      <div className="setting-block">
        <p className="kicker">TEXT SIZE</p>
        <div className="slider-block">
          <label htmlFor="font-size" className="sr-only">Text size</label>
          <input id="font-size" type="range" min="0" max="2" step="1" value={textSize === 'small' ? 0 : textSize === 'medium' ? 1 : 2} onChange={(event) => {
            const next = Number(event.target.value)
            onTextSizeChange(next === 0 ? 'small' : next === 2 ? 'large' : 'medium')
          }} />
          <div className="size-labels"><span>Small</span><span>Medium</span><span>Large</span></div>
        </div>
      </div>

      <div className="setting-block">
        <p className="kicker">DISPLAY</p>
        <label className="toggle-row"><input type="checkbox" checked={compactMode} onChange={(event) => onCompactModeChange(event.target.checked)} /> <span>Compact cards</span></label>
        <label className="toggle-row"><input type="checkbox" checked={highContrast} onChange={(event) => onHighContrastChange(event.target.checked)} /> <span>High contrast mode</span></label>
      </div>

      <div className="setting-block">
        <p className="kicker">HELP & REMINDERS</p>
        <label className="toggle-row"><input type="checkbox" checked={showTips} onChange={(event) => onShowTipsChange(event.target.checked)} /> <span>Show kitchen tips</span></label>
        <label className="toggle-row"><input type="checkbox" checked={notificationReminders} onChange={(event) => onNotificationRemindersChange(event.target.checked)} /> <span>Expiration reminders</span></label>
      </div>
    </div>
  </section>
}

function PageHeading({ eyebrow, title, action }) { return <div className="page-heading"><div><p className="kicker">{eyebrow}</p><h1>{title}</h1></div>{action}</div> }
function StatusStat({ color, number, label }) { return <div className="status-stat"><i className={`light ${color}`} /><strong>{number}</strong><span>{label}</span></div> }
function PantryItem({ item, onRemove }) { const status = dateStatus(item.expiration); return <article className="pantry-item"><i className={`light ${status.className}`} /><div><strong>{item.name}</strong><small>{item.amount || 'Pantry item'}</small></div><span className="expiry-label">{status.label}</span><button className="remove-button" onClick={() => onRemove(item.id)} aria-label={`Remove ${item.name}`}>×</button></article> }
function RecipeCard({ recipe, pantry = [], saved, onSave, onSelect }) { const ingredients = parseList(recipe.NER || recipe.ingredients); const ingredientWords = new Set(recipeIngredients(recipe)); const pantryWords = new Set(pantry.flatMap((item) => tokenize(item.name))); const owned = [...ingredientWords].filter((word) => pantryWords.has(word)).length; return <article className="recipe-result" onClick={() => onSelect(recipe)}><div className="recipe-art">{['🥬', '🍋', '🍅', '🥕', '🍳'][recipe.title.length % 5]}</div><div className="recipe-copy"><div className="recipe-meta"><span className={owned ? 'match-count' : ''}>{recipe._coverage !== undefined ? `${Math.round(recipe._coverage * 100)}% semantic match` : `${owned}/${ingredientWords.size} ingredients`}</span><button className={saved ? 'heart saved' : 'heart'} onClick={(event) => { event.stopPropagation(); onSave(recipe.title) }} aria-label={saved ? 'Remove saved recipe' : 'Save recipe'}>{saved ? '♥' : '♡'}</button></div><h3>{recipe.title}</h3><p>{ingredients.slice(0, 4).join(' · ')}</p>{recipe._matchedWords?.length > 0 && <small className="matched-terms">Matched: {recipe._matchedWords.join(', ')}</small>}<button className="view-link" onClick={() => onSelect(recipe)}>View recipe →</button></div></article> }
function AddModal({ form, setForm, onSubmit, onClose }) { return <div className="modal-backdrop"><form className="add-modal" onSubmit={onSubmit}><button type="button" className="close-button" onClick={onClose} aria-label="Close">×</button><p className="kicker">NEW PANTRY ITEM</p><h2>What did you bring home?</h2><label>Ingredient name<input autoFocus value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="e.g. strawberries" required /></label><label>Amount <span className="optional">optional</span><input value={form.amount} onChange={(event) => setForm({ ...form, amount: event.target.value })} placeholder="e.g. 1 punnet" /></label><label>Expiration date<input type="date" value={form.expiration} onChange={(event) => setForm({ ...form, expiration: event.target.value })} required /></label><button className="primary-button" type="submit">Add to pantry <span>↗</span></button></form></div> }
function RecipeDetail({ recipe, pantry, saved, onSave, onClose }) {
  const ingredients = useMemo(() => parseList(recipe.ingredients || recipe.NER), [recipe.ingredients, recipe.NER])
  const directions = useMemo(() => parseList(recipe.directions), [recipe.directions])
  const pantryWords = new Set(pantry.flatMap((item) => tokenize(item.name)))
  const [showAssistant, setShowAssistant] = useState(false)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [assistantStatus, setAssistantStatus] = useState('idle')
  const [assistantError, setAssistantError] = useState('')
  const [microStatus, setMicroStatus] = useState('idle')
  const [microAnswer, setMicroAnswer] = useState('')
  const [microError, setMicroError] = useState('')
  const [microProgress, setMicroProgress] = useState({ percent: 0, file: '' })
  const microWorker = useRef(null)
  const microQuestion = useRef('')

  useEffect(() => {
    if (!showAssistant) return undefined
    const worker = new Worker(new URL('./recipeAi.worker.js', import.meta.url), { type: 'module' })
    microWorker.current = worker
    worker.onmessage = ({ data }) => {
      if (data.type === 'status') setMicroStatus(data.status)
      if (data.type === 'progress') setMicroProgress(data)
      if (data.type === 'answer') {
        const helperAnswer = missingIngredientAnswer(microQuestion.current, ingredients)
        const cleanedAnswer = cleanAssistantAnswer(data.answer, microQuestion.current)
        setMicroAnswer(helperAnswer && isQuestionEcho(cleanedAnswer, microQuestion.current) ? helperAnswer : cleanedAnswer)
        setMicroStatus('ready')
      }
      if (data.type === 'error') {
        setMicroError(data.error)
        setMicroStatus('error')
      }
    }
    worker.onerror = () => {
      setMicroError('The browser worker could not start the micro-model.')
      setMicroStatus('error')
    }
    return () => {
      worker.terminate()
      microWorker.current = null
    }
  }, [showAssistant, ingredients])

  async function askRecipeAssistant(event) {
    event.preventDefault()
    if (!question.trim()) return
    setAssistantStatus('loading')
    setAnswer('')
    setAssistantError('')
    const prompt = `You answer only the user's question about this recipe. Give one direct, concise answer in 1 or 2 sentences. Do not repeat the question. Do not include labels, headings, or extra questions. If the recipe context is not enough, say you do not know.\n\nRecipe: ${recipe.title}\nIngredients:\n${ingredients.join('\n')}\nDirections:\n${directions.join('\n')}\n\nUser question: ${question.trim()}\nDirect answer:`
    try {
      const response = await fetch('http://127.0.0.1:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'qwen2.5:0.5b', prompt, stream: false }),
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.error || `Ollama returned HTTP ${response.status}.`)
      if (!payload.response?.trim()) throw new Error('Ollama returned an empty response.')
      setAnswer(cleanAssistantAnswer(payload.response, question))
      setAssistantStatus('ready')
    } catch (error) {
      setAssistantStatus('error')
      setAssistantError(`${error.message} Start Ollama and run: ollama pull qwen2.5:0.5b`)
    }
  }

  function askMicroAssistant() {
    if (!question.trim() || !microWorker.current) return
    microQuestion.current = question.trim()
    setMicroAnswer('')
    setMicroError('')
    setMicroProgress({ percent: 0, file: '' })
    setMicroStatus('loading')
    microWorker.current.postMessage({
      type: 'ask',
      prompt: `Answer only the user's question about this recipe in 1 or 2 direct sentences. Do not repeat the question. Do not include Question:, Answer:, headings, or extra questions. If the recipe context is not enough, say you do not know. Recipe: ${recipe.title}. Ingredients: ${ingredients.join('; ')}. Directions: ${directions.join(' ')}. User question: ${question.trim()}. Direct answer:`,
    })
  }

  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="detail-panel" role="dialog" aria-modal="true"><button className="close-button" onClick={onClose} aria-label="Close recipe">×</button><p className="kicker">{recipe.source || 'RECIPE COLLECTION'}</p><h2>{recipe.title}</h2><div className="detail-actions"><button className="save-detail" onClick={onSave}>{saved ? '♥ Saved' : '♡ Save recipe'}</button><button className="ask-recipe-button" onClick={() => setShowAssistant(true)}>Ask AI about this recipe <span>✦</span></button></div><div className="detail-columns"><div><h4>Ingredients</h4>{pantry.length > 0 && <p className="ingredient-key"><i className="light green" /> Highlighted ingredients are in your pantry</p>}<ul className="ingredient-list">{ingredients.map((item, index) => { const owned = tokenize(item).some((word) => pantryWords.has(word)); return <li className={owned ? 'owned-ingredient' : ''} key={`${item}-${index}`}><span>{item}</span>{owned && <small>In pantry</small>}</li> })}</ul></div><div><h4>Directions</h4><ol>{directions.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}</ol></div></div>{showAssistant && <div className="recipe-ai-panel" role="region" aria-label="Ask AI about this recipe"><button className="recipe-ai-close" type="button" onClick={() => setShowAssistant(false)} aria-label="Close recipe AI">×</button><p className="kicker">LOCAL AI OPTIONS</p><h3>Ask about {recipe.title}</h3><p className="recipe-ai-note">Qwen uses Ollama. Browser Qwen 2.5 0.5B runs locally in a Web Worker through Transformers.js and does not require Ollama or WebGPU, but needs a larger first download and more device memory.</p>{(microStatus === 'loading' || microStatus === 'generating') && <div className="micro-progress" role="status"><div className="micro-progress-heading"><strong>{microStatus === 'generating' ? 'Model downloaded' : 'Downloading browser model'}</strong><span>{microStatus === 'generating' ? 'Generating answer...' : `${Math.round(microProgress.percent)}%`}</span></div>{microStatus === 'loading' && <progress max="100" value={microProgress.percent} />}{microStatus === 'loading' && microProgress.file && <small>{microProgress.file.split('/').pop()}</small>}</div>}<form onSubmit={askRecipeAssistant}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Can I substitute an ingredient? What can I prepare ahead?" aria-label="Question about this recipe" /><div className="recipe-ai-actions"><button className="primary-button" type="submit" disabled={!question.trim() || assistantStatus === 'loading'}>{assistantStatus === 'loading' ? 'Asking Qwen...' : 'Ask Qwen'} <span>↗</span></button><button className="micro-button" type="button" onClick={askMicroAssistant} disabled={!question.trim() || microStatus === 'loading'}>{microStatus === 'loading' || microStatus === 'generating' ? 'Loading browser Qwen...' : 'Ask browser Qwen 2.5 0.5B'} <span>◌</span></button></div></form>{assistantError && <p className="recipe-ai-error" role="alert">Ollama Qwen did not answer: {assistantError}</p>}{answer && <div className="recipe-ai-answer"><strong>Ollama Qwen</strong><p>{answer}</p></div>}{microError && <p className="recipe-ai-error" role="alert">Browser Qwen did not answer: {microError}</p>}{microAnswer && <div className="recipe-ai-answer micro-answer"><strong>Browser Qwen + recipe helper</strong><p>{microAnswer}</p></div>}</div>}</section></div>
}

export default PantryApp
