import fs from 'node:fs'
import path from 'node:path'
import Papa from 'papaparse'

const supabaseUrl = process.env.SUPABASE_URL
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY
if (!supabaseUrl || !serviceRoleKey) throw new Error('Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY first.')

const csvPath = process.argv[2] || path.resolve('public/recipes_200000.csv')
const batchSize = 500
let batch = []
let imported = 0

function normalize(row) {
  return {
    title: row.title,
    ingredients: row.ingredients,
    directions: row.directions,
    link: row.link,
    source: row.source,
    NER: row.NER,
  }
}

async function upload(rows) {
  const response = await fetch(`${supabaseUrl}/rest/v1/recipes`, {
    method: 'POST',
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      'Content-Type': 'application/json',
      Prefer: 'return=minimal',
    },
    body: JSON.stringify(rows),
  })
  if (!response.ok) throw new Error(`Import failed with HTTP ${response.status}: ${await response.text()}`)
  imported += rows.length
  if (imported % 5000 === 0) console.log(`Imported ${imported} recipes.`)
}

await new Promise((resolve, reject) => {
  const stream = fs.createReadStream(csvPath)
  Papa.parse(stream, {
    header: true,
    skipEmptyLines: true,
    step(results, parser) {
      parser.pause()
      batch.push(normalize(results.data))
      const pending = batch.length >= batchSize ? upload(batch).then(() => { batch = [] }) : Promise.resolve()
      pending.then(() => parser.resume()).catch(reject)
    },
    complete() {
      upload(batch).then(() => { console.log(`Imported ${imported} recipes total.`); resolve() }).catch(reject)
    },
    error: reject,
  })
})