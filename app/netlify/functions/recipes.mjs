const allowedColumns = 'id,title,ingredients,directions,link,source,NER'

export default async function handler(request) {
  const supabaseUrl = process.env.SUPABASE_URL
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!supabaseUrl || !serviceRoleKey) {
    return json({ error: 'Supabase environment variables are not configured.' }, 500)
  }

  const requestUrl = new URL(request.url)
  const query = requestUrl.searchParams.get('q')?.trim() || ''
  const limit = Math.min(Math.max(Number(requestUrl.searchParams.get('limit')) || 60, 1), 100)
  const source = requestUrl.searchParams.get('source')?.trim() || ''
  const params = new URLSearchParams({ select: allowedColumns, limit: String(limit) })
  if (source && source !== 'All sources') params.set('source', `eq.${source}`)
  if (query) {
    const escapedQuery = query.replace(/[(),]/g, ' ').replace(/[*]/g, '')
    params.set('or', `(title.ilike.*${escapedQuery}*,ingredients.ilike.*${escapedQuery}*,NER.ilike.*${escapedQuery}*)`)
  }

  const response = await fetch(`${supabaseUrl}/rest/v1/recipes?${params}`, {
    headers: { apikey: serviceRoleKey, Authorization: `Bearer ${serviceRoleKey}` },
  })
  const payload = await response.json().catch(() => [])
  if (!response.ok) return json({ error: payload.message || 'Supabase query failed.' }, response.status)
  return json({ recipes: payload })
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}