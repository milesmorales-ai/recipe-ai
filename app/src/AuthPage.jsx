import { useEffect, useState } from 'react'
import PantryApp from './PantryApp.jsx'
import { supabase } from './lib/supabase.js'
import './AuthPage.css'

const emptyForm = {
  name: '',
  email: '',
  password: '',
}

function AuthPage() {
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState(emptyForm)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    if (!supabase) {
      setLoading(false)
      return undefined
    }

    let mounted = true

    supabase.auth.getSession().then(({ data: { session: activeSession } }) => {
      if (mounted) setSession(activeSession)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      if (mounted) setSession(nextSession)
      if (mounted && nextSession) setLoading(false)
    })

    setLoading(false)

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  const handleChange = (event) => {
    const { name, value } = event.target
    setForm((current) => ({ ...current, [name]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setNotice('')

    if (!supabase) {
      setError('Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in your .env file to enable authentication.')
      return
    }

    setIsSubmitting(true)

    try {
      if (mode === 'signup') {
        const { error: signUpError } = await supabase.auth.signUp({
          email: form.email,
          password: form.password,
          options: {
            data: {
              full_name: form.name.trim() || form.email.split('@')[0],
            },
          },
        })

        if (signUpError) throw signUpError

        setNotice('Account created. Check your email and confirm the sign-up link before logging in.')
        setMode('login')
        setForm({ ...emptyForm, email: form.email })
        return
      }

      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: form.email,
        password: form.password,
      })

      if (signInError) throw signInError
    } catch (authError) {
      setError(authError.message || 'Authentication failed. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSignOut = async () => {
    if (!supabase) return
    await supabase.auth.signOut()
    setSession(null)
  }

  if (loading) {
    return (
      <div className="auth-shell auth-loading">
        <div className="spinner" aria-label="Loading authentication" />
      </div>
    )
  }

  if (session) {
    return <PantryApp user={session.user} onSignOut={handleSignOut} />
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="brand-block">
          <div className="brand-mark">rf</div>
          <div>
            <p className="eyebrow">Recipe Finder</p>
            <h1>Welcome back</h1>
          </div>
        </div>

        <div className="auth-toggle" role="tablist" aria-label="Authentication mode">
          <button
            type="button"
            className={mode === 'login' ? 'auth-tab active' : 'auth-tab'}
            onClick={() => setMode('login')}
          >
            Log in
          </button>
          <button
            type="button"
            className={mode === 'signup' ? 'auth-tab active' : 'auth-tab'}
            onClick={() => setMode('signup')}
          >
            Sign up
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'signup' && (
            <label className="field">
              <span>Full name</span>
              <input
                type="text"
                name="name"
                value={form.name}
                onChange={handleChange}
                placeholder="Jamie Parker"
              />
            </label>
          )}

          <label className="field">
            <span>Email</span>
            <input
              type="email"
              name="email"
              value={form.email}
              onChange={handleChange}
              placeholder="you@example.com"
              required
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              name="password"
              value={form.password}
              onChange={handleChange}
              placeholder="Enter your password"
              minLength={6}
              required
            />
          </label>

          {error && <p className="auth-message auth-error">{error}</p>}
          {notice && <p className="auth-message auth-notice">{notice}</p>}

          {!supabase && (
            <p className="auth-message auth-warning">
              Supabase is not configured yet. Add your project URL and anon key to an .env file and reload.
            </p>
          )}

          <button type="submit" className="primary-auth-button" disabled={isSubmitting || !supabase}>
            {isSubmitting ? 'Please wait...' : mode === 'login' ? 'Log in' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  )
}

export default AuthPage
