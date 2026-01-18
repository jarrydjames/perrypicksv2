# Persistence options for PerryPicks (Streamlit Cloud free)

You want: **no paid hosting**, **no paid APIs**, runs on **Streamlit Cloud free**.
So persistence must be:
- simple
- not require external services
- not break if the container restarts

## Option A (V1): session_state only
- Store everything in `st.session_state`
- ✅ easiest
- ❌ you lose history on refresh/new session
- ❌ can’t do real longitudinal performance tracking

Use this only for prototype UX.

## Option B (Recommended MVP): SQLite + Export/Import

### How it works
- Write to a local SQLite database file (e.g., `data/perrypicks.sqlite`)
- Provide:
  - **Export**: download bets + snapshots as CSV/JSON
  - **Import**: upload a prior export to restore history

### Why it’s best-practice here
- ✅ no external dependencies
- ✅ enables your key feature: time-series p(hit) monitoring
- ✅ export makes it durable even if Streamlit resets

### Limitations
- Streamlit Cloud disk persistence isn’t guaranteed across redeploys.
- Export is the safety net.

## Option C (Later, optional): free-tier hosted DB
Possible providers: Supabase/Neon/etc.
- ✅ real durability, multi-device, multi-user potential
- ⚠️ requires secrets + external service accounts
- ⚠️ free tiers change; you said “no paid hosting”, so this should be optional.

## Practical recommendation
Ship V2 with:
- SQLite backend
- export/import
- storage interface (`Storage` protocol / abstract class) so swapping backends is trivial

That gives you best-of-both worlds: real tracking + still free.
