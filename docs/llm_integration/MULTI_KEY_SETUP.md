# Using Multiple Gemini API Keys (Auto-Rotation)

To work around free tier quota limits, you can use multiple Gemini API keys from different Google accounts. The system will automatically rotate to the next key when one hits its quota.

## Setup

### Method 1: Environment Variable (Recommended)

Set multiple API keys separated by commas:

```bash
export GEMINI_API_KEYS="AIza...key1,AIza...key2,AIza...key3"
```

### Method 2: Individual Keys

Set keys separately (less convenient but works):

```bash
# This is what you'd set for multiple accounts
# Only one will be read by default, so use Method 1 instead
export GEMINI_API_KEY="AIza...key1"
```

### Method 3: In Code

Pass multiple keys directly:

```python
from final_sc_review.llm.gemini_client import GeminiClient

client = GeminiClient(
    api_keys=[
        "AIza...key1",
        "AIza...key2",
        "AIza...key3"
    ]
)
```

## How It Works

1. **Start with first key**: System uses the first API key initially
2. **Detect quota exhaustion**: When you hit 429/RESOURCE_EXHAUSTED error
3. **Automatic rotation**: Switches to next available key immediately
4. **Continue seamlessly**: Request retries with new key, no interruption
5. **Track exhausted keys**: Won't retry with exhausted keys

## Example Usage

```bash
# Set up 3 API keys (from 3 different Google accounts)
export GEMINI_API_KEYS="AIzaKey1...,AIzaKey2...,AIzaKey3..."

# Run pilot - will use all 3 keys automatically
python scripts/llm_integration/run_llm_pilot.py --n_samples 50
```

**Output:**
```
INFO | Initialized GeminiClient with model=gemini-1.5-flash, temp=0.0, 3 API key(s)
INFO | Processing query 1...
...
WARNING | Quota exhausted for API key 1
INFO | Rotated to API key 2/3 (1 exhausted)
INFO | Retrying with new API key...
INFO | Processing query 20...
...
WARNING | Quota exhausted for API key 2
INFO | Rotated to API key 3/3 (2 exhausted)
INFO | Retrying with new API key...
...
```

## Free Tier Limits

Each free tier account gets (approximately):
- **60 requests per minute**
- **1,500 requests per day**
- Resets every 24 hours

### Capacity Calculation

With 3 accounts:
- **180 requests/min** combined
- **~4,500 requests/day** total

**For our experiments:**
- Pilot (50 queries): ~50-250 requests → **1 key sufficient**
- Full 5-fold (14,770 queries): ~15,000-75,000 requests → **Need 10-20 keys** or enable billing

## Getting Multiple API Keys

### Create Additional Google Accounts

1. Create new Gmail account (e.g., `myproject.key2@gmail.com`)
2. Go to https://makersuite.google.com/app/apikey
3. Sign in with new account
4. Click "Create API Key"
5. Copy and add to `GEMINI_API_KEYS`
6. Repeat for as many keys as needed

### Best Practices

- Use descriptive account names (e.g., `research.key1@gmail.com`)
- Keep keys in a secure location (password manager)
- Don't commit keys to git
- Consider enabling billing on 1 account instead for production use

## Troubleshooting

**Error: "All API keys exhausted"**
- All keys hit quota limits
- Wait 24 hours for reset OR enable billing

**Keys not rotating**
- Check `GEMINI_API_KEYS` format (comma-separated, no spaces after commas)
- Verify each key works individually first

**Still getting quota errors**
- Free tier limit: 0 means account needs billing enabled
- Create new Google account for fresh quota

## Recommendation

**For Pilot** (50 samples): **1-2 keys** sufficient

**For Full 5-fold CV**:
- **Option A**: 10-20 free accounts (tedious setup)
- **Option B**: Enable billing on 1 account (**recommended**, $2-9 total cost)

Enabling billing is much easier and still very cheap!
