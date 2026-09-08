# agentic_macro

A discretionary macro sleeve you drive from Telegram. You state a worldview; it proposes the
structure that expresses it; you approve; the orders go in through the
[algo_trade executor](../execution/algo_trade).

```
/worldview the Fed cuts three times into an intact labour market

  STRUCTURE  steepener  [dv01_neutral]  ·  confidence high

  Cuts front-load the front end while the long end stays anchored by term premium
  and supply. This is the slope, not the level — DV01-matched so a parallel rally
  makes nothing either way.

  LEGS
    LONG  SHY      1,099 @    82.00  =      $90,118
          front end, rallies hardest as cuts are priced
    SHORT TLT        111 @    88.00  =       $9,768
          long end, anchored by term premium

    hedge ratio: dv01_neutral: each side carries 162,295 of duration-dollars
                 -> 9.17:1 dollars long:short

  CAPITAL
    this view    $99,886 gross (budget $100,000)
    sleeve after $99,886 of $250,000 (40%)

  This places real orders. Reply:
  /confirm a3f9
```

## The idea

The sleeve holds **many worldviews at once**. Each is a thesis you approved plus the legs
that express it, and the book sent to the executor is the **net of every active worldview**,
summed per symbol. Closing a worldview removes exactly its legs — the unwind is computed,
not remembered, so you never have to work out which shares belonged to the oil trade.

```
  #1 "cuts are coming"        long SHY 1099, short TLT 111
  #2 "energy supply is tight" long XLE 400                     ->  /targets
  #3 "dollar tops out"        long UDN 300, short EEM 120          SHY  +1099
                                                                   TLT   -111
                                                                   XLE   +400
                                                                   UDN   +300
                                                                   EEM   -120
```

## The hedge ratio is the trade

A steepener is not "long SHY, short TLT". It is long SHY against short TLT *sized so both
legs carry the same DV01*. SHY runs ~1.8 years of duration and TLT ~16.5, so the correct
ratio is about **9.2 dollars to 1** — sized by equal dollars instead you would own a large
outright duration short that merely contains a steepener, with a P&L driven by the level of
rates rather than the slope you actually had a view on.

So the model never sizes anything. It names a structure, the instruments, the direction, and
a relative weight per leg; the code turns that into share counts using a live price and the
instrument risk data in [`universe.py`](agentic_macro/universe.py). Weights in a
risk-weighted structure are shares of **risk**, not dollars, which is what makes multi-leg
structures come out right without special-casing any of them:

| structure | weights given | dollars produced | property |
|---|---|---|---|
| steepener | SHY 1, TLT 1 | 9.17 : 1 | net DV01 ≈ 0 — pure slope |
| butterfly | IEF 1, SHY 1, TLT 1 | 0.31 : 0.62 : 0.07 | wings equal *and* summing to the belly — pure curvature |
| banks vs index | XLF 1, SPY 1 | 0.95 : 1 | net beta ≈ 0 — pure sector view |

The named structures live in [`playbooks.py`](agentic_macro/playbooks.py) — 28 of them, each
with its legs, its weighting rule, when it works and how it fails.

> **Why not a vector DB?** The whole library is a few thousand tokens: it fits in the prompt,
> it caches, and the model sees *all* of it every call. Retrieval could only subtract — its
> characteristic failure is returning four plausible neighbours and not the right one, and a
> missing playbook does not announce itself, it just produces a vaguer trade that still looks
> reasonable on approval. `playbooks.select()` is the seam: it returns everything today,
> ranked, and is the one function an embedding index would replace if the library ever grows
> past what the prompt can hold.

## Two gates, because a view is not an order

`/confirm` no longer submits. It accepts the worldview and shows the **orders** that would
actually be placed, then `/place` sends them.

The distinction is not ceremony. A worldview's legs say what it wants to *hold*; the order
is the delta against what the broker holds now, netted with every other active view. A view
reading "long 1,099 SHY" is a 299-share buy if another view already holds 800, and a *sell*
if the book is longer than the target. Approving the legs never told you which — so the last
thing approved before money moves is the order list itself, recomputed from live broker state
at the moment of confirmation.

```
/worldview ...   ->  the view: structure, legs, hedge ratio, capital
/confirm m7c2e   ->  the orders: BUY SHY 676 (+0 -> +676), SELL TLT 73 ...
/place m5f9d     ->  sent
```

`/close` and `/sync` show the same order list in their confirmation. Tokens are scoped to the
step they were issued for: a `/confirm` token cannot be replayed at `/place` to skip the
order review, and every token is single-use.

## Paper mode

`AGENTIC_PAPER=true` runs everything — proposals, both gates, the store, the order previews —
without ever contacting the executor. The stored book stands in for the broker, so the order
previews stay truthful instead of showing every position as a fresh trade.

It is a flag rather than commented-out code deliberately: commenting out the submission fails
in the dangerous direction, because the code still *looks* like it trades and every message
still says "is on". The flag is printed on every gate and in the startup banner, so the mode
you are in is never something you have to remember.

## Safety properties

These are the ones the code is shaped around; each has tests that pin it.

**What you approve is exactly what is sent.** The confirmation token carries the *sized
proposal* — symbols, signed share counts, the prices they were sized at — not the worldview
text. Re-running the model on approval is the obvious implementation and a serious bug: the
same thesis can yield a different structure on a second call, and you would be confirming
one trade having read another. Nothing between `/worldview` and `/confirm` calls the model.

**A stale approval is refused, not resized.** Quantities are frozen at approval, so a leg
that has moved more than `AGENTIC_MAX_PRICE_DRIFT` (2%) since pricing is no longer the
notional you read. That refuses and asks for a fresh proposal — checked at `/confirm` *and*
again at `/place`, since the second one is what actually guards the fill.

**A hallucinated ticker is impossible, not caught.** The universe is a closed list, and it
generates the `enum` in the response schema.

**Prices fail closed.** `expected_price` is what the executor values the book with to apply
the allocation cap and to measure slippage, so a missing, zero, negative, stale or NaN price
raises rather than defaulting. A partially priced book is never submitted — `/targets` reads
an omitted name as "close it", so a half-priced fetch would quietly flatten the rest.

**A structure is never silently reshaped.** If the budget cannot afford a leg of a
risk-matched structure, it refuses. A butterfly missing a wing is a steepener — a different
trade, not a smaller one.

**Rounding is toward zero, on both signs.** Rounding up would overspend the sleeve in the
same direction on every leg.

## Setup

```bash
python3.12 -m venv venv_agentic && ./venv_agentic/bin/pip install -r requirements.txt
cp .env.example .env        # then fill it in
./venv_agentic/bin/python -m pytest tests/ -q
```

Four things must be true before it can trade:

0. **`GEMINI_API_KEY` set.** The proposer runs on Gemini through the Google Gen AI SDK
   (`GOOGLE_API_KEY` is accepted as an alias).

1. **Its own Telegram bot.** Create one with @BotFather and add it to the group.

   It can live in the **same chat** as the executor's control bot — that is supported and
   tested. What it must *not* share is the **token**: two processes long-polling
   `getUpdates` on one token fight over every update (Telegram hands each to exactly one
   consumer and 409s the other), which would silently break `/status` and `/kill` on the box
   that holds the positions. See [Sharing a chat](#sharing-a-chat).

2. **`AGENTIC_ALLOWED_USER_IDS` set.** Empty means nobody can trade — it fails closed.

3. **The strategy id registered with the executor**, or every intent is rejected as "not
   active":

   ```
   /addstrategy agentic_macro 250k 15%      (in the executor's control bot, then /confirm)
   ```

## Use

```bash
python run_bot.py                     # the Telegram bot

# or the same sleeve from a terminal — the fastest way to try a view
python3 -m agentic_macro.cli propose --dry-run "The market is overpricing the odds of another rate hike"
python3 -m agentic_macro.cli views
python3 -m agentic_macro.cli book
python3 -m agentic_macro.cli sync --dry-run
```

`propose --dry-run` touches neither the executor nor the store, so it works before the
tunnel is up.

| command | what |
|---|---|
| `/worldview [size] <view>` | propose a structure. Optional leading size: `/worldview 50k oil is tight` |
| `/confirm <token>` | accept the view, and see the **orders** it implies. Sends nothing |
| `/place <token>` | send those orders — the last gate |
| `/views` `/view <id>` | what is on, and why |
| `/close <id>` | retire a worldview; its legs unwind on the next book |
| `/book` | the netted book, and which views each name comes from |
| `/sync` | re-send the netted book — self-heals drift, safe to repeat |
| `/playbooks [name]` | the structures available |

## Layout

| path | what |
|---|---|
| [`universe.py`](agentic_macro/universe.py) | the closed instrument list + the duration/beta data hedge ratios come from |
| [`playbooks.py`](agentic_macro/playbooks.py) | the named structures, and the retrieval seam |
| [`proposer.py`](agentic_macro/proposer.py) | the Gemini call, and the risk-space allocator |
| [`store.py`](agentic_macro/store.py) | worldviews, their legs, and the netting |
| [`strategy.py`](agentic_macro/strategy.py) | the netted book, and how it reaches the executor |
| [`bot.py`](agentic_macro/bot.py) | Telegram: propose, approve, close |
| [`cli.py`](agentic_macro/cli.py) | the same, from a terminal |
| [`executor/`](agentic_macro/executor/) | vendored from `algo_trade/client/` — see the header in each file |

## Sharing a chat

Both bots can sit in one group. Telegram delivers every slash command to both, so three
things keep that readable:

- **Tokens are prefixed** (`m7c2e`). Both bots implement `/confirm` and both issue
  `secrets.token_hex(2)` — identical four-hex tokens. Without a prefix every confirmation
  reaches both, and whichever does not own it replies *"nothing pending with that token"*.
  That is worse than noise: it teaches you to ignore that line on the day a `/kill`
  confirmation genuinely has expired. A token without our prefix now gets silence.
- **Unknown commands get silence.** `/status`, `/kill`, `/allocate` belong to the other bot,
  which already answers real typos — so only one bot ever replies to a given command.
- **`/cmd@TheirBot` is honoured**, not stripped, so an explicitly addressed command is left
  alone.

One asymmetry worth knowing: the executor's `telegram_control.py` still *strips* the
`@botname` suffix and still answers unknown commands, so it will reply to `/worldview` with
"unknown command". Harmless, but I can make the same three changes there if you want the
chat fully quiet — that is a change to the other repo, so I left it alone.

## The model

Gemini, defaulting to `gemini-3.5-flash` at `thinking_level: high` — the model verified to
answer on this key with this response schema. `AGENTIC_MODEL` overrides it.

There is a **fallback chain** (`AGENTIC_FALLBACK_MODELS`) because probing the key found
three different ways a model declines to serve, none of which mean the worldview was bad:
`gemini-3.1-pro-preview` was over its free-tier quota (429), `gemini-3.8-flash` was under
load (503), and the `gemini-2.5-*` models are still returned by `models.list()` but have
been retired (404). A proposal lost to any of those is a trade you never got to consider, so
the next model is asked instead. A **400 is never retried** — a malformed request is the same
on every model, and hiding it behind "no model was available" would send you looking in the
wrong place.

Two Gemini-specific details worth knowing if you touch [`proposer.py`](agentic_macro/proposer.py):
its response schema is an OpenAPI 3.0 subset, so `additionalProperties` must be stripped or
the request is a hard 400; and a refusal or truncation arrives as a normal `200` with a
`finish_reason` of `SAFETY` / `MAX_TOKENS` and a body that may well parse — both are turned
into errors rather than being read as a proposal.

## Known limits

- **Spot FX is not available.** The executor's `place_order` dispatches `sec_type == "FUT"`
  to a futures contract and *everything else to a stock*, so a `CASH`/IDEALPRO leg would be
  sent as a stock named "EUR". `get_forex_contract` exists there but is never called. FX
  exposure here is therefore via currency ETFs (UUP, UDN, FXE, FXY, FXB, FXF, FXA, FXC, CEW),
  which price and size like any other name. Real spot FX needs three dispatch sites in the
  executor changed, plus its netting and risk manager taught about `CASH`.
- **Futures are not used**, so nothing here needs `multiplier` or `resolve_front`. If futures
  legs are ever added, every one needs `instrument.multiplier` or its notional is understated
  by the multiplier and sails past the allocation cap.
- **Nothing here schedules anything.** The sleeve only moves when you tell it to. `/sync` on
  a cron is worth adding if you want drift healed automatically.
- **Duration and beta are approximations** and they drift with yields and regime. They are
  there to get a hedge ratio into the right neighbourhood, not to be precise; the ratio they
  produce is shown to you before anything is sent.
