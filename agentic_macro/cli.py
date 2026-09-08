"""
agentic_macro/cli.py — the same sleeve, driven from a terminal instead of Telegram.

Exists so every part of this can be exercised before a bot token does. `propose --dry-run`
in particular touches neither the executor nor the store: it calls the model, prices the
legs, and prints exactly what Telegram would show you, which is the fastest way to see
whether a worldview produces a structure worth approving.

    python -m agentic_macro.cli propose --dry-run "the Fed cuts three times into an intact labour market"
    python -m agentic_macro.cli views
    python -m agentic_macro.cli book
    python -m agentic_macro.cli sync --dry-run

Exit codes match the executor's convention, so this is safe to put in cron:
0 done or deliberately skipped, 1 refused, 2 the executor was unreachable and NOTHING
was sent.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import config, playbooks, prices, proposer
from .executor.executor_client import ExecutorClient, ExecutorRejected, ExecutorUnreachable
from .store import Store

log = logging.getLogger("agentic-macro.cli")


def _propose(args, store: Store) -> int:
    if args.dry_run:
        allocation = float(config.__dict__.get("DRY_RUN_CAPITAL", 0)) or 100_000.0
        views, used = store.active(), 0.0
    else:
        client = ExecutorClient(strategy_id=config.STRATEGY_ID)
        client.preflight()
        allocation = float(client.allocation()["capital_allocation"])
        views = store.active()
        held = store.net_positions()
        used = sum(v.gross(prices.fetch(held.keys()) if held else {}) for v in views)

    raw = proposer.propose(args.thesis, views, allocation, used)
    budget = min(args.size or proposer.default_budget(allocation, raw.get("confidence", "medium")),
                 allocation * config.MAX_WORLDVIEW_WEIGHT,
                 allocation * config.MAX_GROSS_WEIGHT - used)

    symbols = [leg["symbol"] for leg in (raw.get("legs") or [])]
    proposal = proposer.size(args.thesis, raw, prices.fetch(symbols) if symbols else {}, budget)

    from .bot import _render
    print(_render(proposal, allocation, used, token="<cli>"))

    if args.dry_run or not proposal.legs:
        return 0
    if input("\napprove and place these orders? [y/N] ").strip().lower() != "y":
        print("nothing submitted")
        return 0

    view = store.add(proposal.thesis, proposal.legs, reasoning=proposal.reasoning,
                     created_by="cli")
    from .strategy import submit
    code = submit(store=store)
    print(f"#{view.id} saved" + ("" if code == 0 else f", but the book failed (exit {code})"))
    return code


def _views(args, store: Store) -> int:
    from .bot import cmd_views
    print(cmd_views([], {}))
    return 0


def _book(args, store: Store) -> int:
    from .bot import cmd_book
    print(cmd_book([], {}))
    return 0


def _close(args, store: Store) -> int:
    view = store.close_worldview(args.id, closed_by="cli")
    print(f"#{view.id} closed — its legs unwind on the next book")
    from .strategy import submit
    return submit(store=store)


def _sync(args, store: Store) -> int:
    from .strategy import submit
    return submit(store=store, dry_run=args.dry_run)


def _diagnose(args, store: Store) -> int:
    """Why is the bot not seeing my messages?

    Peeks at the pending update queue WITHOUT advancing the offset, so nothing is consumed
    and a running bot still gets its updates. Run it with the bot stopped for the clearest
    picture: anything you have sent since it stopped should be sitting in the queue, and a
    topic whose messages are absent is a topic the bot is not receiving at all — which is a
    permissions question, not a bug in the poll loop."""
    from . import bot

    me = bot.tg("getMe") or {}
    chat = bot.tg("getChat", chat_id=config.CHAT_ID) or {}
    member = bot.tg("getChatMember", chat_id=config.CHAT_ID, user_id=me.get("id")) or {}

    print(f"bot        @{me.get('username')} (id {me.get('id')})")
    print(f"privacy    {'DISABLED — all messages arrive' if me.get('can_read_all_group_messages') else 'ENABLED — only /commands, @mentions and replies to the bot arrive'}")
    print(f"chat       {chat.get('title')!r}  type={chat.get('type')}  is_forum={chat.get('is_forum')}")
    print(f"membership {member.get('status')}")
    if chat.get("is_forum") and member.get("status") not in ("administrator", "creator"):
        print("\n  ! This is a forum group and the bot is only a member. Promote it to\n"
              "    administrator to receive messages in topics other than General.\n"
              "    A privacy-mode change also needs the bot REMOVED and RE-ADDED to apply.")

    print("\npending updates (nothing consumed):")
    updates = bot.tg("getUpdates", timeout=0)
    if not updates:
        print("   (queue empty — send a message in the problem topic, then re-run)")
    for u in updates or []:
        msg = u.get("message") or u.get("edited_message") or {}
        print(f"   {u['update_id']}  thread={msg.get('message_thread_id')}  "
              f"from={(msg.get('from') or {}).get('username')}  "
              f"text={(msg.get('text') or '')[:50]!r}")
    return 0


def _playbooks(args, store: Store) -> int:
    print(playbooks.catalogue(args.thesis or ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_macro", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="turn a worldview into a proposed structure")
    p.add_argument("thesis")
    p.add_argument("--size", type=float, default=None, help="budget in dollars")
    p.add_argument("--dry-run", action="store_true",
                   help="no executor, no store — just the model and prices")
    p.set_defaults(func=_propose)

    p = sub.add_parser("views", help="the worldviews currently on")
    p.set_defaults(func=_views)

    p = sub.add_parser("book", help="the netted book across active worldviews")
    p.set_defaults(func=_book)

    p = sub.add_parser("close", help="retire a worldview and unwind its legs")
    p.add_argument("id", type=int)
    p.set_defaults(func=_close)

    p = sub.add_parser("sync", help="re-send the netted book")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_sync)

    p = sub.add_parser("diagnose", help="why is the bot not seeing my messages?")
    p.set_defaults(func=_diagnose)

    p = sub.add_parser("playbooks", help="the structures available")
    p.add_argument("thesis", nargs="?", help="rank them against a view")
    p.set_defaults(func=_playbooks)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    store = Store()
    try:
        return args.func(args, store)
    except ExecutorUnreachable as e:
        # The client has already alerted Telegram. Non-zero so cron sees it: nothing went in.
        log.critical("ABORTED — %s", e)
        return 2
    except (ExecutorRejected, proposer.ProposalError, prices.PriceUnavailable, ValueError) as e:
        log.error("%s", e)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
