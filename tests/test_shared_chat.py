"""tests/test_shared_chat.py — behaving in a chat that also holds the executor's control bot.

Two bots in one group is fine — each has its own getUpdates queue keyed by its own token —
but Telegram delivers every slash command to BOTH of them, and the failures that causes are
all the quiet kind:

  * both bots implement /confirm, and both issue `secrets.token_hex(2)`. Identical four-hex
    tokens are indistinguishable, so without a prefix every confirmation reaches both bots
    and the one that does not own the token answers "nothing pending with that token". That
    is not just noise — it trains you to ignore that message on the day a /kill confirmation
    really has expired;
  * an unrecognised command must not be answered. /status and /kill belong to the other bot;
    two bots each complaining about every command the other owns makes the chat unusable;
  * `/cmd@TheOtherBot` is explicitly not for us. Telegram delivers it anyway, so the suffix
    has to be honoured rather than stripped.

None of this affects correctness of a trade — it affects whether the chat stays readable
enough that you notice when something IS wrong.
"""
import pytest

from agentic_macro import bot, config

#: Captured at import, before the autouse `quiet` fixture replaces it. The delivery tests
#: below exercise the real say(); everything above only cares about what it would have said.
REAL_SAY = bot.say


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """Capture what the bot would say instead of sending it."""
    said = []
    monkeypatch.setattr(bot, "say", lambda text, thread=None: said.append(text))
    monkeypatch.setattr(config, "allowed_users", lambda: {42})
    return said


def message(text, user_id=42):
    return {"text": text, "from": {"id": user_id, "username": "sjegoh"},
            "chat": {"id": config.CHAT_ID}}


# --------------------------------------------------------------------------- tokens
def test_our_tokens_are_recognisable_as_ours():
    token = bot.new_token()
    assert token.startswith(config.TOKEN_PREFIX)
    assert bot.is_ours(token)


def test_a_bare_four_hex_token_is_not_ours():
    """The exact shape the executor's control bot issues. If this ever read as ours, we
    would start answering its confirmations."""
    assert not bot.is_ours("a3f9")
    assert not bot.is_ours("")
    assert not bot.is_ours(None)


def test_confirming_another_bots_token_says_nothing(quiet):
    """The specific message that must not appear: "nothing pending with that token"."""
    bot.handle(message("/confirm a3f9"))
    assert quiet == [], f"answered a token that was not ours: {quiet}"


def test_confirming_an_unknown_token_of_OURS_still_reports_it(quiet):
    """Silence is only for other bots' tokens. A genuinely expired token of ours is a real
    error and has to keep saying so."""
    bot.handle(message(f"/confirm {config.TOKEN_PREFIX}dead"))
    assert quiet and "nothing pending" in quiet[0]


# --------------------------------------------------------------------------- commands
def test_an_unknown_command_is_ignored_rather_than_answered(quiet):
    """/status and /kill are the executor bot's. It answers unknown commands, so a real typo
    still gets feedback — from the bot that owns the namespace."""
    for text in ("/status", "/kill", "/positions", "/allocate ovn_volsurge 150k"):
        bot.handle(message(text))
    assert quiet == [], f"answered commands belonging to the other bot: {quiet}"


def test_our_own_commands_still_answer(quiet):
    bot.handle(message("/help"))
    assert quiet and "/worldview" in quiet[0]


def test_a_command_addressed_to_another_bot_is_ignored(quiet, monkeypatch):
    monkeypatch.setattr(bot, "BOT_USERNAME", "AgenticMacroBot")
    bot.handle(message("/help@AlgoExecutorBot"))
    assert quiet == []


def test_a_command_addressed_to_us_is_handled(quiet, monkeypatch):
    monkeypatch.setattr(bot, "BOT_USERNAME", "AgenticMacroBot")
    bot.handle(message("/help@agenticmacrobot"))       # Telegram is case-insensitive here
    assert quiet and "/worldview" in quiet[0]


def test_addressing_is_not_filtered_when_our_username_is_unknown(quiet, monkeypatch):
    """Failing open on the FILTER costs noise, never correctness — better than going mute
    because getMe happened to fail at startup."""
    monkeypatch.setattr(bot, "BOT_USERNAME", None)
    bot.handle(message("/help@AnyoneAtAll"))
    assert quiet and "/worldview" in quiet[0]


# --------------------------------------------------------------------------- allowlist
def test_a_non_allowlisted_user_cannot_trade(quiet):
    bot.handle(message("/worldview the dollar tops out here", user_id=999))
    assert quiet and "not allow-listed" in quiet[0]


def test_a_non_allowlisted_user_gets_silence_on_another_bots_command(quiet):
    """The allowlist check must not become a way to make the bot chatty about commands it
    does not own."""
    bot.handle(message("/kill", user_id=999))
    assert quiet == []


# --------------------------------------------------------------------------- delivery
class FakeTelegram:
    """Records sendMessage calls and fails those aimed at a topic that does not exist."""

    def __init__(self, bad_threads=(2,)):
        self.bad = set(bad_threads)
        self.sent = []

    def __call__(self, method, **payload):
        if method != "sendMessage":
            return True, {}
        thread = payload.get("message_thread_id")
        if thread in self.bad:
            return False, "Bad Request: message thread not found"
        self.sent.append((thread, payload["text"]))
        return True, {"message_id": len(self.sent)}


def test_a_bad_topic_falls_back_to_the_main_thread(monkeypatch):
    """The failure this pins is the bot going completely MUTE: every send fails on a topic
    that does not exist while each handler logs that it ran, so the logs insist it works.
    A reply in the wrong topic is recoverable; a reply you never see is not."""
    fake = FakeTelegram(bad_threads={2})
    monkeypatch.setattr(bot, "tg_call", fake)
    monkeypatch.setattr(config, "THREAD", "2")

    REAL_SAY("hello")
    assert fake.sent, "the reply was dropped instead of falling back"
    assert fake.sent[0][0] is None, "expected the fallback to be the main thread"
    assert fake.sent[0][1] == "hello"


def test_a_good_topic_is_used_as_configured(monkeypatch):
    fake = FakeTelegram(bad_threads={2})
    monkeypatch.setattr(bot, "tg_call", fake)
    monkeypatch.setattr(config, "THREAD", "4")

    REAL_SAY("hello")
    assert fake.sent == [(4, "hello")]


def test_the_fallback_sticks_for_later_chunks(monkeypatch):
    """A long proposal is split into several sends. Re-trying the dead topic on every chunk
    would double the API calls and interleave the message."""
    fake = FakeTelegram(bad_threads={2})
    monkeypatch.setattr(bot, "tg_call", fake)
    monkeypatch.setattr(config, "THREAD", "2")

    REAL_SAY("x" * 9000)
    assert len(fake.sent) == 3
    assert all(thread is None for thread, _ in fake.sent)
