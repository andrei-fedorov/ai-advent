# TooManyRules — сущность агента (день 6, неделя 2; день 8 — работа с токенами,
# день 9 — управление контекстом, день 10 — общий служебный вызов, checkpoint'ы
# и ветки диалога).
#
# Единственное место в проекте, где происходят вызовы LLM API. Модуль
# намеренно ничего не знает ни про Gradio, ни про Too Many Bones: внутри
# только конфиг, стек сообщений, вызов API, замер времени, подсчёт токенов
# и стоимости, логирование и обработка ошибок. Наружу агент отдаёт результат
# (`AgentReply`) и своё состояние для дебага (`debug_state`).
#
# Направление зависимостей одностороннее: app.py → presets.py → agent.py →
# tokens.py, context.py. Из проекта здесь импортируются только листья графа,
# которые сами не импортируют ничего: `tokens.py` (день 8) и `context.py`
# (день 9); замороженный `app_week1.py` не импортируется по-прежнему — см.
# `docs/TooManyRules — Неделя 2 архитектура.md`, §3, и спецификации дня 8, §4,
# дня 9, §4, и дня 10, §5.
#
# С дня 9 вызовов LLM API здесь два: основной (ответ игроку) и служебный —
# до дня 10 он был «свёрткой», а с дня 10 обслуживает любую стратегию с
# памятью (сводку и факты) одним и тем же путём, а слова для лога и панели
# берёт у стратегии. Оба вызова — в этом модуле: правило «вызовы LLM API
# только в `agent.py`» не нарушено.
#
# День 10 также добавляет операцию над самой историей — checkpoint и ветку:
# агент фиксирует точку диалога вместе со снимком памяти стратегий и порождает
# от неё нового агента с общим началом истории. Это не стратегия и не вызов
# API — обычная работа со стеком сообщений и с реестром агентов.

import copy
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

import context
import tokens

# Ключ читается только здесь, поэтому и .env подхватывается здесь же —
# интерфейс про API-ключи ничего не знает.
load_dotenv()

logger = logging.getLogger("toomanyrules.agent")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Дешёвая модель на каждый день — дефолт конфига. С 11.09.2026 это
# `deepseek-flash` (DeepSeek-V4.1-Flash): прежняя `deepseek-v4-flash`
# выведена из эксплуатации. Имя модели — ключ и в таблице цен ниже, и в
# таблице контекстных окон `tokens.py`: переименовали модель — правьте обе,
# иначе стоимость и бюджет молча станут «н/д».
DEEPSEEK_MODEL = "deepseek-flash"
# Флагман: сильнее и дороже, используется пресетом «Флагман + thinking».
DEEPSEEK_MODEL_PRO = "deepseek-v4-pro"

# Цены в долларах за 1M токенов (off-peak), ключ — имя модели. Источник:
# https://api-docs.deepseek.com/quick_start/pricing, сверено 11.09.2026.
# В пиковые часы (01:00-04:00 и 06:00-10:00 UTC, пн-пт) ставки ×2 — считаем по
# off-peak, это осознанно оценка, а не выписка по счёту. Цена входа взята по
# cache miss: кэш промпта дешевле в разы ($0.003/1M у flash), но в оценку он
# не заводится — поля кэша показываются в панели отдельно (день 8, §10).
#
# 11.09.2026 дешёвая модель переименована: `deepseek-v4-flash` выведена из
# эксплуатации, вместо неё `deepseek-flash` (DeepSeek-V4.1-Flash) и цены
# 0.22/0.66 → 0.15/0.60. Старое имя API ещё принимает и обслуживает той же
# моделью по цене flash, но в таблице его нет намеренно: платим за то, что
# реально отвечает, а мёртвый алиас в проекте не используется. Замороженный
# `app_week1.py` со своей копией таблицы остаётся на старом имени и старых
# ценах — это цена заморозки, а не рассинхрон, который надо чинить.
PRICING_PER_M_TOKENS = {
    "deepseek-flash":  {"input": 0.15, "output": 0.60},
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
}

_NO_API_KEY_ERROR = (
    "Не задан DEEPSEEK_API_KEY. Положите ключ в файл src/.env строкой "
    "DEEPSEEK_API_KEY=... (ключ берётся на https://platform.deepseek.com) "
    "и перезапустите приложение."
)

# Признаки переполнения контекста в тексте ошибки API (день 8). Ищутся без
# учёта регистра; у DeepSeek это ошибка с кодом 400 и текстом вида «This
# model's maximum context length is 1048576 tokens. However, you requested
# 1158576 tokens» (проверено живым вызовом 10.09.2026). Список — константа, а
# не регулярка в коде: провайдеры формулируют по-разному, и дополнить его по
# факту должно быть одной строкой.
CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "maximum context",
    "context_length_exceeded",
    "too long",
)


def estimate_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Оценка стоимости одного вызова в долларах по `PRICING_PER_M_TOKENS`.

    Возвращает `None`, если токены неизвестны (API не вернул `usage`) или
    модели нет в таблице цен — на неделе 6 через `AgentConfig.base_url`
    появится локальная модель, для которой цены в долларах бессмысленны.
    При `thinking=on` `completion_tokens` уже включает токены рассуждения,
    отдельного учёта не требуется.
    """
    rates = PRICING_PER_M_TOKENS.get(model)
    if rates is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens * rates["input"] / 1_000_000
        + completion_tokens * rates["output"] / 1_000_000
    )


@dataclass(frozen=True)
class AgentConfig:
    """Всё, что на неделе 1 было разными кусками кода: системный промпт,
    модель, режим рассуждения, температура, лимиты. Другой набор параметров —
    это другой агент (`dataclasses.replace(config, ...)`), а не аргумент
    метода `ask()`."""

    name: str                          # имя агента: попадает в логи и дебаг-панель
    system_prompt: str
    description: str = ""              # для выпадающего списка пресетов в интерфейсе
    model: str = DEEPSEEK_MODEL
    thinking: bool = False             # → extra_body={"thinking": {"type": ...}}
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    keep_history: bool = True          # ведёт ли агент стек сообщений между вызовами
    base_url: str = DEEPSEEK_BASE_URL  # задел на неделю 6 (локальная модель)

    def __post_init__(self) -> None:
        # DeepSeek не поддерживает temperature/top_p в режиме рассуждения
        # (ограничение зафиксировано ещё в спецификации дня 5). Ловим
        # невалидную комбинацию на создании конфига, а не на вызове API.
        if self.thinking and (self.temperature is not None or self.top_p is not None):
            raise ValueError(
                f"Агент «{self.name}»: thinking=True несовместим с temperature/top_p — "
                "DeepSeek не поддерживает их в режиме рассуждения. "
                "Уберите temperature/top_p или выключите thinking."
            )


@dataclass(frozen=True)
class ServiceCall:
    """Служебный вызов модели, сделанный стратегией до основного запроса
    (день 9).

    Это настоящий вызов: он считается в счётчиках агента и процесса наравне с
    обычными и логируется так же. Но это не ход — он не пишет в стек
    сообщений, не создаёт строку журнала ходов и не затирает `_last_reply`:
    в блоке «Последний вызов» должен оставаться ответ игроку, а не сводка.
    """

    kind: str                  # из ContextTask
    label: str
    ok: bool
    error: str | None
    elapsed: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    # Сколько сообщений свёрнуто этим вызовом — количество, а не граница, как
    # `ContextTask.covers`: панели нужно «6 сообщ. уехали в сводку», а не
    # индекс, до которого теперь покрывает сводка.
    covers: int
    folded_tokens: int         # оценка того, сколько эти сообщения весили
    text: str                  # что получилось — для панели
    # В конце и с умолчанием, как поля дня 8 в `AgentReply`. "length" — ответ
    # упёрлась в `max_tokens` и оборван на полуслове: он всё равно применён,
    # но панель и лог говорят об этом вслух (спецификация дня 9, §6.2).
    finish_reason: str | None = None
    # Поля дня 10 — тоже в конце и с умолчанием: как стратегия называет свою
    # память («Сводка», «Факты») и чем отчиталось последнее применённое
    # обновление (`StrategyState.last_update`). Панель показывает служебный
    # вызов по ним и по `label`, не зная, какая стратегия его сделала.
    memory_label: str = ""
    memory_update: str = ""


@dataclass(frozen=True)
class AgentReply:
    """Результат одного вызова `ask()`. Полный набор метрик на каждый ход —
    это содержимое дебаг-панели (и причина, по которой день 6 обходится без
    стриминга)."""

    ok: bool
    text: str
    error: str | None
    finish_reason: str | None
    elapsed: float                     # секунды, замер вокруг вызова API
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    reasoning: str | None              # reasoning_content, если модель его вернула
    model: str
    agent_name: str
    # Поля дня 8 идут в конце и со значением по умолчанию: `AgentReply`
    # создаётся по ключевым аргументам в двух местах (`ask()` и
    # `_failed_reply()`), и порядок существующих полей не должен меняться.
    #
    # Оценка того, что ушло в модель, — по результату `_build_messages()`, то
    # есть по тому же списку сообщений, который отправлен в API. Рядом с ней
    # в панели всегда стоит факт из `usage`: до запроса точного числа не
    # бывает, и в этом весь смысл дня.
    request_tokens: "tokens.RequestTokens | None" = None
    # Оценка по тексту ответа — вторая, более чистая проверка эвристики:
    # у ответа нет служебной разметки сообщений.
    estimated_completion_tokens: int | None = None
    # Кэш промпта: DeepSeek возвращает их в `usage`, другие провайдеры могут
    # не возвращать — тогда остаются `None`. Стоимость по ним сегодня не
    # пересчитывается (спецификация дня 8, §10), но показать их честно нужно:
    # они объясняют, почему счёт растёт медленнее нашей оценки.
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    # Поле дня 9 — тоже в конце и с умолчанием. `None` означает «стратегия
    # служебного вызова не просила», а не «свёртка не удалась»: неудачная
    # свёртка приезжает сюда объектом с `ok=False`.
    service_call: ServiceCall | None = None


@dataclass(frozen=True)
class TurnStats:
    """Одна строка журнала ходов агента: что стоил ход и во что он обошёлся
    накопительно. За один вызов роста токенов не видно в принципе — журнал
    и есть та самая кривая, ради которой затевался день 8."""

    turn: int                        # номер успешного хода агента, с 1
    history_before: int              # сколько сообщений было в стеке до хода
    estimated_prompt_tokens: int     # оценка до запроса, без калибровки
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    cumulative_total_tokens: int
    cumulative_cost_usd: float
    # Поля дня 9 — в конце и с умолчаниями, как поля дня 8 в `AgentReply`.
    # Ради них журнал и заводился: в одной строке видно, что ушло в модель,
    # что ушло бы без сжатия и сколько стоила свёртка на этом ходе.
    strategy: str = ""
    sent_messages: int = 0          # сколько сообщений истории ушло в модель
    full_prompt_tokens: int = 0     # оценка «сколько ушло бы без сжатия»
    service_tokens: int = 0         # служебные токены, потраченные на этом ходе
    service_cost_usd: float | None = None
    # Поля дня 10 — в конце: удобство для пользователя — один из пунктов
    # сравнения стратегий, и «факты делают ход двумя вызовами» должно быть
    # видно числом, а не на слово.
    elapsed: float = 0.0            # время основного вызова
    service_elapsed: float = 0.0    # время служебного вызова на этом ходе


@dataclass(frozen=True)
class ContextView:
    """Что уйдёт в модель на следующем ходе и чего это стоит по сравнению с
    полным контекстом.

    Считается один раз и отдаётся и панели, и логам: иначе «сколько уходит» и
    «сколько сэкономлено» считались бы в двух местах и разъехались бы.
    """

    strategy: str
    state: context.StrategyState
    usage: tokens.ContextUsage     # то, что уйдёт (со сжатием)
    full: tokens.RequestTokens     # то, что ушло бы без сжатия
    sent_messages: int             # сообщений истории в запросе
    history_messages: int          # сообщений в стеке
    saved_tokens: int              # full.total - usage.estimated, может быть < 0
    pending_label: str             # «свёртка сообщений …», "" — задачи нет
    pending_messages: int
    # Поле дня 10 — в конце: `ContextTask.sends` назревшей задачи (сколько
    # сообщений истории уйдёт в запрос, когда она применится); 0, если задачи
    # нет. Статус при переключении стратегии берёт число отсюда, а не считает
    # его сам, — правило это знает только стратегия (спецификация дня 10, §4.1).
    pending_sent_messages: int = 0


@dataclass(frozen=True)
class Checkpoint:
    """Зафиксированная точка текущей сессии (день 10): сколько сообщений в
    истории, какая стратегия активна и снимок памяти всех стратегий на этот
    момент. Сами сообщения не копируются — история от checkpoint'а не
    укорачивается, поэтому «первые 12 сообщений» остаются первыми двенадцатью
    навсегда. Ставится только в конце истории: снимок памяти согласован с
    историей лишь в этой точке (спецификация дня 10, §2.3)."""

    id: str             # "cp1", "cp2", … в пределах сессии
    messages: int        # длина истории в точке сохранения, > 0
    strategy: str        # активная стратегия в этой точке
    memory: dict          # снимок памяти стратегий — тот же вид, что context["memory"]
    created_at: str      # время сохранения, ISO до секунд


@dataclass(frozen=True)
class BranchOrigin:
    """Происхождение ветки: от какого checkpoint'а какой сессии она создана.
    Ссылка историческая — родителя могли сбросить или удалить, и ветка живёт
    дальше как самостоятельный диалог (спецификация дня 10, §2.3)."""

    parent: str          # session_id сессии, от checkpoint'а которой создана ветка
    checkpoint: str      # id этого checkpoint'а
    messages: int         # длина общего префикса
    created_at: str      # время этого checkpoint'а: id после сброса родителя
                          # повторяются, время — нет


class HistoryStore(Protocol):
    """Сохранение истории между запусками (реализация — `storage.py`, день 7).

    Протокол объявлен на дне 6, когда реализаций в проекте ещё не было, —
    чтобы день 7 подставил в эту точку `JsonHistoryStore`, не переписывая
    агента. Туда же, когда появится VPS, встанет SQLite. Агент по-прежнему
    работает и со `store=None` — тогда он живёт только в памяти процесса.

    День 9 дополняет протокол минимально: у `save()` появился третий
    необязательный параметр `context` (память стратегий — для хранилища это
    непрозрачный словарь), и добавился `load_context()`. История пишется
    вместе с контекстом одной записью: разъехаться они не должны.
    """

    def load(self, session_id: str) -> list[dict]: ...
    def save(
        self,
        session_id: str,
        messages: list[dict],
        context: dict | None = None,
    ) -> None: ...
    def load_context(self, session_id: str) -> dict: ...


# --- Счётчики процесса и реестр агентов ----------------------------------
# Наглядный ответ на «один инстанс приложения — много агентов»: сколько
# агентов создано с момента старта, сколько они суммарно потратили и — в
# реестре — сами эти агенты, между которыми переключается интерфейс.
#
# Реестр держит агентов живыми, пока их оттуда не уберут. Без него агент,
# потерявший ссылку из интерфейса (например, после смены пресета), просто
# исчезает, и от «много агентов» остаётся один счётчик.
#
# Gradio обрабатывает запросы в пуле потоков, поэтому и счётчики, и реестр —
# под одним Lock.
_process_lock = threading.Lock()
_process_stats = {
    "agents_created": 0,
    "calls": 0,
    "errors": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}


_agents: list["Agent"] = []

# База для сравнения «без сжатия»: этой стратегией агент меряет, сколько ушло
# бы в модель, если бы контекстом никто не управлял. Экземпляр один на процесс
# и общий для всех агентов — у «Всей истории» нет памяти, делить ей нечего.
_FULL_HISTORY = context.FullHistory()


def process_stats() -> dict:
    """Копия счётчиков процесса (агенты, вызовы, токены, деньги).

    `agents_alive` не хранится, а считается по длине реестра: агентов можно
    удалять, и живых становится меньше, чем созданных. Назад счётчики не
    откатываются — потраченные токены и деньги остались потраченными.
    """
    with _process_lock:
        stats = dict(_process_stats)
        stats["agents_alive"] = len(_agents)
        return stats


def agents() -> list["Agent"]:
    """Все агенты процесса в порядке создания. Наружу отдаётся копия списка,
    сам реестр не уезжает."""
    with _process_lock:
        return list(_agents)


def agent_by_number(number: int | None) -> "Agent | None":
    """Агент по порядковому номеру или `None`, если такого номера в процессе
    нет — например, страница осталась открытой с прошлого запуска."""
    with _process_lock:
        for agent in _agents:
            if agent.number == number:
                return agent
    return None


def delete_agent(number: int) -> bool:
    """Убирает агента из реестра процесса; возвращает, был ли он там.

    Это снятие с учёта, а не гарантированное уничтожение: Python освободит
    агента, когда на него не останется ссылок (например, пока им занята
    вкладка браузера, он доживёт до её следующего действия). Номер удалённого
    агента не переиспользуется — иначе в логах два разных агента выглядели бы
    одинаково.
    """
    removed: "Agent | None" = None
    with _process_lock:
        for index, agent in enumerate(_agents):
            if agent.number == number:
                removed = _agents.pop(index)
                break
        alive = len(_agents)
    if removed is None:
        return False
    logger.info(
        "[%s] агент удалён из реестра; живых агентов: %d",
        removed._log_name, alive,
    )
    return True


def _family_root(agent: "Agent", by_session: dict[str, "Agent"]) -> str:
    """`session_id` корня семейства веток: поднимаемся по `branch.parent`,
    пока родитель есть в реестре. Последний известный `session_id` и есть
    ключ семейства — даже если такого агента в реестре уже нет (родителя
    сбросили или удалили). `visited` защищает от битой ссылки, где родитель
    ссылается сам на себя или на собственного потомка."""
    session = agent.session_id
    visited = {session}
    node: "Agent | None" = agent
    while node is not None and node.branch is not None:
        session = node.branch.parent
        if session in visited:
            break
        visited.add(session)
        node = by_session.get(session)
    return session


def branch_family(agent: "Agent") -> list["Agent"]:
    """Исходный диалог и все его ветки из реестра, в порядке создания
    (спецификация дня 10, §5.2). Агент без происхождения и без веток —
    семейство из одного."""
    registry = agents()
    by_session = {a.session_id: a for a in registry}
    root = _family_root(agent, by_session)
    return [a for a in registry if _family_root(a, by_session) == root]


def branch_parent(agent: "Agent") -> "Agent | None":
    """Родитель ветки, если связь с ним жива: агент с `session_id ==
    branch.parent` есть в реестре, и среди его checkpoint'ов есть checkpoint
    с тем же `id` и тем же `created_at` (спецификация дня 10, §5.2). Иначе
    `None` — родителя сбросили или удалили (после сброса у него новый диалог
    и, возможно, новый `cp1`, но уже с другим временем)."""
    branch = agent.branch
    if branch is None:
        return None
    for candidate in agents():
        if candidate.session_id != branch.parent:
            continue
        for checkpoint in candidate.checkpoints:
            if checkpoint.id == branch.checkpoint and checkpoint.created_at == branch.created_at:
                return candidate
        return None
    return None


def _register_agent(agent: "Agent") -> int:
    """Регистрирует созданного агента и возвращает его порядковый номер.
    Нумерация сквозная, с 1: номер — это значение счётчика `agents_created` на
    момент создания, поэтому после удалений номера в реестре идут с пропусками."""
    with _process_lock:
        _process_stats["agents_created"] += 1
        _agents.append(agent)
        return _process_stats["agents_created"]


def _bump_process_stats(**deltas) -> None:
    with _process_lock:
        for key, delta in deltas.items():
            _process_stats[key] += delta


class Agent:
    """Агент: конфиг + собственный стек сообщений + вызов LLM.

    Стек ведёт агент, а не интерфейс. Системный промпт лежит в конфиге и
    подставляется первым сообщением при каждом вызове, но в самом стеке его
    нет — в стеке только пары user/assistant.
    """

    def __init__(
        self,
        config: AgentConfig,
        session_id: str = "default",
        store: HistoryStore | None = None,
        strategies: dict[str, context.ContextStrategy] | None = None,
        strategy: str | None = None,
        # Начальное состояние ветки (день 10, §5.2): передаёт только `fork()`,
        # `app.py` агентов с `seed` не создаёт. Единственный способ собрать
        # ветку — регистрация в реестре должна остаться последним шагом
        # конструктора (день 6), а ветка, созданная обычным конструктором и
        # дописанная потом, успела бы побыть в реестре пустой.
        seed: dict | None = None,
    ) -> None:
        self._config = config
        self._session_id = session_id
        self._store = store
        self._client: OpenAI | None = None
        self._last_reply: AgentReply | None = None
        # Счётчики за время жизни агента. `reset()` их не трогает — он
        # чистит только диалог: стек сообщений и память стратегий.
        self._totals = {
            # Все вызовы модели: ходы `ask()`, включая неуспешные, и с дня 9 —
            # служебные (свёртки), они считаются наравне с обычными.
            "calls": 0,
            "errors": 0,         # из них закончившихся ошибкой
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            # Суммы для калибровки (день 8): факт и наша оценка по одним и тем
            # же запросам. Считаются только по вызовам, где пришёл `usage`, —
            # у остальных сравнивать не с чем, и попасть в коэффициент они не
            # должны. Отдельно от `prompt_tokens` выше именно поэтому.
            "calibration_calls": 0,
            "calibration_fact_tokens": 0,
            "calibration_estimated_tokens": 0,
            # Счётчики дня 9. Служебные вызовы считаются и в общих счётчиках
            # выше (те же токены, те же деньги), и отдельно здесь: рядом с
            # «сэкономлено» всегда должно стоять «потрачено на свёртки», иначе
            # день превращается в фокус.
            "service_calls": 0,
            "service_tokens": 0,
            "service_cost_usd": 0.0,
            # Сумма экономии по ходам: «сколько ушло бы без сжатия» минус
            # «сколько ушло». На коротком диалоге бывает отрицательной — так и
            # показываем.
            "saved_tokens": 0,
        }
        # Журнал ходов (день 8). Ведёт себя как счётчики агента, а не как стек
        # сообщений: `reset()` его не чистит, на диск он не едет, и после
        # перезапуска процесса он пуст, хотя история восстановлена из файла.
        self._turns: list[TurnStats] = []

        # Стратегия — третья зависимость агента после конфига и хранилища и,
        # как хранилище, необязательная: `strategies=None` даёт единственную
        # «Всю историю», и агент ведёт себя ровно как на дне 8. Экземпляры
        # свои у каждого агента (их создаёт `presets.make_strategies()`):
        # память сводки относится к конкретному диалогу.
        #
        # Стратегия — состояние агента, а не поле `AgentConfig`: конфиг
        # заморожен и описывает вызов модели, а стратегия описывает память и
        # переключается на живом агенте, не теряя диалог (день 9, §6.1).
        self._strategies: dict[str, context.ContextStrategy] = (
            dict(strategies)
            if strategies
            else {context.FULL_HISTORY_NAME: context.FullHistory()}
        )
        # Предупреждения, накопленные до регистрации в реестре: без номера
        # агента строка в логе ничего не опознаёт, поэтому они логируются
        # ниже, вместе со `store_error`.
        self._startup_warnings: list[str] = []
        self._strategy_name = next(iter(self._strategies))
        if strategy is not None:
            self._strategy_name = self._known_strategy(strategy)

        # Хранилище — необязательная зависимость: при `store=None` агент
        # работает целиком в памяти процесса (это и есть режим дня 6).
        self._messages: list[dict] = []
        self._store_error: str | None = None
        # Checkpoint'ы и происхождение ветки (день 10) — пустые до чтения
        # хранилища/`seed`: `_load_context()` заполняет их из блока `context`
        # тем же кодом, что и память стратегий.
        self._checkpoints: list[Checkpoint] = []
        self._branch: BranchOrigin | None = None
        # Запись выключается ровно в одном случае — если не удалось прочитать
        # свою историю: агент, не прочитавший файл, не должен его затирать.
        self._store_writable = store is not None

        if seed is not None:
            # Ветка (день 10, §5.2): начальное состояние приходит параметром,
            # а не из хранилища, но читается тем же кодом, которым агент
            # читает файл сессии — `_normalize_messages()` и `_load_context()`.
            # Хранилище на чтение не трогается: у ветки может ещё не быть
            # ничего своего на диске.
            self._messages = _normalize_messages(seed.get("messages") or [])
            try:
                self._load_context(seed.get("context") or {})
            except Exception as exc:
                self._startup_warnings.append(
                    f"память стратегий ветки не прочиталась, стартуем с "
                    f"пустой: {exc}"
                )
        elif store is not None and config.keep_history:
            try:
                self._messages = _normalize_messages(store.load(session_id))
            except Exception as exc:
                # Стартуем с пустым стеком; файл остаётся на диске как есть —
                # вдруг он ещё починится. Причина уедет в лог ниже, когда у
                # агента появится номер, и в дебаг-панель через store_error.
                self._store_writable = False
                self._store_error = (
                    f"история не загрузилась, запись выключена: {exc}"
                )
            # Память стратегий, checkpoint'ы, происхождение ветки и имя
            # активной стратегии — отдельным чтением и отдельным try: сбой
            # контекста не должен отменять чтение истории. Не прочиталось —
            # стартуем с пустой памятью и предупреждением в лог, файл при
            # этом не трогаем.
            try:
                self._load_context(store.load_context(session_id))
            except Exception as exc:
                self._startup_warnings.append(
                    f"память стратегий не прочиталась, стартуем с пустой: {exc}"
                )
        # Сколько сообщений пришло из хранилища: после первого же хода
        # `len(history)` растёт, и «сколько было восстановлено» иначе не
        # показать. У ветки — всегда 0: из хранилища ей ничего не пришло,
        # сколько сообщений скопировано из checkpoint'а, говорит `branch.messages`.
        self._restored_messages = 0 if seed is not None else len(self._messages)

        # Регистрация в реестре — последним шагом конструктора, когда все
        # поля уже проставлены: наружу не должен попасть недособранный агент.
        self._number = _register_agent(self)
        # Префикс логов: с несколькими агентами одного пресета одно только имя
        # ничего не различает, а с дня 7 у каждого агента ещё и свой файл —
        # поэтому в префиксе и номер, и сессия.
        self._log_name = f"#{self._number} {config.name} · {session_id}"
        if seed is not None and self._branch is not None:
            logger.info(
                "[%s] агент создан как ветка от %s · %s: общий префикс %d "
                "сообщ., стратегия «%s», память стратегий из снимка "
                "checkpoint'а; живых агентов: %d",
                self._log_name, self._branch.parent, self._branch.checkpoint,
                self._branch.messages, self._strategy_name,
                process_stats()["agents_alive"],
            )
        else:
            logger.info(
                "[%s] агент создан: model=%s thinking=%s стратегия=«%s» "
                "восстановлено из хранилища=%d сообщ.%s, живых агентов: %d",
                self._log_name, config.model, _thinking_label(config.thinking),
                self._strategy_name, self._restored_messages,
                _memory_note(self.strategy.describe(self._messages)),
                process_stats()["agents_alive"],
            )
        # Сбой загрузки логируется здесь, а не на месте: до регистрации в
        # реестре у агента ещё нет номера, а без номера строка в логе
        # ничего не опознаёт.
        if self._store_error is not None:
            logger.warning("[%s] %s", self._log_name, self._store_error)
        for warning in self._startup_warnings:
            logger.warning("[%s] %s", self._log_name, warning)

        if seed is not None:
            # История у ветки уже есть и должна пережить перезапуск до
            # первого вопроса — файл пишется сразу, а не после первого
            # успешного ответа (спецификация дня 10, §5.2).
            self._persist()

    # --- Публичный контракт ---------------------------------------------

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def number(self) -> int:
        """Порядковый номер агента в процессе (с 1). Им агент опознаётся в
        переключателе дебаг-панели и в логах."""
        return self._number

    @property
    def strategy(self) -> context.ContextStrategy:
        """Активная стратегия сборки запроса."""
        return self._strategies[self._strategy_name]

    def strategy_names(self) -> list[str]:
        """Имена стратегий, между которыми умеет переключаться этот агент."""
        return list(self._strategies)

    def set_strategy(self, name: str) -> bool:
        """Переключение стратегии на живом агенте; возвращает, произошло ли оно.

        Стек сообщений, файл истории и счётчики не трогаются: меняется способ
        сборки запроса, а не агент и не диалог. Память неактивных стратегий
        при этом не чистится — она привязана к `covered`, индексу в историю,
        поэтому «отстать» не может: вернувшись к сводке после десяти ходов под
        другой стратегией, агент свернёт накопившееся одной свёрткой.
        """
        if name not in self._strategies or name == self._strategy_name:
            return False
        previous = self._strategy_name
        self._strategy_name = name
        logger.info(
            "[%s] стратегия «%s» → «%s»: %s; стек не тронут (%d сообщ.), "
            "история на диске полная",
            self._log_name, previous, name,
            self.strategy.describe(self._messages).note, len(self._messages),
        )
        # В файле меняется активная стратегия — иначе перезапуск вернул бы
        # предыдущую.
        self._persist()
        return True

    @property
    def history(self) -> list[dict]:
        """Копия стека сообщений — только user/assistant, без системного
        промпта и без reasoning_content."""
        return [dict(message) for message in self._messages]

    @property
    def restored_messages(self) -> int:
        """Сколько сообщений пришло из хранилища при создании агента. Ноль —
        и когда хранилища нет, и когда файла сессии ещё не было."""
        return self._restored_messages

    @property
    def checkpoints(self) -> list[Checkpoint]:
        """Checkpoint'ы этой сессии, по порядку сохранения. Копия списка, как
        `history` — наружу сам список не уезжает."""
        return list(self._checkpoints)

    @property
    def branch(self) -> BranchOrigin | None:
        """Происхождение этой сессии, если она ветка. `None` — это не ветка,
        а не «связь с родителем потеряна»: то, жива ли связь, знает только
        `branch_parent()`, у него для этого есть реестр агентов."""
        return self._branch

    @property
    def turns(self) -> list[TurnStats]:
        """Копия журнала ходов — как `history`, наружу список не уезжает.

        Запись появляется только на успешный вызов: у неуспешного нет `usage`,
        и точка на графике из него была бы выдумкой.
        """
        return list(self._turns)

    @property
    def calibration(self) -> float | None:
        """Отношение суммы фактических `prompt_tokens` к сумме собственных
        оценок тех же запросов за время жизни агента.

        `None`, пока успешных вызовов с `usage` не было, — тогда оценка
        показывается сырой. Дальше коэффициент применяется к оценке бюджета
        множителем и уточняется сам с каждым ходом: у нас есть то, чего нет у
        токенизатора, — факт по собственному трафику.
        """
        estimated = self._totals["calibration_estimated_tokens"]
        if not self._totals["calibration_calls"] or estimated <= 0:
            return None
        return self._totals["calibration_fact_tokens"] / estimated

    def context_usage(self, question: str = "") -> tokens.ContextUsage:
        """Бюджет контекста текущего стека: сколько из окна модели уже занято.

        Без аргумента считается стек как есть (системный промпт, память
        стратегии и несвёрнутая история) — это нижняя граница следующего
        запроса. С вопросом — то, что уйдёт в модель, если отправить его
        прямо сейчас; так панель показывает занятость до нажатия «Отправить».

        Считается по результату `_build_messages()` — по тому же списку
        сообщений, который ушёл бы в API: «посчитали» и «отправили» не должны
        разъезжаться. С дня 9 это, значит, **сжатый** запрос; «сколько было бы
        без сжатия» лежит рядом, в `ContextView.full`.
        """
        return self.context_view(question).usage

    def context_view(self, question: str = "") -> ContextView:
        """Что уйдёт в модель на следующем ходе и чего это стоит по сравнению
        с полным контекстом (день 9). Чистый расчёт: ни сети, ни изменения
        памяти — панель зовёт его на каждый свой рендер."""
        return self._context_view(question)[1]

    def ask(self, user_message: str) -> AgentReply:
        """Один ход диалога: подготовить контекст, собрать сообщения, сходить
        в API, дописать пару «вопрос/ответ» в стек.

        С дня 9 фаз две: сначала стратегии дают возможность попросить
        служебный вызов (свёртку), и только потом собирается и уходит основной
        запрос. Служебный вызов ходом не является и через `ask()` не идёт.

        Исключений не бросает: ошибка API или отсутствующий ключ возвращаются
        как `AgentReply(ok=False, error=...)` — в том числе при сбое
        служебного вызова. При ошибке стек не меняется — ни вопрос, ни ответ
        в него не попадают, чтобы повтор не задваивал историю.

        Параметры запроса берутся только из конфига: аргументов, меняющих
        модель/температуру/лимиты, у `ask()` нет — другой набор параметров
        означает другой экземпляр агента.
        """
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return self._failed_reply(str(exc), elapsed=0.0)

        # Фаза подготовки контекста (день 9). Стратегия может попросить
        # служебный вызов — свернуть старую часть диалога в сводку. Сбой
        # свёртки хода не отменяет: память не двигается, запрос собирается тем,
        # что есть, и деградация идёт в сторону «без сжатия», а не в сторону
        # потерянного контекста. Специального кода это не требует — достаточно
        # того, что `covered` не сдвинулся.
        service = self._run_context_task(client, user_message)

        # Счёт до запроса (день 8): считаем ровно тот список сообщений, который
        # сейчас уйдёт в API, — и логируем бюджет до вызова, а не после.
        # Сборка и расчёт идут одним вызовом, чтобы «что отправляем» и «что
        # показываем в панели» не считались двумя путями.
        messages, view = self._context_view(user_message)
        request = view.usage.request
        budget = view.usage
        self._log_context(view)
        self._log_budget(budget)
        # Проверки «а влезет ли» здесь намеренно нет: переполненный запрос
        # отправляется как есть. День 8 показывает поломку, день 9 даёт способ
        # до неё не доходить — но предохранителем не становится (спецификация
        # дня 9, §12): предупреждение в логе и в панели есть, проверки перед
        # вызовом нет.
        history_before = len(self._messages)

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": "enabled" if self._config.thinking else "disabled"
                    }
                },
                **self._optional_params(),
            )
        except Exception as exc:
            logger.exception("[%s] вызов API упал", self._log_name)
            return self._failed_reply(
                _explain_api_error(exc, budget),
                elapsed=time.perf_counter() - started,
                request=request,
                service=service,
            )
        elapsed = time.perf_counter() - started

        choice = response.choices[0]
        text = choice.message.content or ""
        # reasoning_content отдаём в AgentReply для дебага, но в стек не
        # кладём: API не принимает рассуждение обратно в контекст.
        reasoning = getattr(choice.message, "reasoning_content", None) or None
        # `usage` теоретически может отсутствовать (разные провайдеры ведут
        # себя по-разному) — getattr от None вернёт дефолт, а не упадёт.
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        cost_usd = estimate_cost_usd(
            self._config.model, prompt_tokens, completion_tokens
        )

        reply = AgentReply(
            ok=True,
            text=text,
            error=None,
            finish_reason=choice.finish_reason,
            elapsed=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            reasoning=reasoning,
            model=self._config.model,
            agent_name=self._config.name,
            request_tokens=request,
            estimated_completion_tokens=tokens.estimate_tokens(text),
            prompt_cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", None),
            prompt_cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None),
            service_call=service,
        )

        if self._config.keep_history:
            self._messages.append({"role": "user", "content": user_message})
            self._messages.append({"role": "assistant", "content": text})
            self._persist()

        self._record(reply)
        self._record_turn(reply, history_before, view, service)
        logger.info(
            "[%s] model=%s thinking=%s finish_reason=%s time=%.2fs "
            "tokens(prompt/completion/total)=%s/%s/%s "
            "оценка/факт prompt=%s/%s (%s) ответ=%s/%s (%s) cost=%s "
            "стек=%d сообщ., %d символов: %s",
            self._log_name, self._config.model,
            _thinking_label(self._config.thinking),
            reply.finish_reason, reply.elapsed,
            prompt_tokens, completion_tokens, total_tokens,
            request.total, prompt_tokens,
            _delta_str(request.total, prompt_tokens),
            reply.estimated_completion_tokens, completion_tokens,
            _delta_str(reply.estimated_completion_tokens, completion_tokens),
            _cost_str(cost_usd), len(self._messages), len(text), text,
        )
        return reply

    def reset(self) -> None:
        """Очищает стек сообщений, память всех стратегий, checkpoint'ы и
        происхождение ветки. Счётчики за время жизни агента, журнал ходов и
        метрики последнего вызова сохраняются — это разные вещи: сброшен
        диалог, а не агент.

        Память чистится вся, а не только у активной стратегии: сводка
        удалённого диалога не должна пережить сброс и уехать в запрос после
        переключения. Checkpoint'ы и происхождение ветки чистятся вместе со
        стеком по той же причине — они ссылаются на диалог, которого больше
        нет (спецификация дня 10, §5.2). Ветки, созданные раньше, это не
        задевает: у них свои файлы.
        """
        self._messages = []
        for strategy in self._strategies.values():
            strategy.reset()
        self._checkpoints = []
        self._branch = None
        self._persist()
        logger.info(
            "[%s] стек сообщений очищен (reset); память стратегий (%s), "
            "checkpoint'ы и происхождение ветки очищены вместе с ним",
            self._log_name, ", ".join(f"«{name}»" for name in self._strategies),
        )

    def save_checkpoint(self) -> Checkpoint | None:
        """Фиксирует конец текущей истории вместе со снимком памяти всех
        стратегий (спецификация дня 10, §5.2).

        Пустой стек — ветвиться не от чего, `None`. Повторное сохранение без
        изменений (та же длина истории, та же стратегия, тот же снимок
        памяти, что у последнего checkpoint'а) не плодит одинаковых
        checkpoint'ов — возвращается уже существующий.
        """
        if not self._messages or not self._config.keep_history:
            logger.info(
                "[%s] checkpoint не сохранён: стек пуст, ветвиться не от чего",
                self._log_name,
            )
            return None

        # Глубокая копия: память стратегий дальше меняется, снимок — нет.
        memory = copy.deepcopy(self._context_dump()["memory"])
        existing = self._unchanged_checkpoint(memory)
        if existing is not None:
            logger.info(
                "[%s] checkpoint не сохранён: совпадает с существующим %s",
                self._log_name, existing.id,
            )
            return existing

        checkpoint = Checkpoint(
            id=f"cp{self._next_checkpoint_number()}",
            messages=len(self._messages),
            strategy=self._strategy_name,
            memory=memory,
            created_at=_now_iso(),
        )
        self._checkpoints.append(checkpoint)
        self._persist()
        logger.info(
            "[%s] checkpoint %s: %d сообщ., стратегия «%s», снимок памяти "
            "стратегий: %s",
            self._log_name, checkpoint.id, checkpoint.messages,
            checkpoint.strategy, ", ".join(memory) if memory else "нет памяти",
        )
        return checkpoint

    def fork(
        self,
        checkpoint_id: str,
        session_id: str,
        strategies: dict[str, context.ContextStrategy] | None = None,
    ) -> "Agent | None":
        """Создаёт ветку от одного из checkpoint'ов этой сессии: новый агент
        с новой сессией, в историю которого скопирован префикс до
        checkpoint'а, а память стратегий и активная стратегия — из снимка
        (спецификация дня 10, §5.2).

        Родителя (эту сессию) не трогает никак: ни стек, ни память, ни файл —
        семейство веток вычисляется по происхождению, а не хранится у
        родителя (`branch_family()`).
        """
        checkpoint = next(
            (cp for cp in self._checkpoints if cp.id == checkpoint_id), None
        )
        if (
            checkpoint is None
            or not self._config.keep_history
            or checkpoint.messages > len(self._messages)
        ):
            logger.warning(
                "[%s] ветка от checkpoint'а «%s» не создана: такого "
                "checkpoint'а нет или он устарел",
                self._log_name, checkpoint_id,
            )
            return None

        origin = BranchOrigin(
            parent=self._session_id,
            checkpoint=checkpoint.id,
            messages=checkpoint.messages,
            created_at=checkpoint.created_at,
        )
        seed = {
            "messages": self._messages[:checkpoint.messages],
            "context": {
                "strategy": checkpoint.strategy,
                "memory": copy.deepcopy(checkpoint.memory),
                "branch": asdict(origin),
                # Ключа `checkpoints` здесь нет: список checkpoint'ов у ветки
                # пустой (спецификация дня 10, §2.3) — checkpoint'ы не
                # наследуются.
            },
        }
        return Agent(
            self._config,
            session_id=session_id,
            store=self._store,
            strategies=strategies,
            seed=seed,
        )

    def debug_state(self) -> dict:
        """Состояние агента для дебаг-панели: конфиг, стек, метрики
        последнего вызова, накопленное за время жизни и счётчики процесса."""
        view = self.context_view()
        return {
            "number": self._number,
            "config": asdict(self._config),
            "session_id": self._session_id,
            # С дня 7 здесь имя класса хранилища (`JsonHistoryStore`); None
            # означает агента, живущего только в памяти процесса.
            "store": type(self._store).__name__ if self._store else None,
            "restored_messages": self._restored_messages,
            "store_error": self._store_error,
            "history_size": len(self._messages),
            "messages": self.history,
            "last_call": asdict(self._last_reply) if self._last_reply else None,
            "totals": dict(self._totals),
            "process": process_stats(),
            # Ключи дня 8 — в конце: существующие не переименовываются и не
            # переставляются, иначе поехали бы рендеры дебаг-панели.
            # `context` заполняется из того же `ContextView`, что и ключ
            # `context_view` ниже: бюджет за один рендер считается один раз.
            "context": asdict(view.usage),
            "turns": [asdict(turn) for turn in self._turns],
            "calibration": self.calibration,
            # Ключи дня 9 — тоже в конце. В `context_view` уходит описание
            # назревшей свёртки, а не сама `ContextTask`: её `messages` могут
            # весить сотни килобайт, и в панель им не место.
            "strategy": self._strategy_name,
            "context_view": asdict(view),
            # Ключи дня 10 — в конце. Снимок памяти в `checkpoints` панели не
            # нужен (весит килобайты) и не отдаётся — только id, длина,
            # стратегия и время.
            "checkpoints": [
                {
                    "id": cp.id,
                    "messages": cp.messages,
                    "strategy": cp.strategy,
                    "created_at": cp.created_at,
                }
                for cp in self._checkpoints
            ],
            "branch": asdict(self._branch) if self._branch is not None else None,
        }

    # --- Внутреннее ------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """Единственное место, где собирается список сообщений для запроса.

        С дня 9 сборку делает стратегия: агент отдаёт ей системный промпт,
        весь стек и новый вопрос, а что из этого уедет в модель — решает она
        (`context.py`, §4). Сам стек при этом не меняется: сжимается запрос,
        а не память. День 10 добавил сюда «Скользящее окно» и «Факты +
        последние N», не тронув ни строчки в этом методе — обе встают в тот
        же протокол `ContextStrategy`. Ветки — операция над самой историей
        (`fork()`), а не способ собрать запрос, и этого метода не касаются.
        """
        return self.strategy.build(
            self._config.system_prompt, self._messages, user_message
        )

    def _context_view(self, question: str = "") -> tuple[list[dict], ContextView]:
        """Сборка запроса и всё, что про неё нужно знать панели и логам, —
        одним проходом.

        Отдаёт и сам список сообщений, и `ContextView`: `ask()` нужно и то и
        другое, и собирать запрос дважды (один раз чтобы отправить, другой
        чтобы показать) — верный способ разъехаться.

        Чистый метод: `describe()`, `prepare()` и `build()` ничего не меняют и
        в сеть не ходят, поэтому его безопасно звать на каждый рендер панели.
        """
        messages = self._build_messages(question)
        usage = tokens.context_usage(
            self._count_messages(messages),
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            calibration=self.calibration,
        )
        # База для сравнения: во сколько обошёлся бы тот же ход без всякого
        # управления контекстом. Считается тем же счётчиком по той же сборке,
        # только стратегией «Вся история».
        full = self._count_messages(
            _FULL_HISTORY.build(
                self._config.system_prompt, self._messages, question
            )
        )
        state = self.strategy.describe(self._messages)
        task = self.strategy.prepare(self._messages, question)
        return messages, ContextView(
            strategy=self._strategy_name,
            state=state,
            usage=usage,
            full=full,
            # Сколько сообщений истории реально ушло в запрос: `per_message`
            # считается только по ним, память стратегии в этот ряд не входит.
            sent_messages=len(usage.request.per_message),
            history_messages=len(self._messages),
            # Экономия — по сырым оценкам обеих сборок: калибровка это общий
            # множитель, в отношении он бы сократился.
            saved_tokens=full.total - usage.estimated,
            # В панель уходит описание назревшей задачи, а не она сама: её
            # `messages` могут весить сотни килобайт.
            pending_label=task.label if task else "",
            pending_messages=task.covers - state.covered if task else 0,
            pending_sent_messages=task.sends if task else 0,
        )

    def _run_context_task(
        self, client: OpenAI, question: str
    ) -> ServiceCall | None:
        """Служебный вызов, если стратегия его попросила: обновление памяти —
        свёртка в сводку или разбор фактов, в зависимости от активной
        стратегии (день 10). Метод один на обе: слова для лога и панели
        приходят от стратегии (`task.label`, `StrategyState.memory_label`,
        `StrategyState.last_update`) — агент не знает, сводка это или факты.

        Второе (и последнее) место в проекте, где вызывается
        `chat.completions.create`. Набор параметров у него намеренно другой:
        сообщения из задачи, `thinking` всегда выключен (скрытые
        reasoning-токены тратят тот же бюджет `max_tokens`, что и видимый
        ответ, и на маленьком лимите ответ пришёл бы пустым), `max_tokens` из
        задачи, ни температуры, ни `stop`.

        Это настоящий вызов, но не ход: в стек сообщений он не пишет, строку
        журнала ходов не создаёт и `_last_reply` не трогает.
        """
        task = self.strategy.prepare(self._messages, question)
        if task is None:
            return None

        before = self.strategy.describe(self._messages)
        covered_before = before.covered
        folded = self._messages[covered_before:task.covers]
        folded_tokens = sum(
            tokens.estimate_tokens(message.get("content")) for message in folded
        )

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self._config.model,
                messages=task.messages,
                extra_body={"thinking": {"type": "disabled"}},
                **({"max_tokens": task.max_tokens} if task.max_tokens else {}),
            )
        except Exception as exc:
            logger.exception("[%s] служебный вызов упал", self._log_name)
            call = ServiceCall(
                kind=task.kind, label=task.label, ok=False,
                # В `error` только причина, без слова «свёртка»: и лог, и
                # панель подписывают её сами, и дважды это читается плохо.
                error=str(exc),
                elapsed=time.perf_counter() - started,
                prompt_tokens=None, completion_tokens=None, total_tokens=None,
                cost_usd=None, covers=0, folded_tokens=folded_tokens, text="",
                memory_label=before.memory_label,
            )
            self._record_service(call)
            logger.warning(
                "[%s] служебный вызов — %s: не удался (%s); ход не "
                "отменяется: память не сдвинулась, в запрос уйдёт всё "
                "неучтённое (%d сообщ.), вызов повторится на следующем ходе",
                self._log_name, task.label, call.error,
                len(self._messages) - covered_before,
            )
            return call

        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        cost_usd = estimate_cost_usd(
            self._config.model, prompt_tokens, completion_tokens
        )

        if not text:
            call = ServiceCall(
                kind=task.kind, label=task.label, ok=False,
                error="модель вернула пустой ответ",
                elapsed=elapsed,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=total_tokens, cost_usd=cost_usd,
                covers=0, folded_tokens=folded_tokens, text="",
                finish_reason=choice.finish_reason,
                memory_label=before.memory_label,
            )
            self._record_service(call)
            logger.warning(
                "[%s] служебный вызов — %s: модель вернула пустой ответ — "
                "память не сдвинулась, сообщения #%d-#%d останутся в запросе "
                "целиком, вызов повторится на следующем ходе",
                self._log_name, task.label, covered_before, task.covers - 1,
            )
            return call

        # `apply()` — единственный, кто знает, разобралась ли хоть одна
        # строка (день 10, §5.1): непустой ответ ещё не значит успех.
        applied = self.strategy.apply(task, text)
        after = self.strategy.describe(self._messages)
        call = ServiceCall(
            kind=task.kind,
            label=task.label,
            ok=applied,
            error=None if applied else "ответ модели не разобран",
            elapsed=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            covers=task.covers - covered_before if applied else 0,
            folded_tokens=folded_tokens,
            text=text,
            finish_reason=choice.finish_reason,
            memory_label=after.memory_label,
            memory_update=after.last_update if applied else "",
        )
        # Токены потрачены в любом случае — считаем их до того, как решим,
        # годится ли результат.
        self._record_service(call)
        if not applied:
            logger.warning(
                "[%s] служебный вызов — %s: ответ модели не разобран — "
                "память не сдвинулась, начало ответа: %r; в запрос уйдёт "
                "всё неучтённое, вызов повторится на следующем ходе",
                self._log_name, task.label, text[:200],
            )
            return call

        # Немедленное сохранение: обновление уже оплачено, и падение до
        # основного вызова не должно его терять.
        self._persist()
        logger.info(
            "[%s] служебный вызов — %s: учтено ≈%s токенов истории → %s "
            "≈%s токенов (%s); %.2fs, tokens=%s/%s/%s, cost=%s",
            self._log_name, task.label, _num(folded_tokens),
            after.memory_label.lower(),
            _num(tokens.estimate_tokens(after.memory_text)),
            after.last_update, elapsed,
            prompt_tokens, completion_tokens, total_tokens, _cost_str(cost_usd),
        )
        # Обрезанный ответ применён — отказ от него оставил бы задачу
        # назревшей, и она повторялась бы каждый ход за деньги и с тем же
        # итогом. Но молча это не проходит: штатно в потолок упираться нельзя.
        if choice.finish_reason == "length":
            logger.warning(
                "[%s] служебный вызов — %s: ответ упёрся в max_tokens=%s и "
                "оборван на полуслове — применён как есть; если это "
                "повторяется, потолок мал для реальной длины (спецификации "
                "дня 9, §3.1, и дня 10, §3.1)",
                self._log_name, task.label, task.max_tokens,
            )
        return call

    def _count_messages(self, messages: list[dict]) -> tokens.RequestTokens:
        """Разложение готового списка сообщений на систему / память стратегии /
        историю / вопрос.

        Единственное место, где список сообщений превращается в оценку: и
        `ask()`, и `context_usage()` ходят сюда, поэтому «сколько посчитали»
        и «сколько отправили» считаются одним и тем же способом по одному и
        тому же списку.

        С дня 9 разбор идёт по ролям, а не по позициям: первое сообщение —
        системный промпт, последнее — вопрос, а все `system` между ними это
        память стратегии (сводка сегодня, блок фактов у дня 10). Правило
        безопасно: история агента нормализуется до `user`/`assistant` ещё с
        дня 6, и `system` в ней не бывает.
        """
        middle = messages[1:-1]
        return tokens.count_request(
            system_prompt=messages[0]["content"],
            history=[m for m in middle if m.get("role") != "system"],
            question=messages[-1]["content"],
            memory="\n\n".join(
                m.get("content") or "" for m in middle if m.get("role") == "system"
            ),
        )

    def _log_context(self, view: ContextView) -> None:
        """Строка стратегии перед строкой бюджета: что уходит в модель и
        сколько это против полного контекста. По логу должно быть видно, что
        сжатие работает, — не открывая панель.

        Строки «свёртка назрела» здесь нет намеренно: после фазы подготовки
        назревшая свёртка бывает только несостоявшейся, и строка про неё
        противоречила бы предупреждению о сбое, которое уже в логе."""
        logger.info(
            "[%s] стратегия «%s»: в запросе %d сообщ. из %d, ≈%s вместо ≈%s (%s)",
            self._log_name, view.strategy, view.sent_messages,
            view.history_messages, _num(view.usage.estimated),
            _num(view.full.total), _saved_str(view.saved_tokens, view.full.total),
        )

    def _log_budget(self, budget: tokens.ContextUsage) -> None:
        """Строка бюджета в лог перед вызовом API — и отдельное предупреждение,
        когда занято больше `WARN_RATIO`: приближение к лимиту должно быть
        видно в терминале, а не только в панели."""
        request = budget.request
        logger.info(
            "[%s] бюджет: система %s + память %s + история %s + вопрос %s + "
            "служебные %s ≈ %s из %s доступных (%s); окно %s, резерв под "
            "ответ %s",
            self._log_name,
            _num(request.system), _num(request.memory), _num(request.history),
            _num(request.question), _num(request.overhead), _num(budget.used),
            _num(budget.available), _ratio_str(budget.ratio),
            _num(budget.limit), _num(budget.answer_reserve),
        )
        if budget.level in ("warn", "danger", "over"):
            logger.warning(
                "[%s] контекст занят на %s (уровень %s): запрос всё равно "
                "уходит в API — предохранителя перед вызовом в проекте нет "
                "(день 8, §2.2), сжатие это способ до переполнения не "
                "доходить, а не проверка перед вызовом",
                self._log_name, _ratio_str(budget.ratio), budget.level,
            )

    def _optional_params(self) -> dict:
        """Опциональные параметры запроса. То, что в конфиге `None`, в API
        не передаётся вообще — иначе DeepSeek получит явный null вместо
        своего дефолта."""
        params = {
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "stop": self._config.stop,
        }
        return {key: value for key, value in params.items() if value is not None}

    def _get_client(self) -> OpenAI:
        """Клиент DeepSeek (OpenAI-совместимый SDK). Кэшируется на агенте;
        при отсутствующем ключе бросает RuntimeError с понятным текстом,
        который `ask()` превращает в `AgentReply(ok=False)`."""
        if self._client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError(_NO_API_KEY_ERROR)
            self._client = OpenAI(api_key=api_key, base_url=self._config.base_url)
        return self._client

    def _persist(self) -> None:
        """Сохранение стека в хранилище, если оно передано (при `store=None`
        метод ничего не делает).

        Наружу отсюда ничего не бросается: `_persist()` вызывается изнутри
        `ask()`, а `ask()` по контракту исключений не бросает. Ход при сбое
        записи состоялся, ответ пользователю отдаётся как обычно, причина
        видна в панели и в логе, а попытка повторится на следующем ходе:
        разовый сбой диска не должен переводить агента в режим «только
        память» навсегда.
        """
        if self._store is None or not self._config.keep_history:
            return
        if not self._store_writable:
            return
        try:
            self._store.save(
                self._session_id, self.history, context=self._context_dump()
            )
        except Exception as exc:
            self._store_error = f"история не сохранилась: {exc}"
            logger.warning("[%s] %s", self._log_name, self._store_error)
        else:
            self._store_error = None

    def _context_dump(self) -> dict:
        """Память стратегий, имя активной, checkpoint'ы и происхождение ветки —
        одним блоком для хранилища.

        Пустые памяти в файл не пишутся: у «Всей истории» её нет вовсе, и
        строчка `{}` в файле сессии только мешала бы читать его глазами. Так
        же не пишутся пустой список checkpoint'ов и отсутствующее
        происхождение ветки (день 10) — хранилище про них ничего не знает,
        блок `context` для него непрозрачен целиком.
        """
        memory = {
            name: dump
            for name, strategy in self._strategies.items()
            if (dump := strategy.dump())
        }
        data: dict = {"strategy": self._strategy_name, "memory": memory}
        if self._checkpoints:
            data["checkpoints"] = [asdict(cp) for cp in self._checkpoints]
        if self._branch is not None:
            data["branch"] = asdict(self._branch)
        return data

    def _load_context(self, data: dict) -> None:
        """Память стратегий из файла сессии.

        Файл версии 1 (ключа `context` в нём нет) и битый `context` дают
        пустую память и стратегию по умолчанию — читать историю это не мешает.
        Имя стратегии из файла перекрывает переданное конструктором: у
        восстановленного агента умолчания нет, берётся записанное.
        """
        if not isinstance(data, dict) or not data:
            return
        memory = data.get("memory")
        if isinstance(memory, dict):
            for name, dump in memory.items():
                strategy = self._strategies.get(name)
                if strategy is None:
                    self._startup_warnings.append(
                        f"память стратегии «{name}» в файле есть, а самой "
                        f"стратегии в наборе нет — пропущена"
                    )
                    continue
                if not isinstance(dump, dict):
                    strategy.load({})
                    self._startup_warnings.append(
                        f"память стратегии «{name}» в файле имеет тип "
                        f"{type(dump).__name__} вместо словаря — память пустая"
                    )
                    continue
                strategy.load(dump)
                # Стратегия переживает мусор молча: логов у `context.py` нет.
                # Что прочитано не всё, агент узнаёт сам — память, прочитанная
                # целиком, выгружается обратно ровно в тот же словарь.
                if strategy.dump() != dump:
                    self._startup_warnings.append(
                        f"память стратегии «{name}» в файле прочитана не "
                        f"целиком (битые или лишние поля) — работаем с тем, "
                        f"что удалось разобрать"
                    )
        elif memory is not None:
            self._startup_warnings.append(
                f"ключ memory в файле сессии имеет тип "
                f"{type(memory).__name__} вместо словаря — память пустая"
            )
        saved = data.get("strategy")
        if isinstance(saved, str) and saved:
            self._strategy_name = self._known_strategy(saved)
        # Checkpoint'ы и происхождение ветки (день 10) — тем же приёмом, что
        # память стратегий: мусор переживается с предупреждением, историю
        # читать это не мешает. `self._messages` здесь уже проставлены — и при
        # чтении из хранилища, и из `seed` (порядок в `__init__` гарантирует
        # это в обоих случаях).
        self._checkpoints = self._parse_checkpoints(data.get("checkpoints"))
        self._branch = self._parse_branch(data.get("branch"))

    def _parse_checkpoints(self, data: object) -> list[Checkpoint]:
        """Checkpoint'ы из файла сессии. Файл дня 9 (ключа нет) даёт пустой
        список без предупреждений; переживает мусор: не список, не словарь,
        чужие типы, пустые строки, `messages` вне `1..len(history)`,
        повторяющиеся `id`, `memory` не словарь — такой checkpoint
        пропускается с предупреждением, остальные читаются (спецификация
        дня 10, §5.2)."""
        if data is None:
            return []
        if not isinstance(data, list):
            self._startup_warnings.append(
                f"ключ checkpoints в файле имеет тип {type(data).__name__} "
                f"вместо списка — checkpoint'ы не прочитаны"
            )
            return []
        history_len = len(self._messages)
        result: list[Checkpoint] = []
        seen_ids: set[str] = set()
        for item in data:
            checkpoint = self._parse_one_checkpoint(item, history_len, seen_ids)
            if checkpoint is not None:
                result.append(checkpoint)
                seen_ids.add(checkpoint.id)
        return result

    def _parse_one_checkpoint(
        self, item: object, history_len: int, seen_ids: set[str]
    ) -> Checkpoint | None:
        if not isinstance(item, dict):
            self._startup_warnings.append(
                f"checkpoint в файле имеет тип {type(item).__name__} вместо "
                f"словаря — пропущен"
            )
            return None
        checkpoint_id = item.get("id")
        messages = item.get("messages")
        strategy = item.get("strategy")
        memory = item.get("memory")
        created_at = item.get("created_at")
        valid = (
            isinstance(checkpoint_id, str) and checkpoint_id
            and checkpoint_id not in seen_ids
            and isinstance(messages, int) and not isinstance(messages, bool)
            and 1 <= messages <= history_len
            and isinstance(strategy, str) and strategy
            and isinstance(memory, dict)
            and isinstance(created_at, str) and created_at
        )
        if not valid:
            self._startup_warnings.append(
                f"checkpoint «{checkpoint_id}» в файле битый — пропущен"
            )
            return None
        return Checkpoint(
            id=checkpoint_id, messages=messages, strategy=strategy,
            memory=memory, created_at=created_at,
        )

    def _parse_branch(self, data: object) -> BranchOrigin | None:
        """Происхождение ветки из файла сессии. Битое происхождение (не
        словарь, пустые `parent`/`checkpoint`/`created_at`, `messages` вне
        `1..len(history)`) — не ветка, с предупреждением (спецификация
        дня 10, §5.2): эта сессия тогда считается самостоятельным диалогом.
        Снимок памяти внутри checkpoint'а здесь не разбирается — он
        непрозрачен так же, как `context` для хранилища."""
        if data is None:
            return None
        if not isinstance(data, dict):
            self._startup_warnings.append(
                f"ключ branch в файле имеет тип {type(data).__name__} вместо "
                f"словаря — происхождение не прочитано"
            )
            return None
        parent = data.get("parent")
        checkpoint = data.get("checkpoint")
        messages = data.get("messages")
        created_at = data.get("created_at")
        valid = (
            isinstance(parent, str) and parent
            and isinstance(checkpoint, str) and checkpoint
            and isinstance(messages, int) and not isinstance(messages, bool)
            and 1 <= messages <= len(self._messages)
            and isinstance(created_at, str) and created_at
        )
        if not valid:
            self._startup_warnings.append(
                "происхождение ветки в файле битое — эта сессия считается "
                "самостоятельным диалогом"
            )
            return None
        return BranchOrigin(
            parent=parent, checkpoint=checkpoint, messages=messages,
            created_at=created_at,
        )

    def _next_checkpoint_number(self) -> int:
        """Следующий свободный номер checkpoint'а этой сессии, с 1."""
        numbers = [
            int(cp.id[2:]) for cp in self._checkpoints
            if cp.id.startswith("cp") and cp.id[2:].isdigit()
        ]
        return max(numbers, default=0) + 1

    def _unchanged_checkpoint(self, memory: dict) -> Checkpoint | None:
        """Последний checkpoint, если он совпадает с тем, что получился бы
        сейчас, — та же длина истории, та же стратегия, тот же снимок памяти.
        Только последний: более ранний совпадающий checkpoint не должен
        мешать завести новый в другой точке истории."""
        if not self._checkpoints:
            return None
        last = self._checkpoints[-1]
        if (
            last.messages == len(self._messages)
            and last.strategy == self._strategy_name
            and last.memory == memory
        ):
            return last
        return None

    def _known_strategy(self, name: str) -> str:
        """Имя стратегии из файла или от вызывающей стороны, приведённое к
        набору этого агента.

        Имени нет в наборе (стратегию переименовали или это файл дня 10 на
        коде дня 9) — берём первую и предупреждаем; падать из-за этого
        приложение не должно. Тот же приём, что с именем пресета на дне 7.
        """
        if name in self._strategies:
            return name
        fallback = next(iter(self._strategies))
        self._startup_warnings.append(
            f"стратегия «{name}» в наборе не найдена, работаем на «{fallback}»"
        )
        return fallback

    def _failed_reply(
        self,
        error: str,
        elapsed: float,
        request: "tokens.RequestTokens | None" = None,
        service: ServiceCall | None = None,
    ) -> AgentReply:
        """Неуспешный ход. `request` есть только у ошибок API: до вызова
        запрос уже был собран и посчитан, и отклонённый запрос — как раз тот
        случай, когда оценку хочется видеть. У ошибки без ключа считать нечего.
        """
        reply = AgentReply(
            ok=False,
            text="",
            error=error,
            finish_reason=None,
            elapsed=elapsed,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost_usd=None,
            reasoning=None,
            model=self._config.model,
            agent_name=self._config.name,
            request_tokens=request,
            service_call=service,
        )
        self._record(reply)
        logger.warning("[%s] ошибка: %s", self._log_name, error)
        return reply

    def _record_turn(
        self,
        reply: AgentReply,
        history_before: int,
        view: ContextView,
        service: ServiceCall | None,
    ) -> None:
        """Строка журнала ходов — только на успешный вызов и после `_record()`:
        накопительные числа берутся из уже обновлённых счётчиков агента.

        Ход без `usage` в журнал попадает (он состоялся), но в калибровку —
        нет: сравнивать оценку не с чем.
        """
        self._turns.append(
            TurnStats(
                turn=len(self._turns) + 1,
                history_before=history_before,
                estimated_prompt_tokens=(
                    reply.request_tokens.total if reply.request_tokens else 0
                ),
                prompt_tokens=reply.prompt_tokens or 0,
                completion_tokens=reply.completion_tokens or 0,
                total_tokens=reply.total_tokens or 0,
                cost_usd=reply.cost_usd,
                cumulative_total_tokens=self._totals["total_tokens"],
                cumulative_cost_usd=self._totals["cost_usd"],
                strategy=view.strategy,
                sent_messages=view.sent_messages,
                full_prompt_tokens=view.full.total,
                service_tokens=(service.total_tokens or 0) if service else 0,
                service_cost_usd=service.cost_usd if service else None,
                elapsed=reply.elapsed,
                service_elapsed=service.elapsed if service else 0.0,
            )
        )
        # Экономия копится по ходам и может быть отрицательной: на коротком
        # диалоге свёртка дороже, чем её отсутствие.
        self._totals["saved_tokens"] += view.saved_tokens

    def _record_service(self, call: ServiceCall) -> None:
        """Учёт служебного вызова. Он считается в счётчиках агента и процесса
        наравне с обычными — те же токены, та же оценка стоимости — и отдельно
        в своих: рядом с «сэкономлено» в панели всегда стоит «потрачено на
        свёртки».

        `_last_reply` не трогается: в блоке «Последний вызов» должен оставаться
        ответ игроку. В калибровку служебный вызов тоже не идёт — собственной
        оценки этого запроса мы не считали, сравнивать факт не с чем.
        """
        prompt_tokens = call.prompt_tokens or 0
        completion_tokens = call.completion_tokens or 0
        total_tokens = call.total_tokens or 0
        cost_usd = call.cost_usd or 0.0
        errors = 0 if call.ok else 1

        self._totals["calls"] += 1
        self._totals["errors"] += errors
        self._totals["prompt_tokens"] += prompt_tokens
        self._totals["completion_tokens"] += completion_tokens
        self._totals["total_tokens"] += total_tokens
        self._totals["cost_usd"] += cost_usd
        self._totals["service_calls"] += 1
        self._totals["service_tokens"] += total_tokens
        self._totals["service_cost_usd"] += cost_usd

        _bump_process_stats(
            calls=1,
            errors=errors,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

    def _record(self, reply: AgentReply) -> None:
        """Учёт вызова в счётчиках агента и процесса. Неуспешный вызов тоже
        считается вызовом (и отдельно — ошибкой); токены и деньги при ошибке
        не начисляются."""
        self._last_reply = reply
        prompt_tokens = reply.prompt_tokens or 0
        completion_tokens = reply.completion_tokens or 0
        total_tokens = reply.total_tokens or 0
        cost_usd = reply.cost_usd or 0.0
        errors = 0 if reply.ok else 1

        self._totals["calls"] += 1
        self._totals["errors"] += errors
        self._totals["prompt_tokens"] += prompt_tokens
        self._totals["completion_tokens"] += completion_tokens
        self._totals["total_tokens"] += total_tokens
        self._totals["cost_usd"] += cost_usd

        # Калибровка (день 8): в неё идут только успешные вызовы, у которых
        # пришёл `usage`, — и только вместе с оценкой того же самого запроса.
        if reply.ok and reply.prompt_tokens is not None and reply.request_tokens:
            self._totals["calibration_calls"] += 1
            self._totals["calibration_fact_tokens"] += reply.prompt_tokens
            self._totals["calibration_estimated_tokens"] += reply.request_tokens.total

        _bump_process_stats(
            calls=1,
            errors=errors,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Приводит загруженную из хранилища историю (или префикс `seed` ветки,
    день 10) к тому же виду, в котором её ведёт агент: только user/assistant,
    только role и content."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in ("user", "assistant")
    ]


def _now_iso() -> str:
    """Время для checkpoint'а и происхождения ветки, ISO до секунд — тем же
    форматом, что `storage.py` пишет `created_at`/`updated_at`."""
    return datetime.now().isoformat(timespec="seconds")


def _explain_api_error(exc: Exception, budget: tokens.ContextUsage) -> str:
    """Текст ошибки API для `AgentReply.error`.

    Переполнение контекста узнаётся по признакам из `CONTEXT_OVERFLOW_MARKERS`
    и объясняется по-человечески: с API это не «модель забудет начало
    диалога», а отказ целиком — ответа нет, ход не состоялся. Не узнали —
    отдаём текст ошибки как есть, как и на дне 6.
    """
    text = str(exc)
    lowered = text.lower()
    if not any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS):
        return f"Ошибка при обращении к DeepSeek API: {text}"
    # Что делать дальше, зависит от того, что переполнило окно. Сжатие и сброс
    # укорачивают историю, но не сам вопрос: заполнитель на 1.2 млн токенов
    # не влезет ни в какой запрос, и обещать обратное было бы неправдой.
    question = budget.request.question
    if budget.available is not None and question >= budget.available:
        advice = (
            f"Окно переполняет сам вопрос (≈{_num(question)} токенов): ни сброс "
            f"диалога, ни сжатие истории тут не помогут — вопрос придётся "
            f"сократить."
        )
    else:
        # День 10: совет не про «сжимающую» стратегию — окно историю не
        # сжимает, а отбрасывает, но тоже не отправляет всю историю целиком.
        advice = (
            "Вопрос можно задать заново, сбросив диалог или переключив "
            "стратегию управления контекстом на такую, что не отправляет всю "
            "историю целиком (сжатие в сводку или окно последних сообщений)."
        )
    return (
        f"Запрос не влез в контекстное окно модели и отклонён целиком. "
        f"Мы насчитали ≈{_num(budget.used)} токенов "
        f"(сырая оценка {_num(budget.estimated)}) при окне модели "
        f"{_num(budget.limit)} и резерве {_num(budget.answer_reserve)} под "
        f"ответ — доступно было {_num(budget.available)}. "
        f"Ответа нет: превышенный контекст означает отказ, а не забывание "
        f"начала диалога. Стек сообщений не изменился, файл сессии не тронут. "
        f"{advice} Предохранителя перед вызовом нет намеренно. "
        f"Текст ошибки API: {text}"
    )


def _memory_note(state: context.StrategyState) -> str:
    """Кусок строки лога про память восстановленной стратегии: по логу должно
    быть видно, приехала память из файла или нет. Как память называется,
    говорит сама стратегия — агент не знает, сводка это или факты."""
    if not state.memory_text:
        return ""
    return (
        f", память «{state.name}»: {state.memory_label.lower()} "
        f"≈{tokens.estimate_tokens(state.memory_text)} токенов на "
        f"{state.covered} сообщ."
    )


def _saved_str(saved: int, full_total: int) -> str:
    """Экономия в процентах от полного контекста. Со знаком: на коротком
    диалоге сжатие бывает дороже, чем его отсутствие, и показывать это надо
    честно."""
    if full_total <= 0:
        return "н/д"
    sign = "−" if saved >= 0 else "+"
    return f"{sign}{abs(saved) / full_total * 100:.0f}%"


def _thinking_label(thinking: bool) -> str:
    return "on" if thinking else "off"


def _cost_str(cost_usd: float | None) -> str:
    return "н/д" if cost_usd is None else f"${cost_usd:.6f}"


def _num(value: int | None) -> str:
    """Число с разделителями разрядов: строку бюджета в логе читает человек,
    а числа там семизначные."""
    return "н/д" if value is None else f"{value:,}".replace(",", " ")


def _ratio_str(ratio: float | None) -> str:
    return "н/д" if ratio is None else f"{ratio * 100:.1f}%"


def _delta_str(estimated: int | None, fact: int | None) -> str:
    """Расхождение оценки с фактом в процентах, со знаком. Само расхождение
    нигде не хранится — это производная от двух чисел, которые уже есть."""
    if estimated is None or not fact:
        return "н/д"
    return f"{(estimated - fact) / fact * 100:+.1f}%"
