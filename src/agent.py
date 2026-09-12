# TooManyRules — сущность агента (день 6, неделя 2; день 8 — работа с токенами,
# день 9 — управление контекстом).
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
# и дня 9, §4.
#
# С дня 9 вызовов LLM API здесь два: основной (ответ игроку) и служебный
# (свёртка старой части диалога в сводку, которую попросила стратегия). Оба —
# в этом модуле: правило «вызовы LLM API только в `agent.py`» не нарушено.

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
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
    # В конце и с умолчанием, как поля дня 8 в `AgentReply`. "length" — сводка
    # упёрлась в `max_tokens` и оборвана на полуслове: она всё равно применена,
    # но панель и лог говорят об этом вслух (спецификация дня 9, §6.2).
    finish_reason: str | None = None


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
    pending_label: str             # «свёртка назрела: …», "" — не назрела
    pending_messages: int


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
        # Запись выключается ровно в одном случае — если не удалось прочитать
        # свою историю: агент, не прочитавший файл, не должен его затирать.
        self._store_writable = store is not None
        if store is not None and config.keep_history:
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
            # Память стратегий и имя активной — отдельным чтением и отдельным
            # try: сбой контекста не должен отменять чтение истории. Не
            # прочиталось — стартуем с пустой памятью и предупреждением в лог,
            # файл при этом не трогаем.
            try:
                self._load_context(store.load_context(session_id))
            except Exception as exc:
                self._startup_warnings.append(
                    f"память стратегий не прочиталась, стартуем с пустой: {exc}"
                )
        # Сколько сообщений пришло из хранилища: после первого же хода
        # `len(history)` растёт, и «сколько было восстановлено» иначе не
        # показать.
        self._restored_messages = len(self._messages)

        # Регистрация в реестре — последним шагом конструктора, когда все
        # поля уже проставлены: наружу не должен попасть недособранный агент.
        self._number = _register_agent(self)
        # Префикс логов: с несколькими агентами одного пресета одно только имя
        # ничего не различает, а с дня 7 у каждого агента ещё и свой файл —
        # поэтому в префиксе и номер, и сессия.
        self._log_name = f"#{self._number} {config.name} · {session_id}"
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
        """Очищает стек сообщений и память всех стратегий. Счётчики за время
        жизни агента, журнал ходов и метрики последнего вызова сохраняются —
        это разные вещи: сброшен диалог, а не агент.

        Память чистится вся, а не только у активной стратегии: сводка
        удалённого диалога не должна пережить сброс и уехать в запрос после
        переключения.
        """
        self._messages = []
        for strategy in self._strategies.values():
            strategy.reset()
        self._persist()
        logger.info(
            "[%s] стек сообщений очищен (reset); память стратегий (%s) "
            "очищена вместе с ним",
            self._log_name, ", ".join(f"«{name}»" for name in self._strategies),
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
        }

    # --- Внутреннее ------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """Единственное место, где собирается список сообщений для запроса.

        С дня 9 сборку делает стратегия: агент отдаёт ей системный промпт,
        весь стек и новый вопрос, а что из этого уедет в модель — решает она
        (`context.py`, §4). Сам стек при этом не меняется: сжимается запрос,
        а не память. День 10 добавит сюда ещё три стратегии, не тронув ни
        строчки в этом методе.
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
            # В панель уходит описание назревшей свёртки, а не сама задача:
            # её `messages` могут весить сотни килобайт.
            pending_label=task.label if task else "",
            pending_messages=task.covers - state.covered if task else 0,
        )

    def _run_context_task(
        self, client: OpenAI, question: str
    ) -> ServiceCall | None:
        """Служебный вызов, если стратегия его попросила: свернуть старую
        часть диалога в сводку.

        Второе (и последнее) место в проекте, где вызывается
        `chat.completions.create`. Набор параметров у него намеренно другой:
        сообщения из задачи, `thinking` всегда выключен (скрытые
        reasoning-токены тратят тот же бюджет `max_tokens`, что и видимый
        ответ, и на маленьком лимите сводка пришла бы пустой), `max_tokens` из
        задачи, ни температуры, ни `stop`.

        Это настоящий вызов, но не ход: в стек сообщений он не пишет, строку
        журнала ходов не создаёт и `_last_reply` не трогает.
        """
        task = self.strategy.prepare(self._messages, question)
        if task is None:
            return None

        covered_before = self.strategy.describe(self._messages).covered
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
            )
            self._record_service(call)
            logger.warning(
                "[%s] свёртка не удалась: %s; ход не отменяется: память не "
                "сдвинулась, в запрос уйдёт всё несвёрнутое (%d сообщ.), "
                "свёртка повторится на следующем ходе",
                self._log_name, call.error, len(self._messages) - covered_before,
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
        call = ServiceCall(
            kind=task.kind,
            label=task.label,
            ok=bool(text),
            error=None if text else "модель вернула пустую сводку",
            elapsed=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            covers=task.covers - covered_before if text else 0,
            folded_tokens=folded_tokens,
            text=text,
            finish_reason=choice.finish_reason,
        )
        # Токены потрачены в любом случае — считаем их до того, как решим,
        # годится ли результат.
        self._record_service(call)
        if not text:
            logger.warning(
                "[%s] свёртка: модель вернула пустой ответ — память не "
                "сдвинулась, сообщения #%d-#%d останутся в запросе целиком, "
                "свёртка повторится на следующем ходе",
                self._log_name, covered_before, task.covers - 1,
            )
            return call

        self.strategy.apply(task, text)
        # Немедленное сохранение: свёртка уже оплачена, и падение до основного
        # вызова не должно её терять.
        self._persist()
        logger.info(
            "[%s] свёртка: сообщения #%d-#%d (%d шт., ≈%s токенов) → сводка "
            "≈%s токенов; служебный вызов %.2fs, tokens=%s/%s/%s, cost=%s",
            self._log_name, covered_before, task.covers - 1, call.covers,
            _num(folded_tokens), _num(tokens.estimate_tokens(text)), elapsed,
            prompt_tokens, completion_tokens, total_tokens, _cost_str(cost_usd),
        )
        # Обрезанная сводка применена — отказ от неё оставил бы свёртку
        # назревшей, и она повторялась бы каждый ход за деньги и с тем же
        # итогом. Но молча это не проходит: штатно в потолок упираться нельзя.
        if choice.finish_reason == "length":
            logger.warning(
                "[%s] свёртка: сводка упёрлась в max_tokens=%s и оборвана на "
                "полуслове — применена как есть; если это повторяется, потолок "
                "мал для реальной длины сводки (спецификация дня 9, §3.1)",
                self._log_name, task.max_tokens,
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
        """Память стратегий и имя активной — одним блоком для хранилища.

        Пустые памяти в файл не пишутся: у «Всей истории» её нет вовсе, и
        строчка `{}` в файле сессии только мешала бы читать его глазами.
        """
        memory = {
            name: dump
            for name, strategy in self._strategies.items()
            if (dump := strategy.dump())
        }
        return {"strategy": self._strategy_name, "memory": memory}

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
    """Приводит загруженную из хранилища историю к тому же виду, в котором
    её ведёт агент: только user/assistant, только role и content."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in ("user", "assistant")
    ]


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
        advice = (
            "Вопрос можно задать заново, сбросив диалог или переключив стратегию "
            "управления контекстом на сжимающую: тогда старая часть истории "
            "уйдёт в запрос в сжатом виде."
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
