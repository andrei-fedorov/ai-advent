# TooManyRules — сущность агента (день 6, неделя 2; день 8 — работа с токенами).
#
# Единственное место в проекте, где происходят вызовы LLM API. Модуль
# намеренно ничего не знает ни про Gradio, ни про Too Many Bones: внутри
# только конфиг, стек сообщений, вызов API, замер времени, подсчёт токенов
# и стоимости, логирование и обработка ошибок. Наружу агент отдаёт результат
# (`AgentReply`) и своё состояние для дебага (`debug_state`).
#
# Направление зависимостей одностороннее: app.py → presets.py → agent.py →
# tokens.py. Из проекта здесь импортируется ровно один модуль — `tokens.py`,
# лист графа, который сам не импортирует ничего (день 8; замороженный
# `app_week1.py` не импортируется по-прежнему) — см.
# `docs/TooManyRules — Неделя 2 архитектура.md`, §3, и спецификацию дня 8, §4.

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

import tokens

# Ключ читается только здесь, поэтому и .env подхватывается здесь же —
# интерфейс про API-ключи ничего не знает.
load_dotenv()

logger = logging.getLogger("toomanyrules.agent")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Дешёвая модель на каждый день — дефолт конфига.
DEEPSEEK_MODEL = "deepseek-v4-flash"
# Флагман: сильнее и дороже, используется пресетом «Флагман + thinking».
DEEPSEEK_MODEL_PRO = "deepseek-v4-pro"

# Цены DeepSeek V4 в долларах за 1M токенов (off-peak), перенесены из недели 1.
# Источник: https://api-docs.deepseek.com/quick_start/pricing. В пиковые часы
# (01:00-04:00 и 06:00-10:00 UTC, пн-пт) ставки ×2 — считаем по off-peak,
# это осознанно оценка, а не выписка по счёту.
PRICING_PER_M_TOKENS = {
    "deepseek-v4-flash": {"input": 0.22, "output": 0.66},
    "deepseek-v4-pro":   {"input": 0.66, "output": 1.98},
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


class HistoryStore(Protocol):
    """Сохранение истории между запусками (реализация — `storage.py`, день 7).

    Протокол объявлен на дне 6, когда реализаций в проекте ещё не было, —
    чтобы день 7 подставил в эту точку `JsonHistoryStore`, не переписывая
    агента. Сигнатуры с тех пор не менялись и меняться не должны: туда же,
    когда появится VPS, встанет SQLite. Агент по-прежнему работает и со
    `store=None` — тогда он живёт только в памяти процесса.
    """

    def load(self, session_id: str) -> list[dict]: ...
    def save(self, session_id: str, messages: list[dict]) -> None: ...


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
    ) -> None:
        self._config = config
        self._session_id = session_id
        self._store = store
        self._client: OpenAI | None = None
        self._last_reply: AgentReply | None = None
        # Счётчики за время жизни агента. `reset()` их не трогает — он
        # чистит только стек сообщений.
        self._totals = {
            "calls": 0,          # всего вызовов ask(), включая неуспешные
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
        }
        # Журнал ходов (день 8). Ведёт себя как счётчики агента, а не как стек
        # сообщений: `reset()` его не чистит, на диск он не едет, и после
        # перезапуска процесса он пуст, хотя история восстановлена из файла.
        self._turns: list[TurnStats] = []

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
            "[%s] агент создан: model=%s thinking=%s "
            "восстановлено из хранилища=%d сообщ., живых агентов: %d",
            self._log_name, config.model, _thinking_label(config.thinking),
            self._restored_messages,
            process_stats()["agents_alive"],
        )
        # Сбой загрузки логируется здесь, а не на месте: до регистрации в
        # реестре у агента ещё нет номера, а без номера строка в логе
        # ничего не опознаёт.
        if self._store_error is not None:
            logger.warning("[%s] %s", self._log_name, self._store_error)

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

        Без аргумента считается стек как есть (системный промпт + история) —
        это нижняя граница следующего запроса. С вопросом — то, что уйдёт в
        модель, если отправить его прямо сейчас; так панель показывает
        занятость до нажатия «Отправить».

        Считается по результату `_build_messages()` — по тому же списку
        сообщений, который ушёл бы в API: «посчитали» и «отправили» не должны
        разъезжаться.
        """
        return tokens.context_usage(
            self._count_messages(self._build_messages(question)),
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            calibration=self.calibration,
        )

    def ask(self, user_message: str) -> AgentReply:
        """Один ход диалога: собрать сообщения, сходить в API, дописать пару
        «вопрос/ответ» в стек.

        Исключений не бросает: ошибка API или отсутствующий ключ возвращаются
        как `AgentReply(ok=False, error=...)`. При ошибке стек не меняется —
        ни вопрос, ни ответ в него не попадают, чтобы повтор не задваивал
        историю.

        Параметры запроса берутся только из конфига: аргументов, меняющих
        модель/температуру/лимиты, у `ask()` нет — другой набор параметров
        означает другой экземпляр агента.
        """
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return self._failed_reply(str(exc), elapsed=0.0)

        messages = self._build_messages(user_message)
        # Счёт до запроса (день 8): считаем ровно тот список сообщений, который
        # сейчас уйдёт в API, — и логируем бюджет до вызова, а не после.
        request = self._count_messages(messages)
        budget = tokens.context_usage(
            request,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            calibration=self.calibration,
        )
        self._log_budget(budget)
        # Проверки «а влезет ли» здесь намеренно нет: переполненный запрос
        # отправляется как есть. День 8 показывает поломку, чинят её дни 9-10
        # (спецификация дня 8, §2.2) — предупреждение в логе и в панели есть,
        # предохранителя нет.
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
        )

        if self._config.keep_history:
            self._messages.append({"role": "user", "content": user_message})
            self._messages.append({"role": "assistant", "content": text})
            self._persist()

        self._record(reply)
        self._record_turn(reply, history_before)
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
        """Очищает стек сообщений. Счётчики за время жизни агента и метрики
        последнего вызова сохраняются — это разные вещи: сброшен диалог,
        а не агент."""
        self._messages = []
        self._persist()
        logger.info("[%s] стек сообщений очищен (reset)", self._log_name)

    def debug_state(self) -> dict:
        """Состояние агента для дебаг-панели: конфиг, стек, метрики
        последнего вызова, накопленное за время жизни и счётчики процесса."""
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
            "context": asdict(self.context_usage()),
            "turns": [asdict(turn) for turn in self._turns],
            "calibration": self.calibration,
        }

    # --- Внутреннее ------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """Единственное место, где собирается список сообщений для запроса:
        системный промпт из конфига + весь стек + новый вопрос.

        Сюда на днях 9-10 встанет сжатие контекста — поэтому сборка вынесена
        в отдельный метод, а не размазана по вызову (день 8 оказался про
        токены и из этого метода только читает). Сейчас никакого сжатия,
        обрезания и лимита истории нет: в LLM уходит вся переписка целиком —
        ровно то, что день 8 и меряет.
        """
        return (
            [{"role": "system", "content": self._config.system_prompt}]
            + self.history
            + [{"role": "user", "content": user_message}]
        )

    def _count_messages(self, messages: list[dict]) -> tokens.RequestTokens:
        """Разложение готового списка сообщений на систему / историю / вопрос.

        Единственное место, где список сообщений превращается в оценку: и
        `ask()`, и `context_usage()` ходят сюда, поэтому «сколько посчитали»
        и «сколько отправили» считаются одним и тем же способом по одному и
        тому же списку.
        """
        return tokens.count_request(
            system_prompt=messages[0]["content"],
            history=messages[1:-1],
            question=messages[-1]["content"],
        )

    def _log_budget(self, budget: tokens.ContextUsage) -> None:
        """Строка бюджета в лог перед вызовом API — и отдельное предупреждение,
        когда занято больше `WARN_RATIO`: приближение к лимиту должно быть
        видно в терминале, а не только в панели."""
        request = budget.request
        logger.info(
            "[%s] бюджет: система %s + история %s + вопрос %s + служебные %s "
            "≈ %s из %s доступных (%s); окно %s, резерв под ответ %s",
            self._log_name,
            _num(request.system), _num(request.history), _num(request.question),
            _num(request.overhead), _num(budget.used),
            _num(budget.available), _ratio_str(budget.ratio),
            _num(budget.limit), _num(budget.answer_reserve),
        )
        if budget.level in ("warn", "danger", "over"):
            logger.warning(
                "[%s] контекст занят на %s (уровень %s): запрос всё равно уходит "
                "в API — обрезки и сжатия в проекте нет, это дни 9-10",
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
            self._store.save(self._session_id, self.history)
        except Exception as exc:
            self._store_error = f"история не сохранилась: {exc}"
            logger.warning("[%s] %s", self._log_name, self._store_error)
        else:
            self._store_error = None

    def _failed_reply(
        self,
        error: str,
        elapsed: float,
        request: "tokens.RequestTokens | None" = None,
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
        )
        self._record(reply)
        logger.warning("[%s] ошибка: %s", self._log_name, error)
        return reply

    def _record_turn(self, reply: AgentReply, history_before: int) -> None:
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
            )
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
    return (
        f"Запрос не влез в контекстное окно модели и отклонён целиком. "
        f"Мы насчитали ≈{_num(budget.used)} токенов "
        f"(сырая оценка {_num(budget.estimated)}) при окне модели "
        f"{_num(budget.limit)} и резерве {_num(budget.answer_reserve)} под "
        f"ответ — доступно было {_num(budget.available)}. "
        f"Ответа нет: превышенный контекст означает отказ, а не забывание "
        f"начала диалога. Стек сообщений не изменился, файл сессии не тронут — "
        f"вопрос можно задать заново, но сначала диалог придётся сбросить: "
        f"обрезкой и сжатием истории займутся дни 9-10. "
        f"Текст ошибки API: {text}"
    )


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
