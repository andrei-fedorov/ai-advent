"""Граница OpenAI-совместимого API: раунды, инструменты, расход токенов."""

from dataclasses import dataclass, field
import json
import re
import time
from urllib.parse import urlparse

from openai import AsyncOpenAI


@dataclass(frozen=True)
class LlmParams:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    max_tool_rounds: int
    tool_result_max_chars: int
    prices: dict[str, tuple[float, float]]
    timeout_s: float
    log_prefix: str = "[devhelp]"


@dataclass
class LlmRound:
    elapsed: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    finish_reason: str
    calls: list[dict]
    model: str
    cost_usd: float | None


@dataclass
class LlmResult:
    ok: bool = True
    text: str = ""
    finish_reason: str = ""
    rounds: list[LlmRound] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    answered_model: str = ""
    error: str | None = None


def has_tool_markup(text: str) -> bool:
    # Внутренний формат DeepSeek иногда попадает в content вместо tool_calls.
    # Это сигнал ошибки протокола, а не дополнительный способ вызвать инструмент.
    return re.search(r"<[|｜]+DSML[|｜]+\s*(?:calls|tool_calls|invoke|parameter)\b", text) is not None


async def answer(messages: list[dict], tools: list[dict], call_tool, params: LlmParams,
                 *, final_tools=(), final_tool_rounds=0, tool_text_retries=0) -> LlmResult:
    result = LlmResult()
    conversation = list(messages)
    phases = [("основной", tools)] * params.max_tool_rounds if tools else []
    if tools and final_tools:
        phases += [("черновик", list(final_tools))] * final_tool_rounds
    phases += [("ответ", [])]
    previous_phase = None
    corrections_left = tool_text_retries
    try:
        async with AsyncOpenAI(base_url=params.base_url, api_key=params.api_key,
                               timeout=params.timeout_s, max_retries=0) as client:
            for phase, active_tools in phases:
                if phase != previous_phase:
                    if phase == "черновик":
                        conversation.append({"role": "system", "content":
                            "Лимит поиска и чтения исчерпан. Остались только инструменты подготовки результата. "
                            "Если цель требует правок или нового файла, предложи их сейчас по уже собранным "
                            "данным через доступные инструменты. Не запрашивай дополнительное чтение."})
                    elif phase == "ответ" and tools:
                        conversation.append({"role": "system", "content":
                            "Инструменты больше недоступны. Дай итоговый ответ обычным текстом по уже "
                            "собранным данным и результатам вызовов. Если задача осталась незавершённой, "
                            "скажи об этом. Не воспроизводи вызовы инструментов и их внутреннюю разметку."})
                    previous_phase = phase
                catalog = {tool["function"]["name"] for tool in active_tools}
                while True:
                    number = len(result.rounds) + 1
                    request = {"model": params.model, "messages": conversation,
                               "temperature": params.temperature, "max_tokens": params.max_tokens}
                    # В финальном текстовом запросе нет ни tools, ни tool_choice.
                    if active_tools:
                        request.update(tools=active_tools, tool_choice="auto")
                    if urlparse(params.base_url).hostname == "api.deepseek.com":
                        request["extra_body"] = {"thinking": {"type": "disabled"}}
                    started = time.monotonic()
                    try:
                        response = await client.chat.completions.create(**request)
                    except Exception as exc:
                        print(f"{params.log_prefix} LLM {params.model} · раунд {number} · сбой · "
                              f"{time.monotonic() - started:.2f} с: {exc}", flush=True)
                        raise
                    elapsed = time.monotonic() - started
                    choice = response.choices[0]
                    message = choice.message
                    usage = response.usage
                    prompt = usage.prompt_tokens if usage else 0
                    completion = usage.completion_tokens if usage else 0
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached = (getattr(usage, "prompt_cache_hit_tokens", None)
                              or getattr(details, "cached_tokens", 0) or 0)
                    prices = params.prices.get(response.model)
                    cost = (prompt * prices[0] + completion * prices[1]) / 1_000_000 if prices else None
                    calls = [{"name": call.function.name, "arguments": call.function.arguments}
                             for call in message.tool_calls or []]
                    reason = choice.finish_reason or ""
                    result.rounds.append(LlmRound(elapsed, prompt, completion, cached, reason, calls, response.model, cost))
                    result.prompt_tokens += prompt
                    result.completion_tokens += completion
                    result.cached_tokens += cached
                    result.cost_usd = sum(round.cost_usd for round in result.rounds) if all(
                        round.cost_usd is not None for round in result.rounds) else None
                    result.answered_model, result.finish_reason = response.model, reason
                    cost_label = "неизвестна" if cost is None else f"${cost:.6f}"
                    print(f"{params.log_prefix} LLM {response.model} · раунд {number} · finish_reason={reason} · "
                          f"{elapsed:.2f} с · prompt={prompt} completion={completion} cache={cached} · "
                          f"стоимость {cost_label} · вызовы {json.dumps(calls, ensure_ascii=False)}", flush=True)
                    if message.tool_calls:
                        break
                    text = message.content or ""
                    if has_tool_markup(text):
                        print(f"{params.log_prefix} LLM · раунд {number} · неверный формат вызова "
                              f"(DSML в тексте) · осталось исправлений {corrections_left}", flush=True)
                        if corrections_left <= 0:
                            raise ValueError("модель вернула разметку вызова инструмента вместо ответа; "
                                             "задача не завершена")
                        corrections_left -= 1
                        # Ошибочный текст не добавляется в историю и не исполняется.
                        instruction = ("Вызови нужный инструмент через структурированное поле tool_calls "
                                       "из текущего каталога, затем продолжи задачу." if active_tools else
                                       "Инструментов нет. Верни итог обычным текстом без попыток вызвать их.")
                        conversation.append({"role": "system", "content":
                            "Предыдущий ответ имел неверный формат: вместо tool_calls вернулась внутренняя "
                            "разметка в тексте. Вызов не был исполнен. " + instruction})
                        continue
                    if not text.strip():
                        raise ValueError("модель вернула пустой ответ")
                    result.text = text
                    return result
                if not active_tools:
                    raise ValueError("модель вызвала инструменты после исчерпания лимита")
                assistant = {"role": "assistant", "content": message.content,
                             "tool_calls": [{"id": call.id, "type": "function",
                                             "function": {"name": call.function.name,
                                                          "arguments": call.function.arguments}}
                                            for call in message.tool_calls]}
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning is not None:
                    assistant["reasoning_content"] = reasoning
                conversation.append(assistant)
                for call in message.tool_calls:
                    try:
                        if call.function.name not in catalog:
                            raise ValueError("имя инструмента отсутствует в каталоге")
                        arguments = json.loads(call.function.arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("аргументы инструмента должны быть JSON-объектом")
                        tool_result = await call_tool(call.function.name, arguments)
                        content = tool_result.text if tool_result.ok else f"Ошибка инструмента: {tool_result.error}"
                    except (ValueError, TypeError) as exc:
                        content = f"Ошибка инструмента: {exc}"
                        print(f"{params.log_prefix} {call.function.name}: {content}", flush=True)
                    if len(content) > params.tool_result_max_chars:
                        marker = "\n[результат обрезан]"
                        content = content[:max(0, params.tool_result_max_chars - len(marker))] + marker
                    conversation.append({"role": "tool", "tool_call_id": call.id, "content": content})
    except Exception as exc:
        result.ok, result.error = False, str(exc)
    return result
