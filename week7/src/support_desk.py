"""Поддержка: чат клиента — данные CRM, два поиска, ответ модели для локальной страницы."""

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import fnmatch
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from citations import cited_sources
from llm import LlmParams, answer
from mcp_client import McpSession, ToolCall
from search import DocsIndex
import settings as cfg
from support_web import make_app, serve

TICKET_LINE = "Тикет: "


class SupportError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code, self.public_message = status_code, message


def log(text):
    print(f"[support] {text}", flush=True)


def marked(text, label):
    prefix = label + ": "
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise ValueError(f"в ответе CRM нет поля «{label}»")


def excerpt_block(start, hits):
    return "\n\n".join(f"[{n}] {hit.label()}\n{hit.text}" for n, hit in enumerate(hits, start)) or "Выдержек нет."


def unavailable(error):
    return SupportError(502, "данные поддержки недоступны: " + str(error))


class SupportDesk:
    def __init__(self, repo, crm, index, params, warnings):
        self.repo, self.crm, self.index, self.params = repo, crm, index, params
        self.index_warnings = warnings
        self.faq_count = sum(chunk["kind"] == "faq" for chunk in index.chunks)
        self.exclude_docs = tuple(doc for doc in index.files
                                  if any(fnmatch.fnmatch(doc, pattern) for pattern in cfg.SUPPORT_DOCS_EXCLUDE))

    @asynccontextmanager
    async def session(self):
        async with McpSession(cfg.CRM_SERVER[0], [*cfg.CRM_SERVER[1:], "--crm", str(self.crm)],
                              env=None, timeout_s=cfg.MCP_TIMEOUT_S) as session:
            log(f"MCP подключение {session.elapsed:.3f} с · {session.error or 'готово'}")
            if session.error:
                raise unavailable(session.error)
            yield session

    async def call(self, session, name, arguments):
        result = await session.call(name, arguments)
        first = result.text.splitlines()[0] if result.text else ""
        # Для поиска первая строка содержит тему обращения: выводим только id.
        if name == "find_tickets" and first.startswith(TICKET_LINE):
            first = "Тикет " + str(json.loads(first[len(TICKET_LINE):])["id"])
        log(f"MCP {name} {json.dumps(arguments, ensure_ascii=False)} · {result.elapsed:.3f} с · "
            f"{len(result.text)} символов · {first[:200] if result.ok else result.error}")
        return result

    async def client(self, session, who):
        # Номер обращения или id клиента: сначала обращение, затем клиент.
        ticket = await self.call(session, "get_ticket", {"ticket_id": who})
        context = {"who": who, "ticket_id": None, "ticket": None, "subject": None, "environment": {}}
        if ticket.ok:
            try:
                user_id = marked(ticket.text, "Пользователь ID")
                context.update(ticket_id=who, ticket=ticket.text, subject=marked(ticket.text, "Тема"),
                               environment=marked(ticket.text, "Окружение (JSON)"))
            except (ValueError, TypeError) as exc:
                raise unavailable(exc) from exc
        elif "тикет не найден:" in (ticket.error or ""):
            user_id = who
        else:
            raise unavailable(ticket.error or "нет ответа CRM")
        user = await self.call(session, "get_user", {"user_id": user_id})
        if not user.ok and context["ticket"] is None:
            if "пользователь не найден:" in (user.error or ""):
                raise SupportError(404, f"не нашли обращение или клиента: {who}")
            raise unavailable(user.error or "нет ответа CRM")
        user_error = None if user.ok else "данные клиента недоступны: " + str(user.error)
        if user_error:
            log(user_error)
        context.update(user_id=user_id, user=user.text if user.ok else None, user_error=user_error)
        return context

    def health(self):
        return {"chunks": len(self.index.chunks), "faq_chunks": self.faq_count,
                "warnings": self.index_warnings, "model": self.params.model}

    async def chat(self, who, question, history):
        if who is None:
            return await self.respond(question, history, None, None)
        async with self.session() as session:
            context = await self.client(session, who)
            return await self.respond(question, history, context, session)

    async def respond(self, question, history, context, session):
        previous = next((item["content"] for item in reversed(history) if item["role"] == "user"), None)
        parts = [question]
        if previous:
            # Уточнение («а где эта кнопка?») само по себе поиску мало.
            parts.append(previous)
        if context and context["subject"]:
            parts.append(context["subject"])
        if context:
            parts += [str(context["environment"][key]) for key in cfg.CRM_ERROR_KEYS if key in context["environment"]]
        # Вопрос идёт первым: при нехватке места сокращается хвост.
        query = " · ".join(parts)[:cfg.SUPPORT_QUERY_MAX_CHARS]
        log(f"ответ · {context['who'] if context else 'гость'} · реплик в истории {len(history)} · "
            f"запрос {query[:80]!r}")
        faq, docs = await asyncio.to_thread(self.search, query)
        faq_hits = faq.hits[:cfg.SUPPORT_FAQ_TOP_K]
        hits = [*faq_hits, *docs.hits]
        excerpts = [{"n": n, "kind": hit.kind, "label": hit.label(), "score": hit.score, "rank": hit.rank}
                    for n, hit in enumerate(hits, 1)]

        tools, calls = [], []
        if session:
            catalog = await session.tools()
            log(f"MCP каталог {session.tools_elapsed:.3f} с · {', '.join(tool['name'] for tool in catalog)} · "
                f"{session.error or 'готово'}")
            if session.error:
                raise unavailable(session.error)
            tools = [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                                       "parameters": tool["inputSchema"]}}
                     for tool in catalog if tool["name"] in cfg.SUPPORT_MODEL_TOOLS]

        async def call_tool(name, arguments):
            # Модель видит только обращения этого клиента: проверяет код, а не промпт.
            if name == "find_tickets":
                arguments = {**arguments, "user_id": context["user_id"]}
            result = await self.call(session, name, arguments)
            if name == "get_ticket" and result.ok:
                try:
                    owner = marked(result.text, "Пользователь ID")
                except (ValueError, TypeError):
                    owner = None
                if owner != context["user_id"]:
                    log(f"MCP get_ticket {arguments.get('ticket_id')} · отказ: обращение другого клиента")
                    result = ToolCall(ok=False, text="", elapsed=result.elapsed, error="обращение другого клиента")
            calls.append({"name": name, "arguments": arguments, "ok": result.ok})
            return result

        if context is None:
            crm = cfg.NO_CLIENT
        else:
            crm = "Клиент (CRM, get_user):\n" + (context["user"] or context["user_error"])
            if context["ticket"]:
                crm = "Обращение (CRM, get_ticket):\n" + context["ticket"] + "\n\n" + crm
        prompt = cfg.SUPPORT_USER_TEMPLATE.format(repo=self.repo.name, crm=crm, faq=excerpt_block(1, faq_hits),
                                                  docs=excerpt_block(len(faq_hits) + 1, docs.hits), question=question)
        messages = [{"role": "system", "content": cfg.SUPPORT_SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": prompt}]
        result = await answer(messages, tools, call_tool, self.params)
        if not result.ok:
            log("ответ · ошибка LLM: " + str(result.error))
            raise SupportError(502, "ошибка LLM: " + str(result.error))

        sources, warnings = cited_sources(result.text, hits)
        if result.finish_reason == "length":
            warnings.append("⚠️ ответ достиг ограничения длины")
        crm_line = "CRM: клиент не назвался"
        if context:
            counts = Counter(call["name"] for call in calls)
            model_calls = ", ".join(f"{name} ×{count}" for name, count in counts.items()) or "нет"
            ticket_part = f"обращение {context['ticket_id']} (get_ticket) · " if context["ticket_id"] else ""
            crm_line = (f"CRM: {ticket_part}клиент {context['user_id']} ({context['user_error'] or 'get_user'}) · "
                        f"вызвано моделью: {model_calls}")
        cost = "стоимость неизвестна" if result.cost_usd is None else f"≈ ${result.cost_usd:.6f}"
        stats = (f"Поиск {faq.elapsed + docs.elapsed:.2f} с · модель {sum(r.elapsed for r in result.rounds):.2f} с · "
                 f"раундов {len(result.rounds)} · {result.prompt_tokens:,} → {result.completion_tokens:,} токенов · {cost}")
        log(f"ответ готов · источников {len(sources)} · пометок {len(warnings)} · {stats}")
        return {"who": context["who"] if context else None, "question": question, "query": query,
                "answer": result.text, "sources": [{"n": n, "label": label} for n, label in sources],
                "warnings": warnings, "crm": crm_line, "tool_calls": calls, "excerpts": excerpts, "stats": stats}

    def search(self, query):
        results = []
        for label, kinds, exclude, top_k in (("FAQ поддержки", ("faq",), (), cfg.SUPPORT_FAQ_TOP_K),
                                             ("документация", ("current", "docs"), self.exclude_docs, None)):
            result = self.index.search(query, kinds=kinds, exclude_docs=exclude)
            log(f"{label}: поиск {result.elapsed:.2f} с · реранк {result.rerank_elapsed:.2f} с")
            if result.rerank_error:
                log("⚠️ реранкер недоступен, взяты ближайшие векторы: " + result.rerank_error)
            for n, hit in enumerate(result.hits[:top_k], 1):
                score = "—" if hit.score is None else f"{hit.score:.3f}"
                log(f"{label} [{n}] sim={hit.sim:.3f} score={score} до реранка #{hit.rank} · {hit.label()}")
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Чат поддержки для клиентов")
    parser.add_argument("--repo", type=Path, default=cfg.DEFAULT_REPO)
    parser.add_argument("--db", type=Path, default=cfg.INDEX_DB)
    parser.add_argument("--crm", type=Path, default=cfg.CRM_PATH)
    parser.add_argument("--port", type=int, default=cfg.SUPPORT_PORT)
    args = parser.parse_args()
    load_dotenv(cfg.WEEK7_DIR / ".env", override=False)
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        log("отказ: нет LLM_API_KEY в week7/.env")
        return 2
    params = LlmParams(os.environ.get("LLM_BASE_URL", cfg.LLM_BASE_URL), key,
                       os.environ.get("LLM_MODEL", cfg.LLM_MODEL), cfg.LLM_TEMPERATURE, cfg.LLM_MAX_TOKENS,
                       cfg.LLM_MAX_TOOL_ROUNDS, cfg.TOOL_RESULT_MAX_CHARS, cfg.PRICES, cfg.LLM_TIMEOUT_S,
                       log_prefix="[support]")
    repo, db, crm = (path.expanduser().resolve() for path in (args.repo, args.db, args.crm))
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port должен быть от 1 до 65535")
        if not repo.is_dir():
            raise ValueError("--repo не существует или не каталог")
        if not crm.is_file():
            raise ValueError("файл CRM не найден — --crm <путь>")
        index = DocsIndex(db, cfg.MODELS_DIR, rerank_model=cfg.RERANK_MODEL,
                          candidates=cfg.SEARCH_CANDIDATES, top_k=cfg.SEARCH_TOP_K)
        warnings = index.warnings(repo, cfg.DOCS_GLOBS + cfg.CODE_GLOBS, cfg.DOCS_EXCLUDE + cfg.CODE_EXCLUDE)
        if not any(chunk["kind"] == "faq" for chunk in index.chunks):
            warnings.append("FAQ поддержки нет в индексе (settings.FAQ_DOCS) — ./week7/run.sh index")
        for warning in warnings:
            log("⚠️ " + warning)
        index.load_models(allow_download=False)
        desk = SupportDesk(repo, crm, index, params, warnings)
        app = make_app(page=cfg.SUPPORT_PAGE, port=args.port, max_body_bytes=cfg.SUPPORT_MAX_BODY_BYTES,
                       history_messages=cfg.SUPPORT_HISTORY_MESSAGES, chat=desk.chat, health=desk.health)
        ready = (f"репозиторий {repo} · индекс {db}, {len(index.chunks)} кусков, из них FAQ {desk.faq_count} · "
                 f"CRM {crm} · LLM {params.model} · адрес {params.base_url} · чат http://127.0.0.1:{args.port}")
        serve(app, args.port, lambda: log(ready))
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        log(f"отказ: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
