"""Отдельный MCP-сервер CRM: JSON только для чтения, stdout занят JSON-RPC."""

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

TICKET_MAX_CHARS = 8_000
USER_MAX_EVENTS = 20
FIND_MAX_LIMIT = 10
STATUSES = {"open": "открыт", "waiting": "ждёт ответа пользователя", "closed": "закрыт"}
FIND_DESCRIPTION = "Найти обращения в CRM по словам из вопроса или текста ошибки, по пользователю или статусу. Вызывай, чтобы проверить, было ли такое у этого пользователя раньше и чем решилось у других: закрытые тикеты содержат решение."
TICKET_DESCRIPTION = "Прочитать тикет целиком: окружение, текст ошибки, переписку и решение. Вызывай для тикета, найденного через find_tickets."
USER_DESCRIPTION = "Прочитать окружение пользователя, события аккаунта и список его тикетов. Контакт не отдаётся."


def dump(value):
    return json.dumps(value, ensure_ascii=False)


def is_text(value):
    return isinstance(value, str) and bool(value.strip())


def load_crm(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("запись 0: CRM не прочитана как JSON") from exc
    if not isinstance(data, dict) or any(not isinstance(data.get(key), list) for key in ("users", "tickets")):
        raise ValueError("запись 0: нужны списки users и tickets")
    users, tickets = {}, {}
    for number, user in enumerate(data["users"], 1):
        if (not isinstance(user, dict) or not is_text(user.get("id")) or len(user["id"]) > 32
                or not is_text(user.get("name")) or user["id"] in users):
            raise ValueError(f"users, запись {number}: нужны уникальный id (до 32 символов) и имя")
        events = user.get("events", [])
        if (not isinstance(user.get("setup", {}), dict) or not isinstance(events, list)
                or any(not isinstance(event, dict) or not is_text(event.get("at")) or not is_text(event.get("text"))
                       for event in events)):
            raise ValueError(f"users, запись {number}: неверные setup или events")
        users[user["id"]] = user
    for number, ticket in enumerate(data["tickets"], 1):
        if (not isinstance(ticket, dict) or not is_text(ticket.get("id")) or len(ticket["id"]) > 32
                or ticket["id"] in tickets or not is_text(ticket.get("user_id"))
                or ticket["user_id"] not in users or not is_text(ticket.get("subject"))
                or not is_text(ticket.get("status")) or ticket["status"] not in STATUSES
                or not is_text(ticket.get("created"))):
            raise ValueError(f"tickets, запись {number}: неверная шапка, повтор id или неизвестный user_id")
        messages = ticket.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict) or message.get("from") not in ("user", "support")
                       or not is_text(message.get("at")) or not is_text(message.get("text"))
                       for message in messages)):
            raise ValueError(f"tickets, запись {number}: нужны непустые messages (from, at, text)")
        tags = ticket.get("tags", [])
        if (not isinstance(ticket.get("context", {}), dict) or not isinstance(tags, list)
                or any(not isinstance(tag, str) for tag in tags)
                or (ticket.get("resolution") is not None and not isinstance(ticket["resolution"], str))):
            raise ValueError(f"tickets, запись {number}: неверные context, tags или resolution")
        tickets[ticket["id"]] = ticket
    return users, tickets


def single_line(value):
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def render_ticket(ticket, earlier, last_text, question, field_limit):
    # Помеченные JSON-строки однозначны даже при переводах строк и кавычках в данных.
    def clip(value):
        value = str(value)
        return value if len(value) <= field_limit else value[:field_limit] + "… [обрезано]"

    context = ticket.get("context", {})
    lines = [
        f"Тикет {single_line(ticket['id'])} · {STATUSES[ticket['status']]} · создан {single_line(ticket['created'])} · "
        f"{single_line(ticket.get('channel', '—'))} · пользователь {single_line(ticket['user_id'])}",
        "Пользователь ID: " + dump(ticket["user_id"]),
        "Тема: " + dump(clip(ticket["subject"])),
        "Метки: " + ", ".join(single_line(clip(tag)) for tag in ticket.get("tags", [])),
        "Окружение (JSON): " + dump({key: clip(value) for key, value in context.items()}),
        "Последнее сообщение пользователя: " + dump(question),
        *[f"{key}: {clip(value)}" for key, value in context.items()],
        "Переписка:",
    ]
    skipped = len(ticket["messages"]) - 1 - len(earlier)
    if skipped:
        lines.append(f"… ещё {skipped} ранних сообщений не показано")
    for message in earlier:
        lines.append(f"{message['from']} · {message['at']}: {message['text']}")
    last = ticket["messages"][-1]
    lines.append(f"{last['from']} · {last['at']}: {last_text}")
    if ticket["status"] == "closed":
        lines.append("Решение: " + clip(ticket.get("resolution") or "не указано"))
    return "\n".join(lines)


def ticket_text(ticket):
    question = next((message["text"] for message in reversed(ticket["messages"]) if message["from"] == "user"), "")
    earlier = ticket["messages"][:-1]
    last_text = ticket["messages"][-1]["text"]
    result = render_ticket(ticket, earlier, last_text, question, TICKET_MAX_CHARS)
    while len(result) > TICKET_MAX_CHARS and earlier:
        earlier = earlier[1:]
        result = render_ticket(ticket, earlier, last_text, question, TICKET_MAX_CHARS)
    # Если одна запись сама длиннее потолка, последнее сообщение остаётся фрагментом.
    limit = 2_000
    while len(result) > TICKET_MAX_CHARS and limit > 0:
        def shortened(value):
            return value if len(value) <= limit else value[:limit] + "… [обрезано]"
        result = render_ticket(ticket, [], shortened(last_text), shortened(question), limit)
        limit //= 2
    if len(result) > TICKET_MAX_CHARS:
        raise ToolError("шапка тикета превышает потолок ответа")
    return result


def make_server(users, tickets):
    server = MCPServer("crm-сервер", log_level="WARNING")

    def check_id(value, name):
        if not value.strip() or len(value) > 32:
            raise ToolError(f"{name}: нужна непустая строка до 32 символов")

    def logged(name, arguments, action):
        started = time.monotonic()
        result, error = "", None
        try:
            result = action()
            return result
        except ToolError as exc:
            error = str(exc)
            raise
        finally:
            outcome = f"ошибка: {error}" if error else f"{len(result)} символов"
            print(f"[crm-сервер] {name} {dump(arguments)} · {time.monotonic() - started:.3f} с · {outcome}",
                  file=sys.stderr, flush=True)

    @server.tool(description=TICKET_DESCRIPTION)
    def get_ticket(ticket_id: str) -> str:
        def action():
            check_id(ticket_id, "ticket_id")
            if ticket_id not in tickets:
                raise ToolError(f"тикет не найден: {ticket_id}")
            return ticket_text(tickets[ticket_id])
        return logged("get_ticket", {"ticket_id": ticket_id}, action)

    @server.tool(description=USER_DESCRIPTION)
    def get_user(user_id: str) -> str:
        def action():
            check_id(user_id, "user_id")
            if user_id not in users:
                raise ToolError(f"пользователь не найден: {user_id}")
            user = users[user_id]
            events = sorted(user.get("events", []), key=lambda event: event["at"], reverse=True)[:USER_MAX_EVENTS]
            own = sorted((ticket for ticket in tickets.values() if ticket["user_id"] == user_id),
                         key=lambda ticket: ticket["created"], reverse=True)
            lines = [f"Пользователь {user_id} · {user['name']} · с {user.get('since', '—')}", "Окружение:"]
            lines += [f"{key}: {value}" for key, value in user.get("setup", {}).items()]
            lines.append("События аккаунта:")
            lines += [f"{event['at']}: {event['text']}" for event in events]
            lines.append("Тикеты пользователя:")
            lines += [f"{ticket['id']} · {ticket['created']} · {STATUSES[ticket['status']]} · {ticket['subject']}"
                      for ticket in own]
            return "\n".join(lines)
        return logged("get_user", {"user_id": user_id}, action)

    @server.tool(description=FIND_DESCRIPTION)
    def find_tickets(query: str | None = None, user_id: str | None = None, status: str | None = None,
                     limit: Annotated[int, Field(ge=1, le=FIND_MAX_LIMIT)] = 5) -> str:
        def action():
            if user_id is not None:
                check_id(user_id, "user_id")
                if user_id not in users:
                    raise ToolError(f"пользователь не найден: {user_id}")
            if status is not None and status not in STATUSES:
                raise ToolError("status: допустимы open, waiting, closed")
            # Слово совпадает с началом слова тикета: «ключ» находит «ключи», но не «подключить».
            words = {word for word in re.findall(r"\w+", (query or "").casefold()) if len(word) >= 3}
            patterns = [re.compile(r"(?<!\w)" + re.escape(word)) for word in words]
            found = []
            for ticket in tickets.values():
                if user_id is not None and ticket["user_id"] != user_id:
                    continue
                if status is not None and ticket["status"] != status:
                    continue
                haystack = "\n".join([ticket["subject"], *ticket.get("tags", []),
                                      *[f"{key}: {value}" for key, value in ticket.get("context", {}).items()],
                                      *[message["text"] for message in ticket["messages"]]]).casefold()
                score = sum(bool(pattern.search(haystack)) for pattern in patterns)
                if patterns and not score:
                    continue
                found.append((score, ticket))
            found.sort(key=lambda item: (item[0], item[1]["created"]), reverse=True)
            # Одна JSON-запись на строку; клиент не разбирает произвольную тему по разделителям.
            fields = ("id", "created", "status", "user_id", "subject", "resolution")
            return "\n".join("Тикет: " + dump({key: ticket.get(key) for key in fields})
                             for _, ticket in found[:limit]) or "Тикетов не найдено"
        return logged("find_tickets", {"query": query, "user_id": user_id, "status": status, "limit": limit}, action)

    return server


def main():
    parser = argparse.ArgumentParser(description="MCP CRM, только чтение")
    parser.add_argument("--crm", type=Path, required=True)
    args = parser.parse_args()
    try:
        users, tickets = load_crm(args.crm)
    except ValueError as exc:
        print(f"[crm-сервер] отказ: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    make_server(users, tickets).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
