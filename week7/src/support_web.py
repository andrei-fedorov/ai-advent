"""Граница HTTP: локальный сервер, проверки запросов и страница чата."""

import asyncio
import json
import socket
import time
import traceback

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn


def json_response(data, status=200):
    return JSONResponse(data, status_code=status, media_type="application/json; charset=utf-8")


def valid_chat(data, history_messages):
    if not isinstance(data, dict) or set(data) - {"who", "question", "history"}:
        return False
    who, question, history = data.get("who"), data.get("question"), data.get("history", [])
    if who is not None and (not isinstance(who, str) or not who.strip() or len(who.strip()) > 32):
        return False
    if not isinstance(question, str) or not question.strip():
        return False
    return (isinstance(history, list) and len(history) <= history_messages
            and all(isinstance(item, dict) and set(item) == {"role", "content"}
                    and item["role"] in ("user", "assistant") and isinstance(item["content"], str)
                    for item in history))


def make_app(*, page, port, max_body_bytes, history_messages, chat, health):
    busy = False
    tasks = set()

    async def home(request):
        return HTMLResponse(page.read_text(encoding="utf-8"))

    async def state(request):
        return json_response({**health(), "busy": busy})

    async def reply(request):
        nonlocal busy
        if busy:
            return json_response({"error": "ассистент уже отвечает"}, 409)
        # Чтение ограничено по фактическим байтам, включая chunked-запросы.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body_bytes:
                return json_response({"error": f"тело запроса больше {max_body_bytes // 1024} КБ"}, 413)
            body.extend(chunk)
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError):
            return json_response({"error": "тело запроса должно быть JSON"}, 400)
        if not valid_chat(data, history_messages):
            return json_response({"error": "нужны question (непустая строка), who (строка до 32 символов или null) "
                                           f"и history (до {history_messages} реплик user/assistant)"}, 400)
        who = data["who"].strip() if data.get("who") is not None else None
        question, history = data["question"].strip(), data.get("history", [])
        if busy:
            return json_response({"error": "ассистент уже отвечает"}, 409)
        busy = True

        async def complete():
            nonlocal busy
            try:
                return await chat(who, question, history)
            finally:
                busy = False
        task = asyncio.create_task(complete())
        tasks.add(task)

        # Отключение вкладки не отменяет работу; сильная ссылка живёт до завершения.
        def finished(done):
            tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return json_response(await asyncio.shield(task))

    def guarded(handler):
        async def endpoint(request):
            try:
                return await handler(request)
            except Exception as exc:
                if hasattr(exc, 'status_code') and hasattr(exc, 'public_message'):
                    return json_response({'error': exc.public_message}, exc.status_code)
                traceback.print_exc()
                return json_response({'error': str(exc)}, 500)
        return endpoint

    async def http_error(request, exc):
        return json_response({'error': 'адрес не найден' if exc.status_code == 404 else 'метод не поддерживается'}, exc.status_code)

    app = Starlette(exception_handlers={HTTPException: http_error},
                    routes=[Route("/", guarded(home)), Route("/api/chat", guarded(reply), methods=["POST"]),
                            Route("/api/health", guarded(state))])

    class LocalHttp:
        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await app(scope, receive, send)
            started = time.monotonic()
            status = 500
            async def logged_send(message):
                nonlocal status
                if message['type'] == 'http.response.start':
                    status = message['status']
                await send(message)
            headers = [(k.lower(), v.decode('latin1')) for k, v in scope.get('headers', [])]
            hosts = [v for k, v in headers if k == b'host']
            types = [v for k, v in headers if k == b'content-type']
            try:
                if len(hosts) != 1 or hosts[0] not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                    response = json_response({'error': 'недопустимый Host'}, 421)
                elif scope['method'] == 'POST' and (len(types) != 1 or types[0].split(';')[0].strip().lower() != 'application/json'):
                    response = json_response({'error': 'нужен Content-Type: application/json'}, 415)
                else:
                    response = None
                    await app(scope, receive, logged_send)
                if response is not None:
                    await response(scope, receive, logged_send)
            finally:
                print(f"[support] HTTP {scope['method']} {scope['path']} · {status} · {time.monotonic()-started:.3f} с", flush=True)
    return LocalHttp()


def serve(app, port, on_ready):
    # На macOS SO_REUSEADDR даёт занять 127.0.0.1 поверх чужого 0.0.0.0 — сначала пробное подключение.
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
    except OSError:
        pass
    else:
        raise ValueError(f"порт {port} занят — --port <другой>")
    # Один и тот же сокет для проверки порта и uvicorn: между ними порт не освободится.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', port))
            listener.listen(128)
        except OSError as exc:
            raise ValueError(f'порт {port} занят — --port <другой>') from exc
        on_ready()
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, access_log=False, log_level='warning'))
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
