"""Граница клиентского SDK MCP v2, процесс и соединение на один вопрос."""

import asyncio
from dataclasses import dataclass
import time

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment


@dataclass
class ToolCall:
    ok: bool
    text: str
    elapsed: float
    error: str | None = None


def error_text(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(error_text(child) for child in exc.exceptions)
    return str(exc) or type(exc).__name__


class McpSession:
    def __init__(self, command: str, args: list[str], env: dict | None, timeout_s: float):
        # Только белый список SDK, даже если вызывающий передал весь os.environ.
        allowed = get_default_environment()
        self.env = {key: (env or {}).get(key, value) for key, value in allowed.items()}
        self.command, self.args, self.timeout_s = command, args, timeout_s
        self.client = None
        self.error = None
        self.elapsed = 0.0
        self.tools_elapsed = 0.0

    async def __aenter__(self):
        started = time.monotonic()
        client = Client(StdioServerParameters(command=self.command, args=self.args, env=self.env),
                        read_timeout_seconds=self.timeout_s)
        try:
            async with asyncio.timeout(self.timeout_s):
                await client.__aenter__()
            self.client = client
        except Exception as exc:
            detail = error_text(exc)
            stage = "запуск" if "No such file" in detail or "FileNotFound" in detail else "соединение"
            self.error = f"{stage}: {detail}"
        self.elapsed = time.monotonic() - started
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client is not None:
            try:
                await self.client.__aexit__(exc_type, exc, tb)
            except Exception as cleanup:
                self.error = f"соединение: {error_text(cleanup)}"
            finally:
                self.client = None

    async def tools(self) -> list[dict]:
        started = time.monotonic()
        if self.client is None:
            return []
        result = []
        cursor = None
        seen = set()
        try:
            while True:
                async with asyncio.timeout(self.timeout_s):
                    page = await self.client.list_tools(cursor=cursor)
                result.extend({"name": tool.name, "description": tool.description or "",
                               "inputSchema": tool.input_schema} for tool in page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    return result
                if cursor in seen:
                    raise ValueError("tools/list повторил курсор")
                seen.add(cursor)
        except Exception as exc:
            self.error = f"соединение (tools/list): {error_text(exc)}"
            return []
        finally:
            self.tools_elapsed = time.monotonic() - started

    async def call(self, name: str, arguments: dict) -> ToolCall:
        started = time.monotonic()
        if self.client is None:
            return ToolCall(False, "", 0, self.error or "соединение: сервер недоступен")
        try:
            async with asyncio.timeout(self.timeout_s):
                result = await self.client.call_tool(name, arguments)
            text = "\n".join(block.text for block in result.content if block.type == "text")
            return ToolCall(not result.is_error, text, time.monotonic() - started,
                            f"вызов: {text}" if result.is_error else None)
        except Exception as exc:
            return ToolCall(False, "", time.monotonic() - started, f"вызов: {error_text(exc)}")
