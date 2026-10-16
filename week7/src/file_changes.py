"""Черновики Markdown, совместимый с git diff и применение одного файла."""

from dataclasses import dataclass, field
from datetime import datetime
import difflib
import hashlib
import os
from pathlib import Path, PureWindowsPath
import stat
import uuid


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FileDraft:
    kind: str
    original_sha256: str | None
    original_text: str
    new_text: str
    applied: bool = False


@dataclass
class Draft:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    files: dict[str, FileDraft] = field(default_factory=dict)


def validate_path(repo, path, status, kind):
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("недопустимый путь")
    relative = Path(path)
    if (relative.is_absolute() or PureWindowsPath(path).is_absolute()
            or ".." in relative.parts or ".." in PureWindowsPath(path).parts
            or any(char in path for char in "\r\n\t")):
        raise ValueError("путь должен быть относительным, без '..' и внутри репозитория")
    if ".git" in relative.parts:
        raise ValueError("нельзя менять служебный каталог git")
    if relative.suffix.lower() != ".md":
        raise ValueError("менять можно только Markdown (.md)")
    target = repo / relative
    # Проверяем и сами компоненты: статус MCP мог устареть после его получения.
    current = repo
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("путь — символьная ссылка; правь файл, на который она указывает")
    if status.get("symlink"):
        raise ValueError("путь — символьная ссылка; правь файл, на который она указывает")
    if not target.resolve().is_relative_to(repo.resolve()):
        raise ValueError("путь должен быть внутри репозитория")
    state = status.get("status")
    if state == "ignored":
        raise ValueError("путь игнорируется git")
    if kind == "edit":
        if state not in ("tracked", "untracked") or not target.is_file():
            raise ValueError("файл не найден")
    elif state != "absent" or target.exists():
        raise ValueError("файл уже существует")
    if not target.parent.is_dir():
        raise ValueError("родительский каталог не существует")
    return target


def read_text(path):
    with path.open(encoding="utf-8", newline="") as stream:
        text = stream.read()
    if "\0" in text:
        raise ValueError("двоичный файл")
    return text


def check_text(text):
    if not isinstance(text, str) or "\0" in text:
        raise ValueError("нужен текст UTF-8 без нулевых байтов")
    text.encode("utf-8")


def propose_edit(draft, repo, path, old, new, status, *, max_files):
    target = validate_path(repo, path, status, "edit")
    check_text(old)
    check_text(new)
    path = target.relative_to(repo).as_posix()
    if old == new:
        raise ValueError("old и new совпадают")
    if not old:
        raise ValueError("old не должен быть пустым")
    previous = draft.files.get(path)
    if previous and previous.kind != "edit":
        raise ValueError("для нового файла используй propose_file")
    if previous is None:
        if len(draft.files) >= max_files:
            raise ValueError(f"в черновике уже {max_files} файлов")
        original = read_text(target)
        previous = FileDraft("edit", sha(original), original, original)
    count, offset = 0, 0
    while (found := previous.new_text.find(old, offset)) >= 0:
        count += 1
        offset = found + 1
    if count != 1:
        raise ValueError("фрагмент не найден" if not count else f"фрагмент встречается {count} раз — добавь соседние строки")
    previous.new_text = previous.new_text.replace(old, new, 1)
    if previous.new_text == previous.original_text:
        draft.files.pop(path, None)
    else:
        draft.files[path] = previous
    return path


def propose_file(draft, repo, path, content, status, *, max_files, max_chars):
    target = validate_path(repo, path, status, "new")
    check_text(content)
    if len(content) > max_chars:
        raise ValueError(f"новый файл длиннее {max_chars} символов")
    path = target.relative_to(repo).as_posix()
    if path not in draft.files and len(draft.files) >= max_files:
        raise ValueError(f"в черновике уже {max_files} файлов")
    draft.files[path] = FileDraft("new", None, "", content)
    return path


def file_diff(path, item):
    before = "/dev/null" if item.kind == "new" else "a/" + path
    # Табуляция завершает имя: git понимает пути с пробелами и кириллицей.
    lines = list(difflib.unified_diff(item.original_text.splitlines(keepends=True),
                                    item.new_text.splitlines(keepends=True),
                                    fromfile=before + "\t", tofile="b/" + path + "\t", n=3))
    if not lines:
        return (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..e69de29\n"
                if item.kind == "new" else "")
    result = f"diff --git a/{path} b/{path}\n"
    if item.kind == "new":
        result += "new file mode 100644\n"
    for line in lines:
        result += line
        if not line.endswith("\n"):
            result += "\n\\ No newline at end of file\n"
    return result


def file_summary(path, item):
    diff = file_diff(path, item)
    body = diff.split("@@", 2)[-1].splitlines() if "@@" in diff else []
    return {"path": path, "kind": item.kind,
            "added": sum(line.startswith("+") for line in body),
            "removed": sum(line.startswith("-") for line in body),
            "diff": diff, "applied": item.applied}


def write_patch(draft, out_dir, header):
    if not draft.files:
        return None
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{draft.id}.diff"
    path = out_dir / name
    metadata = {"цель": draft.goal, **header, "дата": datetime.now().astimezone().isoformat()}
    with path.open("x", encoding="utf-8", newline="") as stream:
        for key, value in metadata.items():
            for line in str(value).splitlines():
                stream.write(f"# {key}: {line}\n")
        for file, item in draft.files.items():
            stream.write(file_diff(file, item))
    return path


def apply(draft, repo, path, status):
    item = draft.files.get(path)
    if item is None:
        return "отказ", "файла нет в черновике"
    if item.applied:
        return "уже применён", "уже применён"
    temporary = None
    created = False
    try:
        target = validate_path(repo, path, status, item.kind)
        if item.kind == "edit" and sha(read_text(target)) != item.original_sha256:
            raise ValueError("файл изменился после предложения — попроси правку заново")
        temporary = target.with_name(target.name + f".{draft.id}.tmp")
        # Не перезаписываем чужой временный файл или ссылку.
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            created = True
            stream.write(item.new_text)
        if item.kind == "edit":
            temporary.chmod(stat.S_IMODE(target.stat().st_mode))
        # Повторяем проверки непосредственно перед заменой.
        validate_path(repo, path, status, item.kind)
        if item.kind == "edit" and sha(read_text(target)) != item.original_sha256:
            raise ValueError("файл изменился после предложения — попроси правку заново")
        os.replace(temporary, target)
        item.applied = True
        return "записан", "записан"
    except (ValueError, OSError, UnicodeError) as exc:
        return "отказ", str(exc)
    finally:
        # Удаляется только временный файл, созданный этой операцией.
        if temporary is not None and created:
            temporary.unlink(missing_ok=True)
