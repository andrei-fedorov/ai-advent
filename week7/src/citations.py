"""Чистый разбор ссылок на выдержки и подписи источников."""

import re


def cited_numbers(text) -> list[int]:
    # [1], а также списки и диапазоны: [1, 3], [2–4], [1; 5].
    numbers = set()
    for group in re.findall(r"\[(\d+(?:\s*[,;–—-]\s*\d+)*)\]", text):
        for item in re.split(r"\s*[,;]\s*", group):
            bounds = [int(number) for number in re.split(r"\s*[–—-]\s*", item)]
            low, high = bounds[0], bounds[-1]
            numbers.update(range(low, high + 1) if 0 <= high - low <= 20 else bounds)
    return sorted(numbers)


def cited_sources(text, hits):
    sources, warnings = [], []
    numbers = cited_numbers(text)
    for number in numbers:
        if 1 <= number <= len(hits):
            sources.append((number, hits[number - 1].label()))
        else:
            warnings.append(f"⚠️ [{number}] — такой выдержки не было")
    if not numbers:
        warnings.append("⚠️ ответ не ссылается на выдержки — сверить его с документацией нельзя")
    return sources, warnings


def source_lines(text, hits):
    # Порядок верных и неверных номеров совпадает с прежним выводом devhelp.
    numbers = cited_numbers(text)
    if not numbers:
        return ["⚠️ ответ не ссылается на выдержки — сверить его с документацией нельзя"]
    return ["Источники:"] + [
        f"  [{n}] {hits[n-1].label()}" if 1 <= n <= len(hits)
        else f"  ⚠️ [{n}] — такой выдержки не было" for n in numbers]
