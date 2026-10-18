"""Bundled artwork identifiers for navigation folders (not playable media)."""

from __future__ import annotations

from pathlib import Path

PREFIX = "ximalaya-navigation/"
ICONS = (
    "subscriptions",
    "recent",
    "categories",
    "radio",
    "hot",
    "recommendations",
    "book",
    "children",
    "microphone",
    "history",
    "music",
    "mystery",
    "growth",
    "life",
    "health",
    "finance",
    "language",
    "theatre",
    "film",
    "news",
    "traffic",
    "sport",
    "travel",
    "art",
    "crown",
    "gift",
    "new",
    "complete",
    "next",
    "back",
)


def folder_icon(path: str, name: str) -> str:
    """Use a subject icon where known, falling back to the navigation family."""
    roots = {
        "subscriptions": "subscriptions",
        "recent_listening": "recent",
        "discovery": "categories",
        "discovery/radio": "radio",
        "discovery/hot": "hot",
        "discovery/recommendations": "recommendations",
        "discovery/groups": "categories",
    }
    if path in roots:
        return roots[path]
    for words, icon in (
        (("上一页",), "back"),
        (("下一页",), "next"),
        (("儿童",), "children"),
        (("小说", "有声图书"), "book"),
        (("播客", "主播"), "microphone"),
        (("人文", "历史", "国学"), "history"),
        (("音乐",), "music"),
        (("悬疑",), "mystery"),
        (("成长", "大师课", "潜力"), "growth"),
        (("生活", "都市"), "life"),
        (("健康",), "health"),
        (("财经", "经济", "商业"), "finance"),
        (("外语", "方言"), "language"),
        (("广播剧", "相声", "曲艺"), "theatre"),
        (("影视", "娱乐"), "film"),
        (("新闻", "热点"), "news"),
        (("交通",), "traffic"),
        (("体育",), "sport"),
        (("旅游",), "travel"),
        (("文艺",), "art"),
        (("VIP",), "crown"),
        (("免费",), "gift"),
        (("新品",), "new"),
        (("完结",), "complete"),
    ):
        if any(word in name for word in words):
            return icon
    if path.startswith("discovery/radio"):
        return "radio"
    if path.startswith(("discovery/hot/", "discovery/groups/")):
        return "hot"
    return "categories"


def local_image(path: str) -> str | None:
    """Resolve an allowlisted packaged icon path."""
    if not path.startswith(PREFIX):
        return None
    name = path.removeprefix(PREFIX)
    if name not in {f"{icon}.png" for icon in ICONS}:
        raise ValueError("Unknown navigation image")
    return str(Path(__file__).parent / "images" / name)
