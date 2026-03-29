"""Interfejs użytkownika — karty wyników, budowanie kontekstu LLM, CSS."""

import re
from typing import Any

import streamlit as st

from config import GDPR_URL, ISAP_ACT_URL, UODO_PORTAL_BASE
from models import (
    CONTEXT_TYPE_ORDER,
    TPL_ACT_ARTICLE,
    TPL_DECISION,
    TPL_GDPR,
    TPL_HEADER,
    AgentMemory,
    QueryDecomposition,
    SearchResult,
)

UODO_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;600;700;800&display=swap');

    :root {
        --uodo-blue-10: #f5f8f8;
        --uodo-blue-20: #e8f1fd;
        --uodo-blue-30: #dde3ee;
        --uodo-blue-33: #a5b3dd;
        --uodo-blue-35: #6d83cc;
        --uodo-blue-38: #356bcc;
        --uodo-blue-40: #0058cc;
        --uodo-blue-50: #275faa;
        --uodo-blue-60: #0e4591;
        --uodo-blue-80: #092e60;
        --uodo-dark-gray: #3f444f;
        --uodo-light-gray: #c8ccd3;
        --uodo-red: #f25a5a;
        --uodo-red-logo: #cd071e;
        --uodo-red-dark: #b22222;
        --uodo-white: #fff;
        --uodo-black: rgba(26,26,28,1);
        --body-color: rgba(26,26,28,1);
        --content-width: 1070px;
        --link-color: var(--uodo-blue-60);
        --link-hover-color: var(--uodo-blue-40);
        --separator-color: var(--uodo-blue-30);
        --uodo-border-radius: 2px;
    }
</style>
"""

PAGE_HEADER_HTML = """
<div class="page-header">
  <div class="page-header-inner">
    <div>
      <h1>Portal Orzeczeń UODO</h1>
      <div class="page-header-sub">Wyszukiwarka decyzji Prezesa UODO · Ustawa o ochronie danych osobowych · RODO</div>
    </div>
  </div>
</div>
"""


# ─────────────────────────── BUDOWANIE KONTEKSTU LLM ─────────────

_FRAGMENT_STOPWORDS = {
    "jakie",
    "są",
    "w",
    "o",
    "i",
    "z",
    "do",
    "na",
    "co",
    "ile",
    "jak",
    "czy",
    "przez",
    "dla",
    "po",
    "przy",
    "od",
    "ze",
    "to",
}


def _extract_fragment(content: str, query: str, max_len: int = 2000) -> str:
    """Wybiera najbardziej trafny fragment treści decyzji względem zapytania."""
    if not content or len(content) <= max_len:
        return content
    keywords = [
        w.lower()
        for w in re.split(r"\W+", query)
        if w.lower() not in _FRAGMENT_STOPWORDS and len(w) > 2
    ]
    if not keywords:
        return content[:max_len]
    step = 150
    best_score, best_pos = -1, 0
    cl = content.lower()
    for pos in range(0, max(1, len(content) - max_len), step):
        score = sum(cl[pos : pos + max_len].count(kw) for kw in keywords)
        if score > best_score:
            best_score, best_pos = score, pos
    fragment = content[best_pos : best_pos + max_len]
    if best_pos > 0:
        nl = fragment.find("\n")
        if 0 < nl < 150:
            fragment = fragment[nl:].lstrip()
        fragment = "[…]\n" + fragment
    return fragment


def build_context(
    docs: list[dict[str, Any]],
    query: str,
    thread_id: int | None,
    max_chars: int = 18000,
    filters: dict[str, Any] | None = None,
    memory: AgentMemory | None = None,
) -> str:
    """Buduje kontekst dla LLM — szablony Jinja2 + priorytetyzacja decyzji UODO."""
    f = filters or {}
    filter_lines = []
    if f.get("status"):
        filter_lines.append(f"Status decyzji: {f['status']}")
    if f.get("term_decision_type"):
        filter_lines.append(f"Rodzaj decyzji: {', '.join(f['term_decision_type'])}")
    if f.get("term_violation_type"):
        filter_lines.append(f"Rodzaj naruszenia: {', '.join(f['term_violation_type'])}")
    if f.get("term_legal_basis"):
        filter_lines.append(f"Podstawa prawna: {', '.join(f['term_legal_basis'])}")
    if f.get("term_corrective_measure"):
        filter_lines.append(f"Środek naprawczy: {', '.join(f['term_corrective_measure'])}")
    if f.get("term_sector"):
        filter_lines.append(f"Sektor: {', '.join(f['term_sector'])}")
    if f.get("keyword"):
        filter_lines.append(f"Słowo kluczowe: {f['keyword']}")

    filter_note = (
        "UWAGA: Wyniki zawężone filtrami: " + "; ".join(filter_lines) + ".\n"
        if filter_lines
        else ""
    )

    memory_note = ""

    if memory and thread_id:
        related = memory.find_related(query, thread_id)
        if related:
            snippets = []
            for e in related[:3]:
                f"- Poprzednie pytanie: «{e.query}» → znalezione decyzje: "
                if e.search_result.full:
                    for res in e.search_result.full[:3]:
                        (", ".join(res.get("signature", "")))

                else:
                    "brak"
            memory_note = (
                "KONTEKST Z POPRZEDNICH ANALIZ (tej sesji):\n" + "\n".join(snippets) + "\n"
            )

    header = TPL_HEADER.render(query=query, filter_note=filter_note, memory_note=memory_note)
    parts = [header]
    chars = len(header)

    # Decyzje UODO pierwsze, RODO ostatnie
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            CONTEXT_TYPE_ORDER.get(d.get("doc_type", ""), 9),
            -d.get("_score", 0),
        ),
    )

    for i, doc in enumerate(docs_sorted, 1):
        dtype = doc.get("doc_type", "")

        if dtype == "legal_act_article":
            chunk_idx = doc.get("chunk_index", 0)
            total = doc.get("chunk_total", 1)
            suffix = f"(część {chunk_idx + 1}/{total})" if total > 1 else ""
            block = TPL_ACT_ARTICLE.render(
                rank=i,
                art_num=doc.get("article_num", "?"),
                label_suffix=suffix,
                text=doc.get("content_text", ""),
            )
        elif dtype in ("gdpr_article", "gdpr_recital"):
            art_num = doc.get("article_num", "?")
            prefix = "Motyw" if dtype == "gdpr_recital" else f"Art. {art_num}"
            block = TPL_GDPR.render(rank=i, prefix=prefix, text=doc.get("content_text", ""))
        else:
            keywords = doc.get("keywords_text", "") or ", ".join(doc.get("keywords", []))
            acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
            block = TPL_DECISION.render(
                rank=i,
                sig=doc.get("signature", "?"),
                date=doc.get("date_issued", "")[:7],
                status=doc.get("status", ""),
                graph_rel=doc.get("_graph_relation", ""),
                keywords=keywords[:200] if keywords else "",
                acts=", ".join(acts[:5]) if acts else "",
                fragment=_extract_fragment(doc.get("content_text", ""), query),
            )

        if chars + len(block) > max_chars:
            parts.append(f"\n[pominięto {len(docs_sorted) - i + 1} dalszych wyników]")
            break
        parts.append(block)
        chars += len(block)

    return "\n---\n".join(parts)


# ─────────────────────────── KARTY WYNIKÓW ───────────────────────
def decision_url(doc: dict[str, Any]) -> str:
    """Tworzy adres url decyzji.

    1. Jeżeli dokument ma pole 'source_url' zwraca jego zawartość
    2. Jeżeli brakuje tego pola url generowany jest na podstawie sygnatury
    """
    sig = doc.get("signature", "")
    url = doc.get("source_url", "")
    if url:
        return url
    import re as _re

    slug = sig.lower().replace(".", "_")
    year_m = _re.search(r"\b(20\d{2})\b", sig)
    year = year_m.group(1) if year_m else "2024"
    return f"{UODO_PORTAL_BASE}/urn:ndoc:gov:pl:uodo:{year}:{slug}/content"


def render_decision_card(doc: dict[str, Any]) -> None:
    """Wyświetla dane dokumentu typu decyzja."""
    sig = doc.get("signature", "?")
    status = doc.get("status", "")
    date = doc.get("date_published", "") or doc.get("date_issued", "")
    source = doc.get("_source", "")
    graph_rel = doc.get("_graph_relation", "")
    title = doc.get("title_full", "") or doc.get("title", "")
    name = doc.get("title", sig)
    url = decision_url(doc)

    kw_list = doc.get("keywords", [])
    if isinstance(kw_list, str):
        kw_list = [k.strip() for k in kw_list.split(",") if k.strip()]
    taxonomy_values = {
        v.lower() for v in doc.get("term_decision_type", []) + doc.get("term_sector", [])
    }
    kw_list = [k for k in kw_list if k.lower() not in taxonomy_values]

    status_cls = {
        "prawomocna": "status-final",
        "nieprawomocna": "status-nonfinal",
        "uchylona": "status-repealed",
    }.get(status, "status-unknown")

    date_fmt = ""
    if date:
        try:
            from datetime import datetime

            d = datetime.strptime(date[:10], "%Y-%m-%d")
            months = [
                "stycznia",
                "lutego",
                "marca",
                "kwietnia",
                "maja",
                "czerwca",
                "lipca",
                "sierpnia",
                "września",
                "października",
                "listopada",
                "grudnia",
            ]
            date_fmt = f"{d.day} {months[d.month - 1]} {d.year}"
        except Exception:
            date_fmt = date[:10]

    graph_badge = (
        f' <span class="status-badge status-unknown">↗ {graph_rel or "graf"}</span>'
        if source == "graph"
        else ""
    )

    st.markdown(
        f"""
    <article class="doc-list-item">
      <header>
        <a href="{url}" target="_blank">{sig}</a>
        <time><small>opublikowano</small> {date_fmt}</time>
      </header>
      <main>
        <h2 class="d-flex justify-content-between align-items-start gap-2">
          <a href="{url}" target="_blank">{name}</a>
          <span class="status-badge {status_cls}">{status.upper()}{graph_badge}</span>
        </h2>
        <p class="text-muted">{title[:280] + "…" if len(title) > 280 else title}</p>
      </main>
    </article>""",
        unsafe_allow_html=True,
    )

    with st.container():
        if kw_list:
            shown = kw_list[:8]
            rest = len(kw_list) - len(shown)
            tags = " · ".join(f"`{k}`" for k in shown)
            suffix = f" *+{rest} więcej*" if rest > 0 else ""
            st.caption(f"🏷️ {tags}{suffix}")
        all_acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
        if all_acts:
            st.caption("📜 Powołane akty: " + " · ".join(f"`{a}`" for a in all_acts))
        if graph_rel:
            st.caption(f"↗ powiązana przez graf: *{graph_rel}*")
    st.divider()


def render_act_article_card(doc: dict[str, Any]) -> None:
    """Wyświetla dane dokumentu typu artykuł."""
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")[:600]
    label = f"Art. {art_num} u.o.d.o." + (f" (część {chunk_idx + 1}/{total})" if total > 1 else "")

    st.markdown(
        f"""
    <article class="doc-list-item">
      <header>
        <a href="{ISAP_ACT_URL}" target="_blank">{label}</a>
        <span><small>Ustawa o ochronie danych osobowych</small></span>
      </header>
      <main>
        <h2><a href="{ISAP_ACT_URL}" target="_blank">Dz.U. 2019 poz. 1781</a>
          <span class="status-badge status-final ms-2">u.o.d.o.</span>
        </h2>
        <p>{text}{"…" if len(doc.get("content_text", "")) > 600 else ""}</p>
      </main>
      <footer><small class="text-muted">score: {score:.3f}</small></footer>
    </article>""",
        unsafe_allow_html=True,
    )


def render_gdpr_card(doc: dict[str, Any]) -> None:
    """Wyświetla dane dokumentu typu GDPR."""
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")[:500]
    dtype = doc.get("doc_type", "")
    chapter = doc.get("chapter", "")
    chapter_title = doc.get("chapter_title", "")
    is_recital = dtype == "gdpr_recital"
    label = art_num if is_recital else f"Art. {art_num} RODO"
    badge_txt = "motyw RODO" if is_recital else "RODO"
    if not is_recital and total > 1:
        label += f" (część {chunk_idx + 1}/{total})"
    chapter_html = (
        f'<small class="text-muted">Rozdział {chapter} — {chapter_title}</small>'
        if chapter and chapter_title
        else ""
    )

    st.markdown(
        f"""
    <article class="doc-list-item">
      <header>
        <a href="{GDPR_URL}" target="_blank">{label}</a>
        <span class="status-badge status-final">{badge_txt}</span>
      </header>
      <main>
        <h2>{chapter_html}</h2>
        <p>{text}{"…" if len(doc.get("content_text", "")) > 500 else ""}</p>
      </main>
      <footer><small class="text-muted">score: {score:.3f}</small></footer>
    </article>""",
        unsafe_allow_html=True,
    )


def render_card(doc: dict[str, Any]) -> None:
    """Dispatcher — wybiera typ karty na podstawie doc_type."""
    dtype = doc.get("doc_type", "")
    if dtype == "legal_act_article":
        render_act_article_card(doc)
    elif dtype in ("gdpr_article", "gdpr_recital"):
        render_gdpr_card(doc)
    else:
        render_decision_card(doc)


def render_tags(res: SearchResult, kw_filter: str | None = None):
    """Wyświetla podsumowanie wyników wyszukiwania dokumentów."""
    _tag_info: str | None = None
    if kw_filter:
        _tag_info = f" · tag: `{kw_filter}`" if kw_filter.strip() else ""

    if res.tags:
        with st.expander("🏷️ Tagi", expanded=False):
            for t in res.tags:
                _ = st.caption(f"`{t}`")

    caption = f"""
    Znaleziono {len(res.full)} dokumentów 
    ({len(res.decisions)} decyzji, {len(res.act_arts)} u.o.d.o., 
    {len(res.gdpr_docs)} RODO, {len(res.graph_docs)} przez graf) · {res.search_time:.2f}s"""

    if _tag_info:
        caption += _tag_info

    _ = st.caption(caption)


def render_reasoning(decomp: QueryDecomposition, effective_query: str):
    """Wyświetla dekomopzycje zapytania."""
    with st.expander("🧠 Reasoning Step — jak zrozumiałem pytanie", expanded=False):
        _ = st.caption(f"**Typ zapytania:** {decomp.query_type.value}")
        _ = st.caption(f"**Rozumowanie:** {decomp.reasoning}")
        if decomp.search_keywords:
            _ = st.caption(
                "**Słowa kluczowe:** " + " · ".join(f"`{k}`" for k in decomp.search_keywords)
            )
        if decomp.gdpr_articles_hint:
            _ = st.caption("**Wskazane artykuły RODO:** " + ", ".join(decomp.gdpr_articles_hint))
        if decomp.uodo_act_articles_hint:
            _ = st.caption(
                "**Wskazane artykuły u.o.d.o.:** " + ", ".join(decomp.uodo_act_articles_hint)
            )
        if decomp.enriched_query != effective_query:
            _ = st.caption(f"**Wzbogacone zapytanie:** _{decomp.enriched_query}_")


def render_documents(res: SearchResult):
    """Wyświetla listę dokumentów dotyczących zapytania."""
    with st.expander(f"📋 Dokumenty ({len(res.full)})", expanded=False):
        tabs = st.tabs(
            [
                f"Wszystkie ({len(res.full)})",
                f"Decyzje UODO ({len(res.decisions)})",
                f"Ustawa u.o.d.o. ({len(res.act_arts)})",
                f"RODO ({len(res.gdpr_docs)})",
                f"Graf ({len(res.graph_docs)})",
            ]
        )

        with tabs[0]:
            for i, doc in enumerate(res.full, 1):
                render_card(doc)
        with tabs[1]:
            if res.decisions:
                for doc in res.decisions:
                    render_decision_card(doc)
            else:
                _ = st.info("Brak decyzji UODO dla tego zapytania.")
        with tabs[2]:
            if res.act_arts:
                for doc in res.act_arts:
                    render_act_article_card(doc)
            else:
                _ = st.info("Brak artykułów ustawy dla tego zapytania.")
        with tabs[3]:
            if res.gdpr_docs:
                for doc in res.gdpr_docs:
                    render_gdpr_card(doc)
            else:
                _ = st.info("Brak artykułów RODO dla tego zapytania.")
        with tabs[4]:
            if res.graph_docs:
                _ = st.info("Decyzje powiązane przez cytowania z wynikami semantic search.")
                for doc in res.graph_docs:
                    render_decision_card(doc)
            else:
                _ = st.info("Brak wyników z grafu powiązań.")
