#!/usr/bin/env python3
"""
UODO RAG Demo — wyszukiwarka decyzji Prezesa UODO + ustawa o ochronie danych osobowych.

Uruchomienie:
  streamlit run main.py

Wymagania:
  pip install streamlit qdrant-client sentence-transformers networkx groq requests python-dotenv
"""

import time

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_URL,
    RE_QUERY_SIG,
)
from llm import call_llm_stream, decompose_query, get_available_models
from models import AgentMemory, MemoryEntry, QueryDecomposition
from search import (
    fetch_by_signature,
    get_all_tags,
    get_collection_stats,
    get_taxonomy_options,
    hybrid_search,
)
from ui import (
    PAGE_HEADER_HTML,
    UODO_CSS,
    build_context,
    render_act_article_card,
    render_card,
    render_decision_card,
    render_gdpr_card,
)


def main() -> None:
    st.set_page_config(
        page_title="Portal Orzeczeń UODO — Wyszukiwarka",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _ = st.markdown(UODO_CSS, unsafe_allow_html=True)
    _ = st.markdown(PAGE_HEADER_HTML, unsafe_allow_html=True)
    _ = st.markdown("---")

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None

    history_placeholder = st.empty()
    _render_history(history_placeholder, None)
    answer_placeholder = st.empty()
    full_answer = ""
    _ = _render_answer(answer_placeholder, full_answer)

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        _ = st.markdown("## ⚙️ Opcje")

        provider = st.selectbox(
            "Provider LLM", ["Ollama", "Groq"], key="provider_select"
        )

        # Klucz API tylko dla Groq — Ollama używa OLLAMA_CLOUD_API_KEY z .env
        if provider == "Groq":
            api_key = st.text_input(
                "Klucz API Groq",
                type="password",
                value=st.session_state.get("llm_api_key", ""),
                key="api_key_input",
            )
        else:
            api_key = ""
            _ = st.caption(f"🖥️ Ollama: `{OLLAMA_URL}`")

        models = get_available_models(provider, api_key)
        default_model = (
            DEFAULT_OLLAMA_MODEL if provider == "Ollama" else DEFAULT_GROQ_MODEL
        )
        default_idx = next((i for i, m in enumerate(models) if default_model in m), 0)
        selected_model = st.selectbox("Model", models, index=default_idx)

        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"] = selected_model
        st.session_state["llm_api_key"] = api_key

        _ = st.markdown("---")
        use_graph = st.toggle("Graf powiązań", value=True)

        _ = st.markdown("### 📂 Typ dokumentów")
        show_decisions = st.checkbox("Decyzje UODO", value=True)
        show_act = st.checkbox("Ustawa o ochronie danych (u.o.d.o.)", value=True)
        show_gdpr = st.checkbox("RODO (rozporządzenie UE 2016/679)", value=True)

        _ = st.markdown("---")
        try:
            stats = get_collection_stats()
            _ = st.markdown("### 📊 Baza wiedzy")
            st.metric("Decyzje UODO", stats.get("decisions", 0))
            st.metric("Artykuły u.o.d.o.", stats.get("act_chunks", 0))
            if stats.get("edges"):
                st.metric("Powiązania w grafie", stats.get("edges", 0))
        except Exception:
            pass

        # ── Pamięć epizodyczna ───────────────────────────────────────
        memory_placeholder = st.empty()

        if "agent_memory" not in st.session_state:
            st.session_state["agent_memory"] = AgentMemory()
        memory: AgentMemory = st.session_state["agent_memory"]
        _render_memory_history(memory_placeholder, history_placeholder, memory)

    # ── Filtry typów dokumentów ──────────────────────────────────
    doc_types = []
    if show_decisions:
        doc_types.append("uodo_decision")
    if show_act:
        doc_types.append("legal_act_article")
    if show_gdpr:
        doc_types.extend(["gdpr_article", "gdpr_recital"])
    if not doc_types:
        doc_types = [
            "uodo_decision",
            "legal_act_article",
            "gdpr_article",
            "gdpr_recital",
        ]

    taxonomy = get_taxonomy_options()

    # ── Pole wyszukiwania ────────────────────────────────────────
    if "_example_query" in st.session_state:
        st.session_state["query_input"] = st.session_state.pop("_example_query")

    col_q, col_ai, col_btn = st.columns([7, 1.5, 1.2])
    with col_q:
        query = st.text_input(
            "Treść",
            placeholder="Wpisz treść, sygnaturę lub temat...",
            key="query_input",
            label_visibility="collapsed",
        )
    with col_ai:
        use_llm = st.checkbox("🤖 Użyj AI", value=True, key="use_llm_cb")
    with col_btn:
        search_btn = st.button("🔍 Szukaj", type="primary", use_container_width=True)

    # ── Filtry zaawansowane ──────────────────────────────────────
    with st.expander("🔽 Filtry zaawansowane", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            _ = st.markdown(
                '<div class="filter-label">Sygnatura</div>', unsafe_allow_html=True
            )
            sig_filter = st.text_input(
                "Sygnatura",
                placeholder="np. DKN.5110",
                label_visibility="collapsed",
                key="sig_filter",
            )
            _ = st.markdown(
                '<div class="filter-label">Status</div>', unsafe_allow_html=True
            )
            status_filter = st.selectbox(
                "Status",
                ["— wszystkie —", "prawomocna", "nieprawomocna", "uchylona"],
                label_visibility="collapsed",
                key="status_filter",
            )
            _ = st.markdown(
                '<div class="filter-label">Słowa kluczowe</div>', unsafe_allow_html=True
            )
            all_tags = get_all_tags()
            kw_filter = (
                st.selectbox(
                    "Słowo kluczowe",
                    options=[""] + all_tags,
                    label_visibility="collapsed",
                    key="kw_filter",
                )
                or ""
            )
        with fc2:
            _ = st.markdown(
                '<div class="filter-label">Rodzaj decyzji</div>', unsafe_allow_html=True
            )
            tax_decision = st.multiselect(
                "Rodzaj decyzji",
                options=taxonomy.get("term_decision_type", []),
                label_visibility="collapsed",
                key="tax_decision",
            )
            _ = st.markdown(
                '<div class="filter-label">Środek naprawczy</div>',
                unsafe_allow_html=True,
            )
            tax_measure = st.multiselect(
                "Środek naprawczy",
                options=taxonomy.get("term_corrective_measure", []),
                label_visibility="collapsed",
                key="tax_measure",
            )
            _ = st.markdown(
                '<div class="filter-label">Podstawa prawna</div>',
                unsafe_allow_html=True,
            )
            tax_legal_basis = st.multiselect(
                "Podstawa prawna",
                options=taxonomy.get("term_legal_basis", []),
                label_visibility="collapsed",
                key="tax_legal_basis",
            )
        with fc3:
            _ = st.markdown(
                '<div class="filter-label">Rodzaj naruszenia</div>',
                unsafe_allow_html=True,
            )
            tax_violation = st.multiselect(
                "Rodzaj naruszenia",
                options=taxonomy.get("term_violation_type", []),
                label_visibility="collapsed",
                key="tax_violation",
            )
            _ = st.markdown(
                '<div class="filter-label">Sektor</div>', unsafe_allow_html=True
            )
            tax_sector = st.multiselect(
                "Sektor",
                options=taxonomy.get("term_sector", []),
                label_visibility="collapsed",
                key="tax_sector",
            )
            _ = st.markdown(
                '<div class="filter-label">Data ogłoszenia (od–do)</div>',
                unsafe_allow_html=True,
            )
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                date_from = st.text_input(
                    "Od",
                    placeholder="2020-01-01",
                    label_visibility="collapsed",
                    key="date_from",
                )
            with dcol2:
                date_to = st.text_input(
                    "Do",
                    placeholder="2026-12-31",
                    label_visibility="collapsed",
                    key="date_to",
                )

    # ── Przykładowe pytania ──────────────────────────────────────
    with st.expander("💡 Przykładowe pytania", expanded=not bool(query)):
        _ = st.caption("Kliknij pytanie aby je wyszukać:")
        examples = [
            ("🔔", "Kiedy wymagane jest zgłoszenie naruszenia danych?"),
            ("⚖️", "Jakie kary może nałożyć Prezes UODO?"),
            ("🔐", "Brak podstawy prawnej przetwarzania danych"),
            ("✅", "Zgoda na przetwarzanie danych osobowych"),
            ("🧬", "Dane genetyczne"),
            ("🗳️", "Dane osobowe w kampanii wyborczej"),
            ("📋", "Obowiązek informacyjny administratora"),
            ("🤝", "Umowa powierzenia przetwarzania danych"),
            ("🕵️", "Inspektor ochrony danych — konflikt interesów"),
            ("📸", "Zdjęcie tablicy rejestracyjnej w internecie a RODO"),
            ("📜", "DKN.5131.15.2025"),
        ]
        cols = st.columns(2)
        for idx, (emoji, question) in enumerate(examples):
            if cols[idx % 2].button(
                f"{emoji} {question}", key=f"example_{idx}", use_container_width=True
            ):
                st.session_state["_example_query"] = question
                st.rerun()

    # ── Budowanie słownika filtrów ───────────────────────────────
    filters: dict = {"doc_types": doc_types}
    if "uodo_decision" in doc_types:
        if status_filter != "— wszystkie —":
            filters["status"] = status_filter
        if tax_decision:
            filters["term_decision_type"] = tax_decision
        if tax_violation:
            filters["term_violation_type"] = tax_violation
        if tax_legal_basis:
            filters["term_legal_basis"] = tax_legal_basis
        if tax_measure:
            filters["term_corrective_measure"] = tax_measure
        if tax_sector:
            filters["term_sector"] = tax_sector
        if date_from.strip():
            try:
                filters["year_from"] = int(date_from.strip()[:4])
            except ValueError:
                pass
        if date_to.strip():
            try:
                filters["year_to"] = int(date_to.strip()[:4])
            except ValueError:
                pass
    if kw_filter.strip():
        filters["keyword"] = kw_filter.strip()

    # ── Wyszukiwanie ─────────────────────────────────────────────
    effective_query = query
    if sig_filter.strip() and not query.strip():
        effective_query = sig_filter.strip()

    if effective_query and (
        search_btn
        or st.session_state.get("last_query") != effective_query
        or st.session_state.get("last_filters") != str(filters)
    ):
        st.session_state["last_query"] = effective_query
        st.session_state["last_filters"] = str(filters)

        # Reasoning Step — dekompozycja PRZED wyszukiwaniem
        decomp: QueryDecomposition | None = None
        if use_llm and len(effective_query.split()) > 3:
            with st.spinner("🧠 Analizuję pytanie..."):
                decomp = decompose_query(effective_query)
            if decomp and decomp.reasoning:
                with st.expander(
                    "🧠 Reasoning Step — jak zrozumiałem pytanie", expanded=False
                ):
                    _ = st.caption(f"**Typ zapytania:** {decomp.query_type.value}")
                    _ = st.caption(f"**Rozumowanie:** {decomp.reasoning}")
                    if decomp.search_keywords:
                        _ = st.caption(
                            "**Słowa kluczowe:** "
                            + " · ".join(f"`{k}`" for k in decomp.search_keywords)
                        )
                    if decomp.gdpr_articles_hint:
                        _ = st.caption(
                            "**Wskazane artykuły RODO:** "
                            + ", ".join(decomp.gdpr_articles_hint)
                        )
                    if decomp.uodo_act_articles_hint:
                        _ = st.caption(
                            "**Wskazane artykuły u.o.d.o.:** "
                            + ", ".join(decomp.uodo_act_articles_hint)
                        )
                    if decomp.enriched_query != effective_query:
                        _ = st.caption(
                            f"**Wzbogacone zapytanie:** _{decomp.enriched_query}_"
                        )

        search_query = decomp.enriched_query if decomp else effective_query
        if decomp and decomp.year_from_hint and "year_from" not in filters:
            filters["year_from"] = decomp.year_from_hint
        if decomp and decomp.year_to_hint and "year_to" not in filters:
            filters["year_to"] = decomp.year_to_hint

        with st.spinner("🔍 Wyszukuję..."):
            t0 = time.time()
            _tags: list[str] = []
            sig_match = RE_QUERY_SIG.match(effective_query)
            if sig_match:
                sig_norm = sig_match.group(1).upper()
                exact = fetch_by_signature(sig_norm)
                if exact:
                    exact["_source"] = "exact"
                    exact["_score"] = 1.0
                    docs = [exact]
                    if use_graph:
                        for rsig in exact.get("related_uodo_rulings", [])[:5]:
                            rdoc = fetch_by_signature(rsig)
                            if rdoc:
                                rdoc["_source"] = "graph"
                                rdoc["_score"] = 0.9
                                docs.append(rdoc)
                else:
                    _ = st.warning(
                        f"Nie znaleziono decyzji o sygnaturze **{sig_norm}** w bazie."
                    )
                    docs, _tags = hybrid_search(
                        search_query, filters=filters, use_graph=use_graph
                    )
            else:
                docs, _tags = hybrid_search(
                    search_query, filters=filters, use_graph=use_graph
                )
            search_time = time.time() - t0

        if not docs:
            _ = st.warning(
                "Nie znaleziono dokumentów. Spróbuj zmienić filtry lub sformułowanie."
            )
            return

        decisions = [d for d in docs if d.get("doc_type") == "uodo_decision"]
        act_arts = [d for d in docs if d.get("doc_type") == "legal_act_article"]
        gdpr_docs = [
            d for d in docs if d.get("doc_type") in ("gdpr_article", "gdpr_recital")
        ]
        graph_docs = [d for d in docs if d.get("_source") == "graph"]

        _tag_info = f" · tag: `{kw_filter}`" if kw_filter.strip() else ""
        _ = st.caption(
            f"""Znaleziono {len(docs)} dokumentów 
            ({len(decisions)} decyzji, {len(act_arts)} u.o.d.o., 
            {len(gdpr_docs)} RODO, {len(graph_docs)} przez graf) · {search_time:.2f}s"""
            + _tag_info
        )
        if _tags:
            _ = st.caption("🏷️ Tagi: " + " · ".join(f"`{t}`" for t in _tags))

        if use_llm:
            thread_id: int | None = st.session_state["thread_id"]
            print(f"Thread id: {thread_id}")
            context = build_context(
                docs, effective_query, thread_id, filters=filters, memory=memory
            )
            thread = None

            if thread_id:
                thread = memory.entries[thread_id]

            _render_history(history_placeholder, thread)

            full_answer = _render_answer(answer_placeholder, effective_query, context)

            if full_answer:
                entry = MemoryEntry(
                    query=effective_query,
                    enriched_query=search_query,
                    decomposition_summary=decomp.reasoning if decomp else "",
                    top_signatures=[
                        d.get("signature", "")
                        for d in decisions[:5]
                        if d.get("signature")
                    ],
                    top_articles=[
                        f"Art. {d.get('article_num')}"
                        for d in act_arts[:3]
                        if d.get("article_num")
                    ],
                    answer_snippet=full_answer[:300],
                    full_answer=full_answer,
                )

                id = memory.add(entry, thread_id)
                st.session_state["thread_id"] = id

                _render_memory_history(memory_placeholder, history_placeholder, memory)

        _ = st.markdown(f"### 📋 Dokumenty ({len(docs)})")
        tabs = st.tabs(
            [
                f"Wszystkie ({len(docs)})",
                f"Decyzje UODO ({len(decisions)})",
                f"Ustawa u.o.d.o. ({len(act_arts)})",
                f"RODO ({len(gdpr_docs)})",
                f"Graf ({len(graph_docs)})",
            ]
        )

        with tabs[0]:
            for i, doc in enumerate(docs, 1):
                render_card(doc, i)
        with tabs[1]:
            if decisions:
                for i, doc in enumerate(decisions, 1):
                    render_decision_card(doc, i)
            else:
                _ = st.info("Brak decyzji UODO dla tego zapytania.")
        with tabs[2]:
            if act_arts:
                for i, doc in enumerate(act_arts, 1):
                    render_act_article_card(doc, i)
            else:
                _ = st.info("Brak artykułów ustawy dla tego zapytania.")
        with tabs[3]:
            if gdpr_docs:
                for i, doc in enumerate(gdpr_docs, 1):
                    render_gdpr_card(doc, i)
            else:
                _ = st.info("Brak artykułów RODO dla tego zapytania.")
        with tabs[4]:
            if graph_docs:
                _ = st.info(
                    "Decyzje powiązane przez cytowania z wynikami semantic search."
                )
                for i, doc in enumerate(graph_docs, 1):
                    render_decision_card(doc, i)
            else:
                _ = st.info("Brak wyników z grafu powiązań.")


def _render_history(
    placeholder: DeltaGenerator,
    thread: list[MemoryEntry] | None = None,
):
    with placeholder.container(horizontal_alignment="right"):
        if thread:
            for entry in thread:
                with st.container(border=True, width="content"):
                    _ = st.markdown(entry.query)

                with st.container(border=True):
                    _ = st.markdown(entry.full_answer)
        else:
            pass


def _render_answer(
    placeholder: DeltaGenerator,
    effective_query: str | None = None,
    context: str | None = None,
) -> str | None:
    with placeholder.container(horizontal_alignment="right"):
        if effective_query and context:
            with st.container(border=True, width="content"):
                _ = st.markdown(effective_query)

            try:
                answer = ""
                answer_placeholder = st.empty()

                for chunk in call_llm_stream(effective_query, context):
                    answer += chunk

                    with answer_placeholder.container(border=True):
                        _ = st.markdown(answer)
                return answer
            except Exception as e:
                _ = st.error(f"Błąd LLM: {e}")
                return None
        else:
            return None


def on_thread_select(
    history_placeholder: DeltaGenerator, thread_id: int, thread: list[MemoryEntry]
):
    """Callback when thread button is clicked"""
    st.session_state["thread_id"] = thread_id
    st.session_state["query_input"] = ""

    if history_placeholder:
        _ = history_placeholder.empty()
        _render_history(history_placeholder, thread)


def on_new_thread(history_placeholder: DeltaGenerator, memory: AgentMemory):
    id = memory.new_thread()

    st.session_state["thread_id"] = id
    st.session_state["query_input"] = ""

    if history_placeholder:
        _ = history_placeholder.empty()


def _render_memory_history(
    placeholder: DeltaGenerator,
    history_placeholder: DeltaGenerator | None,
    memory: AgentMemory,
) -> None:
    """Render memory history in the sidebar placeholder."""
    with placeholder.container():
        _ = st.markdown("---")
        _ = st.markdown("### 🧠 Historia sesji")
        for i, thread in enumerate(memory.entries):
            if not thread:
                continue

            short_q = thread[0].query[:40] + ("…" if len(thread[0].query) > 40 else "")

            if st.button(
                f"💬 {short_q}",
                key=f"thread_{i}",
                on_click=on_thread_select,
                args=(history_placeholder, i, thread),
            ):
                pass

        if st.button(
            "Nowy wątek",
            key="new_thread_btn",
            on_click=on_new_thread,
            args=(history_placeholder, memory),
        ):
            pass


if __name__ == "__main__":
    main()
