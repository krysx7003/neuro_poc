"""UODO RAG Demo — wyszukiwarka decyzji Prezesa UODO + ustawa o ochronie danych osobowych.

Uruchomienie:
  streamlit run main.py

Wymagania:
  pip install streamlit qdrant-client sentence-transformers networkx groq requests python-dotenv
"""

import time
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from config import (
    DEFAULT_OLLAMA_MODEL,
    RE_QUERY_SIG,
)
from llm import call_llm_stream, decompose_query, get_available_models
from models import (
    AgentMemory,
    MemoryEntry,
    QueryDecomposition,
    SearchResult,
)
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
    render_documents,
    render_reasoning,
    render_tags,
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

    if "use_llm" not in st.session_state:
        st.session_state["use_llm"] = True

    if "use_graph" not in st.session_state:
        st.session_state["use_graph"] = True

    if "agent_memory" not in st.session_state:
        st.session_state["agent_memory"] = AgentMemory()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "last_filters" not in st.session_state:
        st.session_state["last_filters"] = []

    history_placeholder = st.empty()
    answer_placeholder = st.empty()

    full_answer = ""
    _ = answer_query(
        answer_placeholder,
        history_placeholder,
        "",
        full_answer,
    )

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        _ = st.markdown("## ⚙️ Opcje")

        models = get_available_models("Ollama", "")
        default_model = DEFAULT_OLLAMA_MODEL
        default_idx = next((i for i, m in enumerate(models) if default_model in m), 0)
        selected_model = st.selectbox("Model", models, index=default_idx)

        st.session_state["llm_model"] = selected_model

        # ── Pamięć epizodyczna ───────────────────────────────────────
        memory_placeholder = st.empty()
        memory: AgentMemory = st.session_state["agent_memory"]

        _ = st.markdown("---")
        st.session_state["use_graph"] = st.toggle("Graf powiązań", value=True)

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
        st.session_state["use_llm"] = st.checkbox("🤖 Użyj AI", value=True, key="use_llm_cb")
    with col_btn:
        search_btn = st.button("🔍 Szukaj", type="primary", use_container_width=True)

    # ── Filtry zaawansowane ──────────────────────────────────────
    with st.expander("🔽 Filtry zaawansowane", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            _ = st.markdown('<div class="filter-label">Sygnatura</div>', unsafe_allow_html=True)
            sig_filter = st.text_input(
                "Sygnatura",
                placeholder="np. DKN.5110",
                label_visibility="collapsed",
                key="sig_filter",
            )
            _ = st.markdown('<div class="filter-label">Status</div>', unsafe_allow_html=True)
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
            _ = st.markdown('<div class="filter-label">Sektor</div>', unsafe_allow_html=True)
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
    filters: dict[str, Any] = {"doc_types": doc_types}
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

    # Odświerz listę wątków
    _render_memory(memory_placeholder, history_placeholder, memory)

    # ── Wyszukiwanie ─────────────────────────────────────────────
    effective_query = query
    if sig_filter.strip() and not query.strip():
        effective_query = sig_filter.strip()

    if effective_query and (
        search_btn
        or st.session_state.get("last_query") != effective_query
        or st.session_state.get("last_filters") != filters
    ):
        st.session_state["last_query"] = effective_query
        st.session_state["last_filters"] = filters

        answer_query(
            answer_placeholder,
            history_placeholder,
            kw_filter,
            effective_query,
        )


def analyse_query(
    effective_query: str,
) -> QueryDecomposition:
    decomp: QueryDecomposition | None = None

    with st.spinner("🧠 Analizuję pytanie..."):
        decomp = decompose_query(effective_query)

    if decomp and decomp.reasoning:
        render_reasoning(decomp, effective_query)

    return decomp


def search_docs(
    effective_query: str,
    search_query: str,
    kw_filter: str,
) -> SearchResult:
    use_graph: bool = st.session_state["use_graph"]
    filters: dict[str, Any] = st.session_state["last_filters"]

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
                full_docs = [exact]
                if use_graph:
                    for rsig in exact.get("related_uodo_rulings", [])[:5]:
                        rdoc = fetch_by_signature(rsig)
                        if rdoc:
                            rdoc["_source"] = "graph"
                            rdoc["_score"] = 0.9
                            full_docs.append(rdoc)
            else:
                _ = st.warning(f"Nie znaleziono decyzji o sygnaturze **{sig_norm}** w bazie.")
                full_docs, _tags = hybrid_search(search_query, filters=filters, use_graph=use_graph)
        else:
            full_docs, _tags = hybrid_search(search_query, filters=filters, use_graph=use_graph)
        search_time = time.time() - t0

        res = SearchResult.from_docs(full_docs, _tags, search_time)
        render_tags(res, kw_filter)

    return res


def _render_history(
    placeholder: DeltaGenerator,
    thread: list[MemoryEntry] | None = None,
):
    with placeholder.container(horizontal_alignment="right"):
        if thread:
            for entry in thread:
                with st.container(border=True, width="content"):
                    _ = st.markdown(f"#### Zapytanie: {entry.query}")

                    render_reasoning(entry.decomp, entry.enriched_query)

                    res = entry.search_result
                    render_tags(res)

                with st.container(border=True):
                    _ = st.markdown("## Odpowiedź:")
                    _ = st.markdown("---")
                    _ = st.markdown(entry.full_answer)

                    render_documents(res)
        else:
            pass


def answer_query(
    placeholder: DeltaGenerator,
    history_placeholder: DeltaGenerator,
    kw_filter: str,
    effective_query: str | None = None,
):
    with placeholder.container(horizontal_alignment="right"):
        thread_id: int | None = st.session_state["thread_id"]
        use_llm: bool = st.session_state["use_llm"]
        memory: AgentMemory = st.session_state["agent_memory"]
        filters: dict[str, Any] = st.session_state["last_filters"]

        thread = None
        if thread_id is not None:
            thread = memory.threads[thread_id]

        _render_history(history_placeholder, thread)

        if effective_query and filters:
            decomp: QueryDecomposition | None = None
            with st.container(border=True, width="content"):
                if use_llm and len(effective_query.split()) > 3:
                    _ = st.markdown(f"#### Zapytanie: {effective_query}")
                    decomp = analyse_query(effective_query)

                search_query = decomp.enriched_query if decomp else effective_query
                if decomp and decomp.year_from_hint and "year_from" not in filters:
                    filters["year_from"] = decomp.year_from_hint
                if decomp and decomp.year_to_hint and "year_to" not in filters:
                    filters["year_to"] = decomp.year_to_hint

                res = search_docs(
                    effective_query,
                    search_query,
                    kw_filter,
                )

            if not res.full:
                _ = st.warning(
                    "Nie znaleziono dokumentów. Spróbuj zmienić filtry lub sformułowanie."
                )

            context = build_context(
                res.full, effective_query, thread_id, filters=filters, memory=memory
            )

            try:
                answer = ""
                answer_placeholder = st.empty()

                for chunk in call_llm_stream(effective_query, context):
                    answer += chunk

                    with answer_placeholder.container(border=True):
                        _ = st.markdown("## Odpowiedź:")
                        _ = st.markdown("---")
                        _ = st.markdown(answer)
                        render_documents(res)

                if decomp:
                    entry = MemoryEntry(
                        query=effective_query,
                        enriched_query=search_query,
                        decomp=decomp,
                        search_result=res,
                        full_answer=answer,
                    )

                    print(f"Thread id: {thread_id}")
                    id = memory.add(entry, thread_id)
                    st.session_state["thread_id"] = id

                # FIXME:
                # Po wygenerowaniu odpowiedzi uruchom ponownie
                # by poprawnie odświerzyć listę wątków. Czy
                # da się to rozwiązać inaczej?
                st.rerun()

            except Exception as e:
                _ = st.error(f"Błąd LLM: {e}")


def on_thread_select(
    history_placeholder: DeltaGenerator, thread_id: int, thread: list[MemoryEntry]
):
    """Callback when thread button is clicked."""
    st.session_state["thread_id"] = thread_id
    st.session_state["query_input"] = ""

    if history_placeholder:
        _ = history_placeholder.empty()
        _render_history(history_placeholder, thread)


def on_new_thread(history_placeholder: DeltaGenerator, memory: AgentMemory):
    """Callback when thr button is clicked."""
    id = memory.new_thread()

    st.session_state["thread_id"] = id
    st.session_state["query_input"] = ""

    if history_placeholder:
        _ = history_placeholder.empty()


def _render_memory(
    placeholder: DeltaGenerator,
    history_placeholder: DeltaGenerator | None,
    memory: AgentMemory,
) -> None:
    """Render memory history in the sidebar placeholder."""
    with placeholder.container():
        _ = st.markdown("---")
        _ = st.markdown("### 🧠 Historia sesji")
        for i, thread in enumerate(memory.threads):
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
