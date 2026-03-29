"""ragas_eval.py — Pełna ewaluacja RAG (10 pytań) przy użyciu biblioteki Ragas.
Wymaga: pip install ragas datasets langchain langchain-community langchain-huggingface
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# Importujemy funkcje z Twojego działającego pliku eval.py
from eval import semantic_search, call_llm

# ====================================================================
# 1. KONFIGURACJA SĘDZIEGO (LLM-as-a-Judge) ORAZ EMBEDDINGS
# ====================================================================
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_URL = os.getenv("ollama_url", os.getenv("OLLAMA_URL", "http://localhost:11434"))
# Sędzią będzie główny model. Wpisz odpowiednią nazwę:
JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "mistral-large-3:675b-cloud") 

judge_llm = ChatOllama(
    base_url=OLLAMA_URL,
    model=JUDGE_MODEL,
    temperature=0.0
)

# Model embeddingów
judge_embeddings = HuggingFaceEmbeddings(
    model_name="sdadas/mmlw-retrieval-roberta-large"
)

# ====================================================================
# 2. DEFINICJA ZBIORU TESTOWEGO (10 Złotych Pytań + Ground Truth)
# ====================================================================
TEST_DATA = [
    {
        "id": "GQ-001",
        "question": "Jakie kary może nałożyć Prezes UODO?",
        "ground_truth": "Prezes UODO może nałożyć administracyjne kary pieniężne. Ich wysokość zależy od rodzaju naruszenia i wynosi, zgodnie z art. 83 RODO, do 10 000 000 EUR lub 2% całkowitego rocznego obrotu, albo do 20 000 000 EUR lub 4% obrotu. Zgodnie z polską u.o.d.o. (art. 102) na jednostki sektora finansów publicznych kara wynosi do 100 000 zł lub 10 000 zł."
    },
    {
        "id": "GQ-002",
        "question": "Kiedy wymagane jest zgłoszenie naruszenia danych do UODO?",
        "ground_truth": "Zgodnie z art. 33 RODO, administrator ma obowiązek zgłosić naruszenie ochrony danych osobowych do UODO bez zbędnej zwłoki – w miarę możliwości nie później niż w terminie 72 godzin po stwierdzeniu naruszenia – chyba że jest mało prawdopodobne, by skutkowało ono ryzykiem naruszenia praw lub wolności osób fizycznych."
    },
    {
        "id": "GQ-003",
        "question": "Co to jest podstawa prawna przetwarzania danych osobowych?",
        "ground_truth": "Podstawa prawna to warunek legalności przetwarzania wskazany w art. 6 RODO. Przetwarzanie jest zgodne z prawem m.in. wtedy, gdy: osoba wyraziła zgodę, jest to niezbędne do wykonania umowy, wypełnienia obowiązku prawnego ciążącego na administratorze, ochrony żywotnych interesów, wykonania zadania w interesie publicznym lub do celów wynikających z prawnie uzasadnionych interesów."
    },
    {
        "id": "GQ-004",
        "question": "Jakie obowiązki ma administrator danych wobec osoby, której dane dotyczą?",
        "ground_truth": "Administrator ma przede wszystkim obowiązek informacyjny (art. 13 i 14 RODO), czyli podania m.in. swojej tożsamości, celów i podstaw prawnych przetwarzania oraz czasu przechowywania danych. Ponadto musi ułatwiać osobie, której dane dotyczą, wykonywanie jej praw (np. prawa dostępu, sprostowania, usunięcia)."
    },
    {
        "id": "GQ-005",
        "question": "Czym jest inspektor ochrony danych i kiedy trzeba go wyznaczyć?",
        "ground_truth": "Inspektor Ochrony Danych (IOD) to osoba wspierająca administratora w przestrzeganiu przepisów RODO. Zgodnie z art. 37 RODO jego wyznaczenie jest obowiązkowe, gdy przetwarzania dokonuje organ lub podmiot publiczny, główna działalność polega na monitorowaniu osób na dużą skalę lub na przetwarzaniu szczególnych kategorii danych (wrażliwych) na dużą skalę."
    },
    {
        "id": "GQ-006",
        "question": "Jakie prawa przysługują osobie, której dane są przetwarzane?",
        "ground_truth": "Osobie tej przysługuje prawo: dostępu do danych, sprostowania błędnych danych, usunięcia danych (prawo do bycia zapomnianym), ograniczenia przetwarzania, przenoszenia danych do innego administratora oraz prawo sprzeciwu wobec przetwarzania."
    },
    {
        "id": "GQ-007",
        "question": "Co to jest umowa powierzenia przetwarzania danych?",
        "ground_truth": "Zgodnie z art. 28 RODO, jest to umowa (lub inny instrument prawny), na mocy której administrator danych powierza przetwarzanie danych osobowych podmiotowi przetwarzającemu (procesorowi) w jego imieniu. Umowa ta musi określać m.in. przedmiot, czas trwania, charakter i cel przetwarzania oraz obowiązki i prawa stron."
    },
    {
        "id": "GQ-008",
        "question": "Jakie dane uznaje się za szczególne kategorie danych osobowych?",
        "ground_truth": "Zgodnie z art. 9 RODO (tzw. dane wrażliwe), są to dane ujawniające pochodzenie rasowe lub etniczne, poglądy polityczne, przekonania religijne lub światopoglądowe, przynależność do związków zawodowych oraz dane genetyczne, biometryczne, dane dotyczące zdrowia, seksualności lub orientacji seksualnej."
    },
    {
        "id": "GQ-009",
        "question": "Kiedy można przekazywać dane osobowe do krajów trzecich?",
        "ground_truth": "Zgodnie z rozdziałem V RODO, dane można przekazać do państwa trzeciego (poza EOG), jeśli Komisja Europejska wydała decyzję stwierdzającą odpowiedni stopień ochrony. W braku takiej decyzji, transfer jest możliwy pod warunkiem zapewnienia odpowiednich zabezpieczeń (np. standardowe klauzule umowne, wiążące reguły korporacyjne - BCR) lub na podstawie wyjątków (np. wyraźna zgoda, realizacja umowy)."
    },
    {
        "id": "GQ-010",
        "question": "Co to jest minimalizacja danych i zasada ograniczenia celu?",
        "ground_truth": "Zgodnie z art. 5 RODO, zasada minimalizacji danych oznacza, że zbierane dane muszą być adekwatne, stosowne i ograniczone tylko do tego, co niezbędne do celów ich przetwarzania. Zasada ograniczenia celu mówi, że dane muszą być zbierane w konkretnych, wyraźnych celach i nie wolno ich dalej przetwarzać w sposób z tymi celami niezgodny."
    }
]

def main():
    print(f"\nRozpoczynam zbieranie wyników dla {len(TEST_DATA)} pytań...")
    
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # ====================================================================
    # 3. WYKONANIE PIPELINE'U RAG (Pobranie kontekstu i generacja odpowiedzi)
    # ====================================================================
    for idx, item in enumerate(TEST_DATA, 1):
        question = item["question"]
        print(f" [{idx}/{len(TEST_DATA)}] Odpytuję: {item['id']}...")
        
        # 1. Wyszukaj dokumenty w Qdrant
        try:
            docs = semantic_search(question, top_k=5)
            contexts = [doc.get("content_text", "") for doc in docs]
            joined_context = "\n\n".join(contexts) 
            
            # 2. Wygeneruj odpowiedź Twoim LLM
            answer = call_llm(question, joined_context)
            
            # 3. Zapisz do słownika
            data_for_ragas["question"].append(question)
            data_for_ragas["answer"].append(answer)
            data_for_ragas["contexts"].append(contexts)
            data_for_ragas["ground_truth"].append(item["ground_truth"])
        except Exception as e:
            print(f"    ❌ BŁĄD przy pytaniu {item['id']}: {e}")

    dataset = Dataset.from_dict(data_for_ragas)

    print("\nDane zebrane. Uruchamiam Sędziego (Ragas) do oceny metryk...")
    print("To może potrwać kilka minut. Trwa analiza wektorów i odpowiedzi...")

    # ====================================================================
    # 4. EWALUACJA METRYK RAGAS
    # ====================================================================
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        
        run_config=RunConfig(
            max_workers=1,       
            timeout=180,         
            max_retries=5        
        )
    )

    print("\n" + "="*50)
    print(" 🏆 ŚREDNIE WYNIKI EWALUACJI RAGAS (0.0 - 1.0):")
    print("="*50)
    
    # Wyświetlamy uśrednione wyniki dla całego systemu
    print(result)
    
    # Zapisz szczegółowe wyniki do pliku CSV
    df_results = result.to_pandas()
    output_file = "ragas_full_evaluation.csv"
    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Gotowe! Szczegółowe wyniki dla każdego pytania zapisano w: {output_file}")

if __name__ == "__main__":
    main()