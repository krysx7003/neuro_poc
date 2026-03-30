import glob
import json
import os

# Definiujemy listę kluczy, które muszą być w dokumencie (wymagane 100%)
ALLOWED_KEYS = [
    "Tytuł",
    "Data orzeczenia",
    "Sąd",
    "Sędziowie",
    "Sentencja",
    "Symbol z opisem",
    "Data wpływu",
    "Treść wyniku",
    "Uzasadnienie",
    "Skarżony organ",
    "Powołane przepisy",
    "Hasła tematyczne",
]

# Pola, które mają być przekonwertowane na listy (tablice) w formacie JSON
LIST_KEYS = ["Sędziowie", "Skarżony organ", "Powołane przepisy", "Hasła tematyczne"]


def parse_court_document(text):
    lines = text.split("\n")
    metadata = {}
    current_key = None
    current_value = []

    known_text_blocks = ["Sentencja", "Uzasadnienie", "Teza", "Wskazówki"]
    title_captured = False

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped and current_key is None:
            continue

        # 1. Przechwycenie tytułu
        if not title_captured and "|" not in line and line_stripped not in known_text_blocks:
            metadata["Tytuł"] = line_stripped
            title_captured = True
            continue

        # 2. Nagłówki dużych sekcji (np. Uzasadnienie, Sentencja)
        if line_stripped in known_text_blocks:
            if current_key:
                metadata[current_key] = "\n".join(current_value).strip().rstrip("|")
            current_key = line_stripped
            current_value = []
            continue

        # 3. Klucz-Wartość ze znakiem '|'
        if "|" in line and current_key not in known_text_blocks:
            if current_key:
                metadata[current_key] = "\n".join(current_value).strip().rstrip("|")

            parts = line.split("|", 1)
            current_key = parts[0].strip()
            val = parts[1].strip()
            current_value = [val]

            if val.endswith("|"):
                metadata[current_key] = val[:-1].strip()
                current_key = None
                current_value = []
        else:
            if current_key:
                if line_stripped.endswith("|") and current_key not in known_text_blocks:
                    current_value.append(line_stripped[:-1].strip())
                    metadata[current_key] = "\n".join(current_value).strip()
                    current_key = None
                    current_value = []
                else:
                    current_value.append(line_stripped)

    # Zapisz ostatni otwarty blok
    if current_key:
        metadata[current_key] = "\n".join(current_value).strip().rstrip("|")

    # --- FILTROWANIE I FORMATOWANIE SPECJALNE ---

    # Krok 1: Zostawiamy tylko dozwolone klucze
    filtered_metadata = {k: v for k, v in metadata.items() if k in ALLOWED_KEYS}

    # Krok 2: Tworzenie list dla wybranych kluczy
    for key in LIST_KEYS:
        if key in filtered_metadata:
            raw_text = filtered_metadata[key]
            # Dzielimy po nowej linii i usuwamy puste elementy
            filtered_metadata[key] = [item.strip() for item in raw_text.split("\n") if item.strip()]

    return filtered_metadata


def process_folder(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    search_pattern = os.path.join(input_folder, "*.txt")
    txt_files = glob.glob(search_pattern)
    total_files = len(txt_files)

    if total_files == 0:
        print(f"Brak plików .txt w folderze: '{input_folder}'")
        return

    print(f"Znaleziono {total_files} plików. Rozpoczynam rygorystyczne przetwarzanie...\n")

    saved_count = 0
    skipped_count = 0

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        output_filepath = os.path.join(output_folder, f"{name_without_ext}.json")

        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            parsed_data = parse_court_document(content)

            # --- SPRAWDZANIE KOMPLETNOŚCI METADANYCH ---
            # Tworzymy listę kluczy, których brakuje w przeparsowanych danych
            missing_keys = [key for key in ALLOWED_KEYS if key not in parsed_data]

            # Jeśli lista brakujących kluczy nie jest pusta, pomijamy plik
            if missing_keys:
                print(f"[POMINIĘTO] {filename} -> Brakuje: {', '.join(missing_keys)}")
                skipped_count += 1
                continue

            # Jeśli plik ma wszystkie klucze, zapisujemy go
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=4)

            print(f"[OK] Zapisano kompletny plik: {name_without_ext}.json")
            saved_count += 1

        except Exception as e:
            print(f"[BŁĄD] Problem przy pliku {filename}: {e}")
            skipped_count += 1

    # --- PODSUMOWANIE ---
    print("\n" + "=" * 50)
    print(" PODSUMOWANIE ZAPISU (STRICT MODE)")
    print("=" * 50)
    print(f"Wszystkich plików: {total_files}")
    print(f"✅ Zapisano pomyślnie (kompletne): {saved_count}")
    print(f"❌ Pominięto (brakujące dane lub błąd): {skipped_count}")
    print("=" * 50)


# --- KONFIGURACJA I URUCHOMIENIE ---
if __name__ == "__main__":
    folder_z_dokumentami = "C:/Users/wrobl/Desktop/LEXSEARCH/NEUROPOC/daneUODO/NSARefs"
    folder_na_jsony = "C:/Users/wrobl/Desktop/LEXSEARCH/NEUROPOC/daneUODO/NSAMetadata"

    process_folder(folder_z_dokumentami, folder_na_jsony)

