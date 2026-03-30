import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_unique_filename(base_name):
    """Generuje unikalną nazwę pliku na podstawie pełnej ścieżki."""
    # Użyj pełnej ścieżki względnej jako unikalny identyfikator
    unique_id = Path(base_name).parent.name  # Nazwa folderu źródłowego
    safe_id = "".join(c for c in unique_id if c.isalnum() or c in (" ", "-", "_")).rstrip()
    return f"{safe_id}_{Path(base_name).stem}.txt"


def xml_to_text(xml_content):
    """Wyciąga czysty tekst z zawartości XML (usuwa tagi)."""
    try:
        root = ET.fromstring(xml_content)
        return ET.tostring(root, encoding="unicode", method="text").strip()
    except ET.ParseError:
        text = re.sub(r"<[^>]*>", "", xml_content)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"<!\[CDATA\[.*?\]\]>", "", text, flags=re.DOTALL)
        return re.sub(r"\s+", " ", text).strip()


def convert_single_xml(xml_file, output_dir="."):
    """Konwertuje pojedynczy plik XML."""
    if not os.path.exists(xml_file):
        print(f"Błąd: {xml_file} nie istnieje.")
        return False

    try:
        with open(xml_file, encoding="utf-8", errors="ignore") as f:
            xml_content = f.read()

        plain_text = xml_to_text(xml_content)

        if not plain_text:
            print(f"{xml_file}: brak treści tekstowej")
            return False

        # **KLUCZOWA ZMIANA**: Unikalna nazwa z folderem źródłowym
        txt_filename = get_unique_filename(xml_file)
        txt_file = os.path.join(output_dir, txt_filename)

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(plain_text)

        print(f"{xml_file} → {txt_file} ({len(plain_text)} znaków)")
        return True

    except Exception as e:
        print(f"Błąd {xml_file}: {e}")
        return False


def convert_xml_folder(folder_path, output_dir="."):
    """Konwertuje wszystkie .xml z folderu i PODFOLDERÓW."""
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder {folder_path} nie istnieje.")
        return False

    pattern = os.path.join(folder_path, "**", "*.xml")
    xml_files = glob.glob(pattern, recursive=True)

    if not xml_files:
        print(f"Brak plików XML w: {folder_path} (w tym podfolderach)")
        return False

    print(f"Znaleziono {len(xml_files)} plików XML")

    os.makedirs(output_dir, exist_ok=True)
    success_count = 0

    for xml_file in sorted(xml_files):
        if convert_single_xml(xml_file, output_dir):
            success_count += 1

    print(f"\nPomyślnie skonwertowano {success_count}/{len(xml_files)} plików")
    return success_count > 0


def main():
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python xml2txt.py plik.xml [katalog_wyjściowy]")
        print("  python xml2txt.py folder/ [katalog_wyjściowy]")
        sys.exit(1)

    target_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if os.path.isfile(target_path) and target_path.lower().endswith(".xml"):
        convert_single_xml(target_path, output_dir)
    elif os.path.isdir(target_path):
        convert_xml_folder(target_path, output_dir)
    else:
        print(f"Błąd: {target_path} nie jest plikiem .xml ani folderem")
        sys.exit(1)


if __name__ == "__main__":
    main()
