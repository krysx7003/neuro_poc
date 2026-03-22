import os
import sys
import glob
from pathlib import Path
from striprtf.striprtf import rtf_to_text
import chardet

def convert_single_rtf(rtf_file, output_dir='.'):
    """
    Konwertuje pojedynczy plik RTF na TXT.
    """
    if not os.path.exists(rtf_file):
        print(f"Błąd: {rtf_file} nie istnieje.")
        return False
    
    try:
        # Odczytaj plik z automatycznym wykrywaniem kodowania
        with open(rtf_file, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        # Konwertuj RTF na tekst
        with open(rtf_file, 'r', encoding=encoding, errors='ignore') as f:
            rtf_content = f.read()
            plain_text = rtf_to_text(rtf_content)
        
        # Nazwa pliku wyjściowego
        base_name = Path(rtf_file).stem
        txt_file = os.path.join(output_dir, f"{base_name}.txt")
        
        # Zapisz TXT
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(plain_text)
        
        print(f"{rtf_file} → {txt_file}")
        return True
        
    except Exception as e:
        print(f"Błąd konwersji {rtf_file}: {e}")
        return False

def convert_rtf_folder(folder_path, output_dir='.'):
    """
    Konwertuje wszystkie pliki .rtf z folderu.
    """
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder {folder_path} nie istnieje.")
        return False
    
    # Znajdź wszystkie pliki .rtf
    pattern = os.path.join(folder_path, "*.rtf")
    rtf_files = glob.glob(pattern, recursive=False)
    
    if not rtf_files:
        print(f"Brak plików RTF w folderze: {folder_path}")
        return False
    
    print(f"Znaleziono {len(rtf_files)} plików RTF")
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    
    for rtf_file in sorted(rtf_files):
        if convert_single_rtf(rtf_file, output_dir):
            success_count += 1
    
    print(f"\nPomyślnie skonwertowano {success_count}/{len(rtf_files)} plików")
    return success_count > 0

def main():
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python rtf2txt.py sciezka_do_pliku.rtf [katalog_wyjściowy]")
        print("  python rtf2txt.py sciezka_do_folderu/ [katalog_wyjściowy]")
        sys.exit(1)
    
    target_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    # Automatyczne wykrywanie: plik czy folder?
    if os.path.isfile(target_path) and target_path.lower().endswith('.rtf'):
        convert_single_rtf(target_path, output_dir)
    elif os.path.isdir(target_path):
        convert_rtf_folder(target_path, output_dir)
    else:
        print(f"Błąd: {target_path} nie jest plikiem .rtf ani folderem")
        sys.exit(1)

if __name__ == "__main__":
    main()
