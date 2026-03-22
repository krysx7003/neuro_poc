import tarfile
import os
import sys
import glob
from pathlib import Path

def extract_tar(plik_tar, katalog_wyjściowy='.'):
    """
    Wypakowuje plik .tar do podanego katalogu.
    
    Args:
        plik_tar (str): Ścieżka do pliku .tar(.gz, .bz2 itp.)
        katalog_wyjściowy (str): Katalog docelowy (domyślnie bieżący)
    Returns:
        bool: True jeśli pomyślnie wypakowano
    """
    if not os.path.exists(plik_tar):
        print(f"Błąd: Plik {plik_tar} nie istnieje.")
        return False
    
    try:
        os.makedirs(katalog_wyjściowy, exist_ok=True)
        with tarfile.open(plik_tar, 'r:*') as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=katalog_wyjściowy, filter='data')
            else:
                tar.extractall(path=katalog_wyjściowy)
        print(f"Wypakowano {plik_tar} do: {katalog_wyjściowy}")
        return True
    except Exception as e:
        print(f"Błąd {plik_tar}: {e}")
        return False

def extract_folder(folder_path, katalog_wyjściowy='.'):
    """
    Wypakowuje wszystkie pliki .tar* z folderu.
    """
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder {folder_path} nie istnieje.")
        return False
    
    # Znajdź wszystkie pliki .tar, .tar.gz, .tar.bz2 itp.
    pattern = os.path.join(folder_path, "*.tar*")
    tar_files = glob.glob(pattern)
    
    if not tar_files:
        print(f"Brak plików .tar w folderze: {folder_path}")
        return False
    
    print(f"Znaleziono {len(tar_files)} archiwów w {folder_path}")
    
    success_count = 0
    for tar_file in sorted(tar_files):
        # Nazwa podfolderu = nazwa archiwum bez rozszerzenia
        base_name = Path(tar_file).stem
        target_dir = os.path.join(katalog_wyjściowy, base_name)
        if extract_tar(tar_file, target_dir):
            success_count += 1
    
    print(f"\nPomyślnie wypakowano {success_count}/{len(tar_files)} archiwów")
    return success_count > 0

def main():
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python skrypt.py sciezka_do_pliku.tar [katalog_wyjściowy]")
        print("  python skrypt.py sciezka_do_folderu/ [katalog_wyjściowy]")
        sys.exit(1)
    
    target_path = sys.argv[1]
    katalog_wyjściowy = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    # Sprawdź czy to plik czy folder
    if os.path.isfile(target_path) and target_path.lower().endswith('.tar'):
        extract_tar(target_path, katalog_wyjściowy)
    elif os.path.isdir(target_path):
        extract_folder(target_path, katalog_wyjściowy)
    else:
        print(f"Błąd: {target_path} nie jest plikiem .tar ani folderem")
        sys.exit(1)

if __name__ == "__main__":
    main()
