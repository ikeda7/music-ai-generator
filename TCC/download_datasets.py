"""
Script para download automático de datasets MIDI para treinamento.
Baixa múltiplos datasets populares para garantir treinamento extensivo e de qualidade.
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
import time

# Tenta importar tqdm, mas funciona sem ele também
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm se não estiver disponível
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args):
            pass


class DownloadProgressBar:
    """Barra de progresso personalizada para downloads."""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if HAS_TQDM:
            if not self.pbar:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()
        else:
            # Fallback simples sem tqdm
            if total_size > 0:
                downloaded = block_num * block_size
                percent = (downloaded / total_size) * 100
                if block_num % 100 == 0:  # Atualiza a cada 100 blocos
                    print(f"\rProgresso: {percent:.1f}%", end='', flush=True)


def download_file(url: str, dest_path: str, description: str = ""):
    """
    Baixa um arquivo com barra de progresso.
    
    Args:
        url: URL do arquivo
        dest_path: Caminho de destino
        description: Descrição para exibição
    """
    print(f"\nBaixando {description}...")
    print(f"URL: {url}")
    print(f"Destino: {dest_path}")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, DownloadProgressBar())
        print(f"[OK] Download concluido: {dest_path}")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao baixar: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """Extrai arquivo ZIP."""
    print(f"\nExtraindo {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[OK] Extracao concluida em: {extract_to}")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao extrair: {e}")
        return False


def extract_tar(tar_path: str, extract_to: str):
    """Extrai arquivo TAR/TAR.GZ."""
    print(f"\nExtraindo {tar_path}...")
    try:
        if tar_path.endswith('.gz'):
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"[OK] Extracao concluida em: {extract_to}")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao extrair: {e}")
        return False


def download_maestro(datasets_dir: str):
    """
    Baixa o dataset MAESTRO (piano performances).
    Link: https://magenta.tensorflow.org/datasets/maestro
    """
    print("\n" + "="*60)
    print("DATASET: MAESTRO")
    print("="*60)
    print("Descrição: Mais de 200 horas de performances de piano com MIDI")
    
    maestro_dir = os.path.join(datasets_dir, "maestro")
    os.makedirs(maestro_dir, exist_ok=True)
    
    # MAESTRO v3.0.0 - versão mais recente
    # Nota: O download direto pode não funcionar, então fornecemos instruções
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    zip_path = os.path.join(maestro_dir, "maestro-v3.0.0-midi.zip")
    
    if download_file(url, zip_path, "MAESTRO v3.0.0"):
        extract_zip(zip_path, maestro_dir)
        # Remove arquivo ZIP após extração (com delay para Windows)
        if os.path.exists(zip_path):
            try:
                time.sleep(0.5)  # Pequeno delay para Windows
                os.remove(zip_path)
            except PermissionError:
                print(f"[AVISO] Nao foi possivel remover {zip_path} (arquivo em uso). Voce pode remove-lo manualmente.")
        return True
    else:
        print("\n[AVISO] Download automatico do MAESTRO falhou.")
        print("   Por favor, baixe manualmente de: https://magenta.tensorflow.org/datasets/maestro")
        print(f"   E extraia os arquivos MIDI em: {maestro_dir}")
        return False


def download_groove(datasets_dir: str):
    """
    Baixa o dataset Groove (drum patterns).
    Link: https://magenta.tensorflow.org/datasets/groove
    """
    print("\n" + "="*60)
    print("DATASET: Groove")
    print("="*60)
    print("Descrição: Padrões de bateria MIDI de alta qualidade")
    
    groove_dir = os.path.join(datasets_dir, "groove")
    os.makedirs(groove_dir, exist_ok=True)
    
    # Groove MIDI dataset
    url = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
    zip_path = os.path.join(groove_dir, "groove-midi.zip")
    
    if download_file(url, zip_path, "Groove MIDI"):
        extract_zip(zip_path, groove_dir)
        if os.path.exists(zip_path):
            try:
                time.sleep(0.5)
                os.remove(zip_path)
            except PermissionError:
                print(f"[AVISO] Nao foi possivel remover {zip_path} (arquivo em uso). Voce pode remove-lo manualmente.")
        return True
    else:
        print("\n[AVISO] Download automatico do Groove falhou.")
        print("   Por favor, baixe manualmente de: https://magenta.tensorflow.org/datasets/groove")
        print(f"   E extraia os arquivos MIDI em: {groove_dir}")
        return False


def download_pop909(datasets_dir: str):
    """
    Baixa o dataset POP909 (música pop chinesa com múltiplos tracks).
    Link: https://github.com/music-x-lab/POP909-Dataset
    """
    print("\n" + "="*60)
    print("DATASET: POP909")
    print("="*60)
    print("Descrição: 909 músicas pop com melodia, piano, baixo e outros tracks")
    
    pop909_dir = os.path.join(datasets_dir, "pop909")
    os.makedirs(pop909_dir, exist_ok=True)
    
    # POP909 via GitHub releases
    url = "https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(pop909_dir, "pop909-master.zip")
    
    if download_file(url, zip_path, "POP909"):
        extract_zip(zip_path, pop909_dir)
        # Move arquivos MIDI para o diretório principal
        extracted_dir = os.path.join(pop909_dir, "POP909-Dataset-master")
        if os.path.exists(extracted_dir):
            # Procura por arquivos MIDI
            for root, dirs, files in os.walk(extracted_dir):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        src = os.path.join(root, file)
                        dst = os.path.join(pop909_dir, file)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
            # Remove diretório extraído
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
        if os.path.exists(zip_path):
            try:
                time.sleep(0.5)
                os.remove(zip_path)
            except PermissionError:
                print(f"[AVISO] Nao foi possivel remover {zip_path} (arquivo em uso). Voce pode remove-lo manualmente.")
        return True
    else:
        print("\n[AVISO] Download automatico do POP909 falhou.")
        print("   Por favor, baixe manualmente de: https://github.com/music-x-lab/POP909-Dataset")
        print(f"   E extraia os arquivos MIDI em: {pop909_dir}")
        return False


def download_lakh_midi_sample(datasets_dir: str):
    """
    Baixa uma amostra do Lakh MIDI Dataset (dataset muito grande).
    Nota: O dataset completo tem ~170k arquivos, então baixamos uma amostra.
    """
    print("\n" + "="*60)
    print("DATASET: Lakh MIDI (Sample)")
    print("="*60)
    print("Descrição: Amostra do Lakh MIDI Dataset (~170k músicas)")
    print("[AVISO] NOTA: O dataset completo e muito grande. Baixando apenas links de referencia.")
    
    lakh_dir = os.path.join(datasets_dir, "lakh_midi")
    os.makedirs(lakh_dir, exist_ok=True)
    
    # Cria arquivo com instruções de download
    instructions = """Lakh MIDI Dataset - Instruções de Download

O Lakh MIDI Dataset completo contém aproximadamente 170.000 arquivos MIDI.
Devido ao tamanho, não podemos baixá-lo automaticamente.

Para baixar:
1. Acesse: https://colinraffel.com/projects/lmd/
2. Baixe o arquivo "lmd_full.tar.gz" (requer ~50GB de espaço)
3. Extraia em: {lakh_dir}

Ou use apenas a amostra do MAESTRO e outros datasets menores para começar.
""".format(lakh_dir=lakh_dir)
    
    readme_path = os.path.join(lakh_dir, "README_DOWNLOAD.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"[OK] Instrucoes salvas em: {readme_path}")
    return False  # Não baixamos automaticamente


def download_classical_piano(datasets_dir: str):
    """
    Baixa uma coleção de músicas clássicas de piano (fonte alternativa).
    """
    print("\n" + "="*60)
    print("DATASET: Classical Piano (KernScores)")
    print("="*60)
    print("Descrição: Músicas clássicas de piano em formato MIDI")
    
    classical_dir = os.path.join(datasets_dir, "classical_piano")
    os.makedirs(classical_dir, exist_ok=True)
    
    # KernScores - coleção de música clássica
    # Nota: Este é um exemplo, pode precisar de ajustes
    print("[AVISO] Este dataset requer download manual.")
    print("   Sugestões:")
    print("   - https://kern.humdrum.org/")
    print("   - https://www.piano-midi.de/")
    print(f"   - Salve arquivos MIDI em: {classical_dir}")
    
    return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Baixa datasets MIDI para treinamento')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                       help='Diretório para salvar os datasets (padrão: ./datasets)')
    parser.add_argument('--maestro', action='store_true',
                       help='Baixar apenas MAESTRO')
    parser.add_argument('--groove', action='store_true',
                       help='Baixar apenas Groove')
    parser.add_argument('--pop909', action='store_true',
                       help='Baixar apenas POP909')
    parser.add_argument('--all', action='store_true',
                       help='Baixar todos os datasets disponíveis')
    
    args = parser.parse_args()
    
    # Se nenhum argumento específico, baixa todos
    if not any([args.maestro, args.groove, args.pop909]):
        args.all = True
    
    # Cria diretório de datasets
    datasets_dir = os.path.abspath(args.datasets_dir)
    os.makedirs(datasets_dir, exist_ok=True)
    
    print("="*60)
    print("DOWNLOAD DE DATASETS MIDI")
    print("="*60)
    print(f"Diretório de destino: {datasets_dir}")
    print(f"Espaço disponível necessário: ~5-10 GB (para datasets principais)")
    print("\nIniciando downloads...")
    
    results = {}
    
    # Download de datasets
    if args.all or args.maestro:
        results['maestro'] = download_maestro(datasets_dir)
    
    if args.all or args.groove:
        results['groove'] = download_groove(datasets_dir)
    
    if args.all or args.pop909:
        results['pop909'] = download_pop909(datasets_dir)
    
    # Informações sobre outros datasets
    if args.all:
        download_lakh_midi_sample(datasets_dir)
        download_classical_piano(datasets_dir)
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DO DOWNLOAD")
    print("="*60)
    
    for dataset, success in results.items():
        status = "[OK] SUCESSO" if success else "[X] FALHOU / MANUAL"
        print(f"{dataset.upper()}: {status}")
    
    print(f"\nDatasets salvos em: {datasets_dir}")
    print("\nPara treinar com todos os datasets:")
    print(f"  python train.py --data_path {datasets_dir}")
    
    print("\nPara treinar com um dataset específico:")
    print(f"  python train.py --data_path {datasets_dir}/maestro")
    print(f"  python train.py --data_path {datasets_dir}/groove")
    print(f"  python train.py --data_path {datasets_dir}/pop909")

