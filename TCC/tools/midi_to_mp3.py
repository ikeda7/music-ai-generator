"""
Conversor MIDI -> MP3 em batch.

Detecta automaticamente o melhor backend disponível na máquina:
1. MuseScore CLI       — alta qualidade, soundfont embutido (recomendado)
2. fluidsynth + ffmpeg — qualidade alta, requer .sf2 separado

Uso:
    # Arquivo único
    python tools/midi_to_mp3.py samples/preview/preview_C_pop.mid

    # Pasta inteira (batch)
    python tools/midi_to_mp3.py samples/preview/

    # Soundfont customizado (só fluidsynth)
    python tools/midi_to_mp3.py samples/ --soundfont C:/soundfonts/FluidR3.sf2

    # Bitrate diferente (default 192k)
    python tools/midi_to_mp3.py samples/preview/ --bitrate 256

    # Forçar backend específico
    python tools/midi_to_mp3.py samples/ --backend musescore
    python tools/midi_to_mp3.py samples/ --backend fluidsynth

Setup (escolha um):

    Opção A — MuseScore (mais simples):
        Baixe e instale: https://musescore.org/pt-br/download
        Script detecta automaticamente no Windows.

    Opção B — fluidsynth + ffmpeg (Linux/Mac/Windows):
        Windows:  choco install fluidsynth ffmpeg
        Linux:    sudo apt install fluidsynth ffmpeg
        Mac:      brew install fluidsynth ffmpeg

        Baixe um SoundFont GM (~140 MB) — recomendado FluidR3_GM:
        https://member.keymusician.com/Member/FluidR3_GM/index.html

        Aponte o caminho via --soundfont ou variável SOUNDFONT_PATH.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


MUSESCORE_PATHS = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]


def find_musescore():
    """Detecta MuseScore no PATH ou em caminhos comuns por SO."""
    for name in ('mscore', 'musescore', 'MuseScore4', 'MuseScore3'):
        path = shutil.which(name)
        if path:
            return path
    for path in MUSESCORE_PATHS:
        if os.path.isfile(path):
            return path
    return None


def find_fluidsynth():
    return shutil.which('fluidsynth')


def find_ffmpeg():
    return shutil.which('ffmpeg')


def find_soundfont(custom=None):
    """Localiza .sf2 via arg -> env -> caminhos comuns."""
    if custom and os.path.isfile(custom):
        return custom
    env_path = os.environ.get('SOUNDFONT_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path
    candidates = [
        r"C:\Soundfonts\FluidR3_GM.sf2",
        r"C:\Program Files\FluidR3_GM\FluidR3_GM.sf2",
        os.path.expanduser("~/soundfonts/FluidR3_GM.sf2"),
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def convert_via_musescore(musescore, midi_path, mp3_path):
    """MuseScore CLI converte direto MIDI -> MP3 usando seu soundfont interno."""
    cmd = [musescore, '-o', mp3_path, midi_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"    stderr: {r.stderr[-300:]}")
    return r.returncode == 0 and os.path.isfile(mp3_path) and os.path.getsize(mp3_path) > 0


def convert_via_fluidsynth(fluidsynth, ffmpeg, soundfont, midi_path, mp3_path, bitrate):
    """fluidsynth renderiza WAV; ffmpeg codifica MP3 com o bitrate desejado."""
    wav_path = mp3_path[:-4] + '.wav'
    fs_cmd = [fluidsynth, '-ni', '-g', '1.5', soundfont, midi_path,
              '-F', wav_path, '-r', '44100']
    r1 = subprocess.run(fs_cmd, capture_output=True, text=True, timeout=120)
    if r1.returncode != 0 or not os.path.isfile(wav_path):
        print(f"    fluidsynth stderr: {r1.stderr[-300:]}")
        return False
    ff_cmd = [ffmpeg, '-y', '-loglevel', 'error', '-i', wav_path,
              '-b:a', f'{bitrate}k', mp3_path]
    r2 = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=120)
    try:
        os.remove(wav_path)
    except OSError:
        pass
    if r2.returncode != 0:
        print(f"    ffmpeg stderr: {r2.stderr[-300:]}")
    return r2.returncode == 0 and os.path.isfile(mp3_path)


def collect_files(input_path):
    """Retorna lista de .mid a converter (arquivo único ou pasta)."""
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(('.mid', '.midi')) else []
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, '*.mid'))
                       + glob.glob(os.path.join(input_path, '*.midi')))
        return files
    return []


def main():
    parser = argparse.ArgumentParser(description='MIDI -> MP3 batch converter')
    parser.add_argument('input', help='Arquivo .mid ou diretório com .mids')
    parser.add_argument('--backend', choices=['auto', 'musescore', 'fluidsynth'],
                        default='auto', help='Backend de conversão (default auto)')
    parser.add_argument('--soundfont', default=None,
                        help='Caminho do .sf2 (só fluidsynth). Default: detecta automaticamente')
    parser.add_argument('--bitrate', type=int, default=192,
                        help='Bitrate MP3 em kbps (só fluidsynth, default 192)')
    parser.add_argument('--output_dir', default=None,
                        help='Pasta de saída (default: mesma do .mid)')
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        print(f"Nenhum arquivo .mid encontrado em '{args.input}'")
        return 1

    # Seleciona backend
    musescore = find_musescore() if args.backend in ('auto', 'musescore') else None
    fluidsynth = find_fluidsynth() if args.backend in ('auto', 'fluidsynth') else None
    ffmpeg = find_ffmpeg() if args.backend in ('auto', 'fluidsynth') else None
    soundfont = find_soundfont(args.soundfont) if args.backend in ('auto', 'fluidsynth') else None

    backend = None
    if args.backend == 'musescore':
        if not musescore:
            print("ERRO: MuseScore não encontrado. Instale ou use --backend fluidsynth")
            return 1
        backend = 'musescore'
    elif args.backend == 'fluidsynth':
        if not (fluidsynth and ffmpeg and soundfont):
            print("ERRO: fluidsynth/ffmpeg/soundfont incompleto:")
            print(f"  fluidsynth: {fluidsynth or 'NÃO ENCONTRADO'}")
            print(f"  ffmpeg:     {ffmpeg or 'NÃO ENCONTRADO'}")
            print(f"  soundfont:  {soundfont or 'NÃO ENCONTRADO (use --soundfont)'}")
            return 1
        backend = 'fluidsynth'
    else:
        # auto: prioriza MuseScore (simpler), fallback fluidsynth
        if musescore:
            backend = 'musescore'
        elif fluidsynth and ffmpeg and soundfont:
            backend = 'fluidsynth'
        else:
            print("ERRO: nenhum backend disponível.")
            print("  Instale MuseScore (https://musescore.org/download) OU")
            print("  fluidsynth + ffmpeg + um soundfont .sf2")
            return 1

    print(f"Backend: {backend}")
    if backend == 'musescore':
        print(f"  MuseScore: {musescore}")
    else:
        print(f"  fluidsynth: {fluidsynth}")
        print(f"  ffmpeg:     {ffmpeg}")
        print(f"  soundfont:  {soundfont}")
        print(f"  bitrate:    {args.bitrate}k")

    print(f"\nConvertendo {len(files)} arquivo(s)...\n")
    ok = 0
    for midi_path in files:
        out_dir = args.output_dir or os.path.dirname(midi_path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        mp3_name = os.path.splitext(os.path.basename(midi_path))[0] + '.mp3'
        mp3_path = os.path.join(out_dir, mp3_name)

        print(f"  {os.path.basename(midi_path)} -> {mp3_name}")
        if backend == 'musescore':
            success = convert_via_musescore(musescore, midi_path, mp3_path)
        else:
            success = convert_via_fluidsynth(fluidsynth, ffmpeg, soundfont,
                                              midi_path, mp3_path, args.bitrate)
        if success:
            ok += 1
            size_kb = os.path.getsize(mp3_path) // 1024
            print(f"    [OK] {size_kb} KB")
        else:
            print(f"    [X] falhou")

    print(f"\n{ok}/{len(files)} conversões bem-sucedidas.")
    return 0 if ok == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())
