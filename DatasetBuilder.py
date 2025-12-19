"""
YOUTUBE REDHEAD FACE EXTRACTION PIPELINE (VERSÃO COMPLETA)

Implementado por: Rodrigo e Gabriel
Objetivo:
    Automatizar a construção de um dataset de FACES DE PESSOAS COM CABELO RUIVO (redhead).
    - Busca vídeos no YouTube por queries relacionadas a ruivos
    - Baixa os vídeos
    - Percorre frames (intervalo configurável)
    - Detecta rostos (Haar Cascade do OpenCV)
    - Expande bounding box para cobrir região do cabelo
    - Classifica cabelo ruivo usando análise HSV multirregional (faixas expandidas)
    - Caso seja ruivo, salva o rosto recortado e registra metadados com DeepFace

Dependências (requirements.txt):
    yt-dlp
    opencv-python
    deepface
    youtube-search-python
    tensorflow
    numpy

Estrutura de saída:
    ./videos/                # vídeos baixados
    ./faces/redhead/         # rostos classificados como ruivo (dataset)
    ./faces/non_redhead/     # (opcional) rostos NÃO ruivos
    annotations.csv          # metadados por face salva
    processed_links.txt      # histórico de URLs processadas

Observações / Recomendações:
    - O classificador HSV reduz falsos positivos comparado ao método simples,
      mas ainda é sensível a iluminação. Para maior precisão, combine com
      um classificador CNN treinado em imagens de cabelo (MobileNet etc.)
    - Ajuste os thresholds (porcentagem ruivo, número de regiões válidas) conforme seu dataset.
    - Teste com diferentes queries e intervalos de frame para equilibrar performance x cobertura.

Uso:
    python pipeline_redhead.py
"""

import os
import csv
import subprocess
import cv2
import numpy as np
from deepface import DeepFace
from youtubesearchpython import VideosSearch
import time
import gc
import pandas as pd

# ======================= CONFIGURAÇÕES =======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# YouTube / busca
SEARCH_QUERIES = [
    "entrevista pessoa ruiva",
    "pessoa ruiva falando",
    "ruivo brasil",
    "ruivas brasileiras",
    "redhead interview",
    "redhead vlog",
    "ginger people talking",
    "celebridades ruivas",
    "atores ruivos famosos",
    "atriz ruiva entrevista",
]
MAX_VIDEOS_PER_QUERY = 1  # quantos vídeos baixar por query (ajuste conforme quiser)

# Download e paths
SAVE_DIR = os.path.join(SCRIPT_DIR, "videos")          # vídeos baixados
FACES_DIR = os.path.join(SCRIPT_DIR, "faces")          # pasta principal de faces
REDHEAD_DIR = os.path.join(FACES_DIR, "redhead")       # rostos ruivos (dataset alvo)
NON_REDHEAD_DIR = os.path.join(FACES_DIR, "non_redhead") # opcional: rostos não-ruivos
PROCESSED_LINKS_FILE = os.path.join(SCRIPT_DIR, "processed_links.txt")
ANNOTATION_FILE_RED = os.path.join(SCRIPT_DIR, "annotations_red.csv")   # CSV de pessoas ruivas 
ANNOTATION_FILE_NOT_RED = os.path.join(SCRIPT_DIR, "annotations_not_red.csv")   # CSV de pessoas não ruivas

# Extração de frames
FRAME_INTERVAL = 10           # analisa 1 frame a cada N frames (reduz processamento)
MAX_EXTRACTION_SECONDS = 120  # processa apenas os primeiros N segundos do vídeo (evitar vídeos longos)
MIN_FACE_SIZE = (40, 40)     # tamanho mínimo do face bbox detectado

# Face detection (OpenCV Haar Cascade)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Classificador ruivo (HSV) - parâmetros (tuneáveis)
REQUIRED_VALID_REGIONS = 3      # número mínimo de regiões que devem indicar ruivo
REQUIRED_RATIO = 0.15         # porcentagem mínima de pixels ruivos combinados (ex.: 0.15 = 15%)
MIN_REGION_RATIO = 0.45       # mín imo de pixels ruivos por região para considerar a região válida
MIN_MEAN_SAT = 50             # saturação média mínima na região (descarta tons apagados)
MIN_MEAN_VAL = 50             # brilho médio mínimo na região (descarta sombras profundas)

# Salvar non-ruivos também?
SAVE_NON_REDHEAD = True

# DeepFace config
DEEPFACE_ENFORCE = False     # enforce_detection para DeepFace (False evita lançar erros se pequena face)
DEEPFACE_BACKEND = "opencv"

# Outros
IMAGE_QUALITY = 90  # qualidade JPEG ao salvar
# ============================================================

def setup_environment():
    """Cria diretórios e arquivos necessários"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(REDHEAD_DIR, exist_ok=True)
    if SAVE_NON_REDHEAD:
        os.makedirs(NON_REDHEAD_DIR, exist_ok=True)
    # CSV de anotações
    if not os.path.exists(ANNOTATION_FILE_RED):
        with open(ANNOTATION_FILE_RED, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "video_id", "race", "age", "gender", "emotion", "timestamp_s", "hair_color", "source_video"])

    # CSV de anotações de pessoas não ruivas 
    if not os.path.exists(ANNOTATION_FILE_NOT_RED):
        with open(ANNOTATION_FILE_NOT_RED, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "video_id", "race", "age", "gender", "emotion", "timestamp_s", "hair_color", "source_video"])

    # processed links
    if not os.path.exists(PROCESSED_LINKS_FILE):
        open(PROCESSED_LINKS_FILE, "w", encoding='utf-8').close()

def save_processed_link(link):
    """Grava link processado para evitar reprocessamento"""
    with open(PROCESSED_LINKS_FILE, "a", encoding='utf-8') as f:
        f.write(link + "\n")

def load_processed_links():
    """Retorna conjunto de links já processados"""
    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, "r", encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def search_youtube_videos(query, limit=1):
    """Busca vídeos do YouTube usando youtube-search-python"""
    try:
        search = VideosSearch(query, limit=limit)
        results = search.result()
        return [(video['title'], video['link']) for video in results.get('result', [])]
    except Exception as e:
        print(f"[search_youtube_videos] erro na busca '{query}': {e}")
        return []

def download_video(url, output_dir):
    """
    Baixa vídeo com yt-dlp.
    Retorna (success_bool, caminho_para_arquivo, video_id)
    """
    try:
        # extrair id simples (funciona para URLs padrão)
        video_id = url.split("v=")[-1].split("&")[0]
        output_template = os.path.join(output_dir, f"{video_id}.mp4")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-split-chapters",
            "--no-keep-video",
            "-o", output_template,
            "--no-playlist",
            url
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(output_template):
            print(f"[download_video] Vídeo salvo em: {output_template}")
            return True, output_template, video_id
        else:
            print(f"[download_video] Falha: arquivo esperado não foi criado: {output_template}")
            return False, None, None
    except Exception as e:
        print(f"[download_video] Erro ao baixar {url}: {e}")
        return False, None, None

def get_next_image_index(output_dir):
    """Retorna próximo índice numérico para nome de arquivo (00001.jpg, etc.)"""
    existing = [f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    indices = []
    for f in existing:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            indices.append(int(name))
    return max(indices, default=0) + 1

def lerColunaRace(caminho):
    # Lendo a coluna race do .csv
    try:
        # Abre o arquivo
        arquivo = pd.read_csv(caminho)
        if 'race' in arquivo.columns:
            # Retorna Lista de todos os elementos da coluna
            return arquivo['race'].astype(str).tolist()
        else:
            print("Coluna 'race' não encontrada")

    except FileNotFoundError:
        print("Erro: arquivo não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro :{e}")

def confusionMatrix():
    # Lendo a coluna race do .csv de pessoas ruivas
    listaRed=lerColunaRace(ANNOTATION_FILE_RED)

    # Lendo a coluna race do .csv de pessoas não ruivas
    listaNotRed=lerColunaRace(ANNOTATION_FILE_NOT_RED)

    # TInha que terminar o código ainda, mas percebi que não vai me ajudar 
# ------------------ CLASSIFICADOR RUIVO (HSV multirregional) ------------------
def is_redhead(image_bgr, face_coords):
    """
    Classificador refinado de cabelo ruivo:
    - Analisa até 3 regiões: acima da testa, lado esquerdo e lado direito do rosto
    - Usa múltiplas faixas HSV para cobrir vermelho, laranja e marrom-avermelhado
    - Verifica saturação/valor médios e porcentagem de pixels ruivos por região
    - Requer que >= REQUIRED_VALID_REGIONS indiquem cabelo ruivo e que a taxa combinada exceda REQUIRED_RATIO

    Args:
        image_bgr: frame original em BGR (OpenCV)
        face_coords: tuple (x, y, w, h) do bbox do rosto

    Returns:
        bool: True se classificado como ruivo
    """
    x, y, w, h = face_coords
    h_img, w_img = image_bgr.shape[:2]

    # definir regiões candidatas
    regions = []
    # cima (acima da testa) — usa 0.6 * altura do rosto como 'altura' de região
    y1 = max(y - int(h * 0.6), 0)
    y2 = y
    x1 = max(x, 0)
    x2 = min(x + w, w_img)
    regions.append(image_bgr[y1:y2, x1:x2])

    # lado esquerdo (pegar lateral do rosto para cabelo nas laterais)
    lx1 = max(x - int(w * 0.3), 0)
    lx2 = x
    ly1 = y
    ly2 = min(y + int(h * 0.4), h_img)
    regions.append(image_bgr[ly1:ly2, lx1:lx2])

    # lado direito
    rx1 = x + w
    rx2 = min(x + w + int(w * 0.3), w_img)
    ry1 = y
    ry2 = min(y + int(h * 0.4), h_img)
    regions.append(image_bgr[ry1:ry2, rx1:rx2])

    total_pixels = 0
    red_pixels = 0
    valid_regions = 0

    for region in regions:
        if region is None or region.size == 0:
            continue

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # máscaras para faixas que capturam tons ruivos/laranja/marrom-avermelhado
        mask1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([15, 255, 255]))    # vermelho-alaranjado
        mask2 = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([25, 255, 255]))   # laranja / cobre
        mask3 = cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 255])) # vermelho escuro
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        region_total = mask.size
        region_red = int(np.sum(mask > 0))

        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        total_pixels += region_total
        red_pixels += region_red

        # considerar a região como "válida" para ruivo se:
        # - média de saturação e brilho aceitáveis (não sombra)
        # - proporção local de pixels ruivos > MIN_REGION_RATIO
        if mean_sat >= MIN_MEAN_SAT and mean_val >= MIN_MEAN_VAL and (region_red / (region_total + 1e-9)) >= MIN_REGION_RATIO:
            valid_regions += 1

    if total_pixels == 0:
        return False

    combined_ratio = red_pixels / total_pixels

    # critérios finais: pelo menos REQUIRED_VALID_REGIONS e combined_ratio > REQUIRED_RATIO
    return (valid_regions >= REQUIRED_VALID_REGIONS) and (combined_ratio > REQUIRED_RATIO)
# ------------------------------------------------------------------------------

def extract_and_classify_faces(video_path, video_id, output_dir, interval_frames=FRAME_INTERVAL, max_seconds=MAX_EXTRACTION_SECONDS):
    """
    Percorre o vídeo, detecta faces e salva aqueles classificados como ruivos.
    Retorna quantidade de faces salvas.
    """
    print(f"[extract_and_classify_faces] Iniciando processamento: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[extract_and_classify_faces] Erro ao abrir: {video_path}")
        return 0

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(fps * max_seconds)
    saved_faces = 0
    img_index_red = get_next_image_index(REDHEAD_DIR)
    img_index_not_red = get_next_image_index(NON_REDHEAD_DIR)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > max_frames:
            break

        # processar apenas frames no intervalo desejado
        if frame_count % interval_frames != 0:
            continue

        try:
            # equalizar contraste em gray para melhorar detecção de faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)

            for (x, y, w, h) in faces:
                # Expandir bbox para incluir cabelo (configurável)
                pad_w = int(0.2 * w)
                pad_h_top = int(0.6 * h)   # sobe mais para capturar cabelo
                pad_h_bottom = int(0.1 * h)

                x1 = max(x - pad_w, 0)
                y1 = max(y - pad_h_top, 0)
                x2 = min(x + w + pad_w, frame.shape[1])
                y2 = min(y + h + pad_h_bottom, frame.shape[0])

                head_region = frame[y1:y2, x1:x2]  # região contendo rosto + cabelo

                if head_region is None or head_region.size == 0:
                    continue

                # testar ruivo (no head_region) — passamos coordenadas relativas ao rosto original
                face_coords_relative = (x - x1, y - y1, w, h)  # coordenadas do rosto dentro head_region
                is_r = is_redhead(head_region, face_coords_relative)

                timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if is_r:
                    # recortar o rosto original (apenas o rosto, sem muita margem)
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop is None or face_crop.size == 0:
                        continue

                    filename = f"{img_index_red:05d}.jpg"
                    out_path = os.path.join(REDHEAD_DIR, filename)
                    cv2.imwrite(out_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])

                    # DeepFace analyze (metadados)
                    try:
                        results = DeepFace.analyze(
                            face_crop,
                            actions=['race', 'age', 'gender', 'emotion'],
                            enforce_detection=DEEPFACE_ENFORCE,
                            detector_backend=DEEPFACE_BACKEND
                        )
                        # DeepFace.analyze pode retornar dict ou lista (varia por versão); tratar ambos
                        if isinstance(results, list) and len(results) > 0:
                            res = results[0]
                        elif isinstance(results, dict):
                            res = results
                        else:
                            res = {}
                    except Exception as e:
                        # se DeepFace falhar, ainda salvamos a face mas registramos campos vazios
                        print(f"[DeepFace] erro na análise: {e}")
                        res = {}

                    race = res.get('dominant_race', '')
                    age = res.get('age', '')
                    gender = res.get('gender', '')
                    emotion = res.get('dominant_emotion', '')

                    # gravar anotação
                    with open(ANNOTATION_FILE_RED, "a", newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, video_id, race, age, gender, emotion, f"{timestamp_s:.2f}", "redhead", os.path.basename(video_path)])

                    saved_faces += 1
                    img_index_red += 1

                else:
                    # opcional: salvar faces não-ruivas para dataset negativo
                    if SAVE_NON_REDHEAD:
                        face_crop = frame[y:y+h, x:x+w]
                        if face_crop is None or face_crop.size == 0:
                            continue
                        filename = f"nr_{int(time.time()*1000)}_{np.random.randint(0,9999):04d}.jpg"
                        out_path = os.path.join(NON_REDHEAD_DIR, filename)
                        cv2.imwrite(out_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])

                        # DeepFace analyze (metadados)
                        try:
                            results = DeepFace.analyze(
                                face_crop,
                                actions=['race', 'age', 'gender', 'emotion'],
                                enforce_detection=DEEPFACE_ENFORCE,
                                detector_backend=DEEPFACE_BACKEND
                            )
                            # DeepFace.analyze pode retornar dict ou lista (varia por versão); tratar ambos
                            if isinstance(results, list) and len(results) > 0:
                                res = results[0]
                            elif isinstance(results, dict):
                                res = results
                            else:
                                res = {}
                        except Exception as e:
                            # se DeepFace falhar, ainda salvamos a face mas registramos campos vazios
                            print(f"[DeepFace] erro na análise: {e}")
                            res = {}

                        race = res.get('dominant_race', '')
                        age = res.get('age', '')
                        gender = res.get('gender', '')
                        emotion = res.get('dominant_emotion', '')

                        # gravar anotação
                        with open(ANNOTATION_FILE_NOT_RED, "a", newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([filename, video_id, race, age, gender, emotion, f"{timestamp_s:.2f}", "non_redhead", os.path.basename(video_path)])

                        saved_faces += 1
                        img_index_not_red += 1

            # fim loop faces
        except Exception as e:
            print(f"[extract_and_classify_faces] erro no frame {frame_count}: {e}")
            continue

    cap.release()
    print(f"[extract_and_classify_faces] finalizado {os.path.basename(video_path)} — salvos {saved_faces} faces ruivas")
    return saved_faces

def main():
    """Função principal do pipeline"""
    print("=== Iniciando pipeline de extração de faces ruivas ===")
    setup_environment()
    processed_links = load_processed_links()
    total_saved = 0

    for query in SEARCH_QUERIES:
        print(f"\n[main] Buscando por: '{query}'")
        videos = search_youtube_videos(query, limit=MAX_VIDEOS_PER_QUERY)
        if not videos:
            print(f"[main] Nenhum vídeo encontrado para '{query}'")
            continue

        for title, link in videos:
            try:
                print(f"\n[main] Video: {title}")
                print(f"[main] URL: {link}")

                if link in processed_links:
                    print("[main] Vídeo já processado — pulando")
                    continue

                success, video_path, video_id = download_video(link, SAVE_DIR)
                if not success:
                    print("[main] Falha no download — pular vídeo")
                    save_processed_link(link)  # opcional: marcar para evitar laços repetidos
                    continue

                # processar e salvar faces ruivas
                faces_saved = extract_and_classify_faces(video_path, video_id, REDHEAD_DIR, interval_frames=FRAME_INTERVAL, max_seconds=MAX_EXTRACTION_SECONDS)
                total_saved += faces_saved

                # marcar link como processado
                save_processed_link(link)
                processed_links.add(link)

            except Exception as e:
                print(f"[main] erro processando vídeo '{title}': {e}")
                continue

    print(f"\n=== Pipeline concluído: total de faces ruivas salvas = {total_saved} ===")

if __name__ == "__main__":
    main()
