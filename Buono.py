#!/usr/bin/env python3
import os
import json
import random
import requests  # pyright: ignore[reportMissingModuleSource]
import geopandas as gpd  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from PIL import Image, ImageDraw  # pyright: ignore[reportMissingImports]
from io import BytesIO
import math
from pathlib import Path
from shapely.geometry import box  # pyright: ignore[reportMissingModuleSource]

# === CONFIGURAZIONE ===
osm_file = "/workspace/belgium-roads.osm.pbf"
dataset_id = "001"
dataset_name = "Strade"
num_images = 2000
image_size = 512
max_attempts = 100  # Numero massimo di tentativi per trovare patch con strade
data = "imm,lab,all"  # Opzioni: imm, lab, all (separate da virgola)
# imm = solo immagini satellitari → imagesTr/
# lab = solo maschere binarie strade → labelsTr/
# all = immagini satellitari + strade → allTr/

# Struttura nnU-Net
nnunet_raw_base = "/workspace/nnUNet_raw"
dataset_dir = os.path.join(nnunet_raw_base, f"Dataset{dataset_id}_{dataset_name}")
images_dir = os.path.join(dataset_dir, "imagesTr")  # Immagini satellitari RGB (imm)
labels_dir = os.path.join(dataset_dir, "labelsTr")  # Maschere binarie strade (lab)
all_dir = os.path.join(dataset_dir, "allTr")  # Immagini satellitari + strade (all)
labels_viz_dir = os.path.join(dataset_dir, "labelsTr_viz")  # Versioni visualizzabili delle label (0/255)

# Parsing opzioni data
data_set = {x.strip() for x in data.split(',') if x.strip()}
SAVE_IMM = 'imm' in data_set
SAVE_LAB = 'lab' in data_set
SAVE_ALL = 'all' in data_set

# Crea cartelle necessarie
if SAVE_IMM:
    os.makedirs(images_dir, exist_ok=True)
if SAVE_LAB:
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(labels_viz_dir, exist_ok=True)  # Anche versioni visualizzabili
if SAVE_ALL:
    os.makedirs(all_dir, exist_ok=True)

# Crea/aggiorna dataset.json se non esiste
dataset_json_path = os.path.join(dataset_dir, "dataset.json")
if not os.path.exists(dataset_json_path):
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "road": 1
        },
        "numTraining": num_images,  # Sarà aggiornato alla fine
        "file_ending": ".png",
        "name": dataset_name,
        "description": "Road segmentation from satellite imagery (RGB)",
        "reference": "Francesco Girardello - PhD Project",
        "licence": "proprietary",
        "release": "1.0"
    }
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"✓ Creato dataset.json in {dataset_dir}")

print("Caricamento dati OSM...")
gdf = gpd.read_file(osm_file, layer='lines')
roads = gdf[gdf['highway'].notna()].copy()

if len(roads) == 0:
    print("ERRORE: Nessuna strada trovata!")
    exit(1)

print(f"Strade prima del filtro: {len(roads)}")

# === FILTRO 1: Solo strade principali visibili da satellite ===
ALLOWED_HIGHWAY_TYPES = [
    'motorway', 'motorway_link',      # Autostrade
    'trunk', 'trunk_link',            # Strade di scorrimento
    'primary', 'primary_link',        # Strade primarie
    'secondary', 'secondary_link',    # Strade secondarie
    'tertiary', 'tertiary_link',      # Strade terziarie
    'residential',                     # Strade residenziali (larghe)
    # 'unclassified',                 # Strade non classificate (spesso strette)
    # 'service',                       # Strade di servizio (parcheggi, etc)
    # 'track', 'path', 'footway', 'cycleway'  # NON includere: troppo stretti/nascosti
]

roads = roads[roads['highway'].isin(ALLOWED_HIGHWAY_TYPES)].copy()

if len(roads) == 0:
    print("ERRORE: Nessuna strada valida dopo il filtro!")
    exit(1)

print(f"✓ Strade dopo filtro tipo: {len(roads)} (eliminati sentieri/piste nascoste)")

# Converti a WGS84 se necessario
if roads.crs is not None and roads.crs != 'EPSG:4326':
    print("Conversione coordinate a WGS84...")
    roads = roads.to_crs('EPSG:4326')
    print("✓ Coordinate convertite")

print(f"CRS strade: {roads.crs}")

# Ottieni bounds
bounds = roads.total_bounds  # [minx, miny, maxx, maxy]
print(f"Bounds: {bounds}")

# Dimensione patch in gradi (circa 500m)
patch_size_deg = 0.005

def latlon_to_tile(lat, lon, zoom):
    """Converte lat/lon in tile coordinate per OSM tiles"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def latlon_to_pixel_in_tile(lat, lon, zoom, tile_x, tile_y, tile_size=256):
    """Converte lat/lon in coordinate pixel all'interno di un tile"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    
    # Coordinate tile in floating point
    x_tile_float = (lon + 180.0) / 360.0 * n
    y_tile_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    # Coordinate pixel all'interno del tile
    x_pixel = (x_tile_float - tile_x) * tile_size
    y_pixel = (y_tile_float - tile_y) * tile_size
    
    return x_pixel, y_pixel

def download_satellite_image(bbox, zoom=17, size=512):
    """Scarica immagine satellitare per un bbox specifico"""
    # Calcola centro del bbox
    lon_center = (bbox[0] + bbox[2]) / 2
    lat_center = (bbox[1] + bbox[3]) / 2
    
    # Ottieni tile che contiene il centro
    tile_x, tile_y = latlon_to_tile(lat_center, lon_center, zoom)
    
    # Scarica tile centrale e quelli adiacenti (griglia 3x3)
    tiles = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y+dy}/{tile_x+dx}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    tiles.append((dx, dy, img))
            except:
                # Tile nero se fallisce
                tiles.append((dx, dy, Image.new('RGB', (256, 256), color='black')))
    
    # Crea immagine composita 3x3
    composite = Image.new('RGB', (256*3, 256*3))
    for dx, dy, tile in tiles:
        composite.paste(tile, ((dx+1)*256, (dy+1)*256))
    
    # Trova coordinate pixel degli angoli del bbox nell'immagine composita
    corners_pixel = []
    for lon, lat in [(bbox[0], bbox[3]), (bbox[2], bbox[1])]:  # top-left, bottom-right
        x_px, y_px = latlon_to_pixel_in_tile(lat, lon, zoom, tile_x-1, tile_y-1, 256)
        corners_pixel.append((x_px, y_px))
    
    # Crop e resize
    x1, y1 = corners_pixel[0]
    x2, y2 = corners_pixel[1]
    
    cropped = composite.crop((x1, y1, x2, y2))
    resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
    
    return resized

def calculate_vegetation_score(img):
    """Calcola score di vegetazione (0-1). Più alto = più verde/alberi"""
    img_array = np.array(img)
    
    # Calcola pseudo-NDVI (Normalized Difference Vegetation Index)
    # NDVI = (NIR - Red) / (NIR + Red)
    # Per RGB usiamo: (Green - Red) / (Green + Red + epsilon)
    
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    
    # Pseudo vegetation index
    epsilon = 1e-6
    veg_index = (g - r) / (g + r + epsilon)
    
    # Conta pixel "verdi" (vegetazione)
    green_pixels = np.sum(veg_index > 0.15)  # Soglia empirica
    total_pixels = veg_index.size
    
    vegetation_score = green_pixels / total_pixels
    return vegetation_score

def is_patch_valid(img, max_vegetation=0.60, min_brightness=30):
    """Valida se una patch è adatta per il training
    
    Args:
        img: Immagine PIL
        max_vegetation: Percentuale massima di vegetazione tollerata (0-1)
        min_brightness: Luminosità media minima (0-255)
    
    Returns:
        (bool, str): (is_valid, reason)
    """
    # Check 1: Troppa vegetazione
    veg_score = calculate_vegetation_score(img)
    if veg_score > max_vegetation:
        return False, f"Troppa vegetazione ({veg_score:.1%})"
    
    # Check 2: Troppo scura (ombre/nuvole)
    img_array = np.array(img)
    mean_brightness = np.mean(img_array)
    if mean_brightness < min_brightness:
        return False, f"Troppo scura (brightness={mean_brightness:.1f})"
    
    # Check 3: Troppi pixel neri (tile mancanti)
    black_pixels = np.sum(np.all(img_array == 0, axis=2))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    black_ratio = black_pixels / total_pixels
    if black_ratio > 0.10:  # Max 10% pixel neri
        return False, f"Troppi tile mancanti ({black_ratio:.1%})"
    
    return True, "OK"

def process_satellite_image(sat_img, bbox, size=512):
    """Processa l'immagine satellitare con matplotlib per avere lo stesso formato"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # pyright: ignore[reportMissingImports]
    from matplotlib.figure import Figure  # pyright: ignore[reportMissingImports]
    
    fig = Figure(figsize=(size/100, size/100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_facecolor('black')
    
    # Mostra solo l'immagine satellitare
    ax.imshow(sat_img, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], aspect='equal', interpolation='bilinear')
    # Reinforza limiti/aspect (nel caso qualche artist modifichi gli assi)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect('equal', adjustable='box')
    
    fig.set_size_inches(size/100, size/100)
    canvas.draw()
    
    buf = canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 4)
    
    # Converti in RGB
    rgb_array = img_array[:, :, :3]
    
    return Image.fromarray(rgb_array, mode='RGB')

def create_road_binary_mask(roads_subset, bbox, size=512, line_width=5):
    """Rasterizza le geometrie delle strade su una maschera binaria mono-canale (L)."""
    if roads_subset.crs is not None and roads_subset.crs != 'EPSG:4326':
        roads_subset = roads_subset.to_crs('EPSG:4326')

    minx, miny, maxx, maxy = bbox
    sx = size / (maxx - minx)
    sy = size / (maxy - miny)

    def to_px(pt):
        x, y = pt
        px = (x - minx) * sx
        py = size - (y - miny) * sy
        return (px, py)

    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    for geom in roads_subset.geometry:
        if geom is None or geom.is_empty:
            continue
        geom_type = geom.geom_type
        if geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                draw.line([to_px(pt) for pt in coords], fill=255, width=line_width)
        elif geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    draw.line([to_px(pt) for pt in coords], fill=255, width=line_width)
        elif geom_type == 'GeometryCollection':
            for sub in geom.geoms:
                if sub.geom_type == 'LineString':
                    coords = list(sub.coords)
                    if len(coords) >= 2:
                        draw.line([to_px(pt) for pt in coords], fill=255, width=line_width)
                elif sub.geom_type == 'MultiLineString':
                    for line in sub.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            draw.line([to_px(pt) for pt in coords], fill=255, width=line_width)

    # Converti da 0/255 a 0/1 per nnU-Net (le strade sono 1, background è 0)
    mask_array = np.array(mask)
    mask_array = (mask_array > 0).astype(np.uint8)  # Converti 255 → 1
    return Image.fromarray(mask_array, mode='L')

def create_road_mask(roads_subset, bbox, sat_img, size=512, line_width=5, mask_black_areas=True):
    """Crea maschera con strade bianche sopra l'immagine satellitare
    
    Args:
        mask_black_areas: Se True, rimuove le strade dalle aree completamente nere (tile mancanti)
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # pyright: ignore[reportMissingImports]
    from matplotlib.figure import Figure  # pyright: ignore[reportMissingImports]
    
    # Assicurati che le strade siano nel sistema di coordinate corretto
    if roads_subset.crs is not None and roads_subset.crs != 'EPSG:4326':
        roads_subset = roads_subset.to_crs('EPSG:4326')
    
    fig = Figure(figsize=(size/100, size/100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_facecolor('black')
    # Evita che nuovi artist modifichino i limiti automaticamente
    ax.autoscale(False)
    
    # Mostra l'immagine satellitare come sfondo
    ax.imshow(sat_img, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], aspect='equal', interpolation='bilinear')
    
    # Disegna strade in bianco sopra
    roads_subset.plot(ax=ax, color='white', linewidth=line_width, alpha=1.0)
    # Reimposta i limiti dopo il plot (GeoPandas può autoscalare)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect('equal', adjustable='box')
    
    fig.set_size_inches(size/100, size/100)
    canvas.draw()
    
    buf = canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 4)
    
    # Converti in RGB
    rgb_array = img_array[:, :, :3]
    result = Image.fromarray(rgb_array, mode='RGB')
    
    # Maschera le strade dove l'immagine satellitare è completamente nera (tile mancanti)
    if mask_black_areas:
        sat_array = np.array(sat_img)
        result_array = np.array(result)
        
        # Trova pixel completamente neri nell'immagine satellitare (tile mancanti)
        black_mask = (sat_array[:, :, 0] == 0) & (sat_array[:, :, 1] == 0) & (sat_array[:, :, 2] == 0)
        
        # Nelle aree nere, copia l'immagine satellitare originale (nero) invece delle strade bianche
        result_array[black_mask] = sat_array[black_mask]
        
        result = Image.fromarray(result_array, mode='RGB')
    
    return result

def find_patch_with_roads(roads, bounds, patch_size_deg, max_attempts=100):
    """Trova una patch casuale che contiene almeno una strada"""
    margin = patch_size_deg * 2
    
    for attempt in range(max_attempts):
        # Genera coordinate casuali
        x_center = random.uniform(bounds[0] + margin, bounds[2] - margin)
        y_center = random.uniform(bounds[1] + margin, bounds[3] - margin)
        
        # Crea bbox perfettamente quadrato
        half_size = patch_size_deg / 2
        bbox = [
            x_center - half_size,
            y_center - half_size,
            x_center + half_size,
            y_center + half_size
        ]
        
        # Verifica se ci sono strade
        bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        roads_in_patch = roads[roads.intersects(bbox_geom)]
        
        if len(roads_in_patch) > 0:
            return bbox, roads_in_patch, (x_center, y_center)
    
    return None, None, None

print(f"\nGenerazione {num_images} immagini con strade...\n")

saved_images = 0
attempts = 0

while saved_images < num_images and attempts < max_attempts * num_images:
    attempts += 1
    
    # Cerca una patch con strade
    bbox, roads_in_patch, center = find_patch_with_roads(roads, bounds, patch_size_deg, max_attempts=50)
    
    if bbox is None:
        print(f"⚠️  Nessuna patch con strade trovata dopo {attempts} tentativi")
        continue
    
    x_center, y_center = center
    print(f"Patch {saved_images+1}/{num_images} - Centro: ({x_center:.4f}, {y_center:.4f})")
    print(f"  Trovate {len(roads_in_patch)} strade")
    
    sat_img_raw = None
    # === SCARICA TILE SATELLITARE (solo se serve imm o all) ===
    if SAVE_IMM or SAVE_ALL:
        print("  Scaricando immagine satellitare...")
        sat_img_raw = download_satellite_image(bbox, zoom=17, size=image_size)
        
        # === VALIDAZIONE PATCH (FILTRO 2: Qualità immagine) ===
        is_valid, reason = is_patch_valid(sat_img_raw, max_vegetation=0.60, min_brightness=30)
        if not is_valid:
            print(f"  ⚠️  Patch scartata: {reason}")
            continue  # Salta questa patch e prova la prossima
    
    # === SALVA IMMAGINE SATELLITARE RGB (imm) ===
    if SAVE_IMM and sat_img_raw is not None:
        print("  Processando immagine satellitare...")
        sat_img_processed = process_satellite_image(sat_img_raw, bbox, size=image_size)
        # Salva immagine RGB completa (nnU-Net NaturalImage2DIO gestisce RGB automaticamente)
        img_filename = f"{dataset_name.lower()}_{saved_images:04d}_0000.png"
        sat_img_processed.save(os.path.join(images_dir, img_filename))
    
    # === SALVA IMMAGINE SATELLITARE + STRADE (all) ===
    if SAVE_ALL and sat_img_raw is not None:
        print("  Creando immagine satellitare + strade...")
        all_img = create_road_mask(roads_in_patch, bbox, sat_img_raw, size=image_size)
        all_filename = f"{dataset_name.lower()}_{saved_images:04d}.png"
        all_img.save(os.path.join(all_dir, all_filename))
    
    # === SALVA MASCHERA STRADE BINARIA (lab) ===
    if SAVE_LAB:
        print("  Creando maschera strade binaria...")
        lab_mask = create_road_binary_mask(roads_in_patch, bbox, size=image_size)
        
        # Verifica che ci siano abbastanza pixel strada
        road_pixels = np.sum(np.array(lab_mask) > 0)
        if road_pixels < 50:  # Almeno 50 pixel di strada
            print(f"  ⚠️  Troppo pochi pixel strada ({road_pixels}), patch scartata")
            continue
        # lab_mask ora contiene valori 0 e 1 (corretto per nnUNet)
        lbl_filename = f"{dataset_name.lower()}_{saved_images:04d}.png"
        lab_mask.save(os.path.join(labels_dir, lbl_filename))
        
        # Salva anche versione visualizzabile (0/255) per debug
        lab_mask_viz = lab_mask.point(lambda p: p * 255)
        lab_mask_viz.save(os.path.join(labels_viz_dir, lbl_filename))
    
    print(f"  ✓ Salvata\n")
    saved_images += 1

if saved_images < num_images:
    print(f"⚠️  ATTENZIONE: Salvate solo {saved_images}/{num_images} immagini")
else:
    print("✓ COMPLETATO!")

# Aggiorna dataset.json con il numero reale di immagini salvate
if os.path.exists(dataset_json_path):
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    dataset_json["numTraining"] = saved_images
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"✓ Aggiornato dataset.json con numTraining: {saved_images}")

print(f"\n📁 File salvati in:")
if SAVE_IMM:
    print(f"  Images (imm): {images_dir}")
if SAVE_LAB:
    print(f"  Labels per nnUNet (lab): {labels_dir} [valori 0/1]")
    print(f"  Labels per visualizzazione: {labels_viz_dir} [valori 0/255]")
if SAVE_ALL:
    print(f"  All (all): {all_dir}")
