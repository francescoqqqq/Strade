#!/usr/bin/env python3
import os
import random
import requests
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import math
from shapely.geometry import box

# === CONFIGURAZIONE ===
osm_file = "/home/francesco/dottorato/strade/belgium-roads.osm.pbf"
output_dir = "/home/francesco/dottorato/strade/"
num_images = 5
image_size = 512
max_attempts = 100  # Numero massimo di tentativi per trovare patch con strade
data = "imm,true"  # Opzioni: imm, lab, true (separate da virgola)

# Crea cartelle
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
true_dir = os.path.join(output_dir, "true")

# Parsing opzioni data
data_set = {x.strip() for x in data.split(',') if x.strip()}
SAVE_IMM = 'imm' in data_set
SAVE_LAB = 'lab' in data_set
SAVE_TRUE = 'true' in data_set

if SAVE_IMM:
    os.makedirs(images_dir, exist_ok=True)
if SAVE_LAB:
    os.makedirs(labels_dir, exist_ok=True)
if SAVE_TRUE:
    os.makedirs(true_dir, exist_ok=True)

print("Caricamento dati OSM...")
gdf = gpd.read_file(osm_file, layer='lines')
roads = gdf[gdf['highway'].notna()].copy()

if len(roads) == 0:
    print("ERRORE: Nessuna strada trovata!")
    exit(1)

print(f"✓ Caricate {len(roads)} strade")

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

def process_satellite_image(sat_img, bbox, size=512):
    """Processa l'immagine satellitare con matplotlib per avere lo stesso formato"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    
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

def create_road_binary_mask(roads_subset, bbox, size=512, line_width=3):
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

    return mask

def create_road_mask(roads_subset, bbox, sat_img, size=512):
    """Crea maschera con strade bianche sopra l'immagine satellitare"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    
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
    roads_subset.plot(ax=ax, color='white', linewidth=3, alpha=1.0)
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
    
    return Image.fromarray(rgb_array, mode='RGB')

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
    # === SCARICA TILE SATELLITARE (solo se serve imm o lab) ===
    if SAVE_IMM or SAVE_LAB:
        print("  Scaricando immagine satellitare...")
        sat_img_raw = download_satellite_image(bbox, zoom=17, size=image_size)
    
    # === PROCESSA IMMAGINE BASE (senza strade) ===
    if SAVE_IMM and sat_img_raw is not None:
        print("  Processando immagine base...")
        sat_img_processed = process_satellite_image(sat_img_raw, bbox, size=image_size)
        sat_img_processed.save(os.path.join(images_dir, f"road_{saved_images:04d}.png"))
    
    # === CREA MASCHERA STRADE ===
    if SAVE_LAB and sat_img_raw is not None:
        print("  Creando maschera strade con sfondo satellitare...")
        mask = create_road_mask(roads_in_patch, bbox, sat_img_raw, size=image_size)
        mask.save(os.path.join(labels_dir, f"road_{saved_images:04d}.png"))
    if SAVE_TRUE:
        if not (SAVE_IMM or SAVE_LAB):
            print("  Creando maschera strade (solo strade, senza satellite)...")
        true_mask = create_road_binary_mask(roads_in_patch, bbox, size=image_size)
        true_mask.save(os.path.join(true_dir, f"road_{saved_images:04d}.png"))
    
    print(f"  ✓ Salvata\n")
    saved_images += 1

if saved_images < num_images:
    print(f"⚠️  ATTENZIONE: Salvate solo {saved_images}/{num_images} immagini")
else:
    print("✓ COMPLETATO!")

if SAVE_IMM:
    print(f"  Images: {images_dir}")
if SAVE_LAB:
    print(f"  Labels: {labels_dir}")
if SAVE_TRUE:
    print(f"  True: {true_dir}")
