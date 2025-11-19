# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-15

### Added
- **Initial release** of Dataset001_Strade
- Script `made_dataset.py` for automatic dataset generation from OSM + satellite imagery
- Support for ESRI World Imagery tiles (zoom 17, ~2.39m/pixel)
- Binary road segmentation masks (0/1 format for nnU-Net)
- Visualization versions of labels (0/255) for debugging
- Automatic patch sampling with road presence verification
- 2000 images (512×512 RGB) + corresponding masks
- Complete nnU-Net integration:
  - `dataset.json` with proper metadata
  - Compatible with `NaturalImage2DIO` reader
  - 3-channel RGB input support
- Preprocessing completed (nnUNet_preprocessed/)
- Training started (fold_0 available)

### Features
- **Multi-output modes:**
  - `imm`: Satellite images only → `imagesTr/`
  - `lab`: Binary road masks → `labelsTr/`
  - `all`: Satellite + roads overlay → `allTr/`
- **Robust tile downloading:**
  - 3×3 tile grid for large coverage
  - Fallback to black tiles on download failure
  - Black area masking to avoid false roads
- **Coordinate handling:**
  - Automatic CRS conversion to WGS84
  - Precise lat/lon to pixel mapping
  - Web Mercator tile system support

### Documentation
- Comprehensive `README.md` with:
  - Project structure
  - Dataset statistics
  - Installation instructions
  - Training guide
  - Inference examples
  - Troubleshooting section
- `visualize_samples.py` utility script for dataset inspection
- `config.yaml` for centralized configuration (future use)
- `requirements.txt` for dependency management
- `.gitignore` for version control

### Technical Details
- **Dataset Statistics:**
  - 2000 images (704 MB)
  - 2000 labels (9.8 MB)
  - Average road coverage: ~3%
  - Median image size: 479×512 pixels
- **nnU-Net Configuration:**
  - Architecture: 2D U-Net (8 stages)
  - Patch size: 512×512
  - Batch size: 12
  - Normalization: ZScore per channel (R, G, B)
  - Features per stage: [32, 64, 128, 256, 512, 512, 512, 512]

### Dependencies
- nnunetv2 >= 2.0
- geopandas >= 0.14.0
- shapely >= 2.0.0
- Pillow >= 10.0.0
- matplotlib >= 3.8.0
- numpy >= 1.24.0
- requests >= 2.31.0

## [Upcoming]

### Planned Features
- [ ] Support for custom OSM highway types filtering
- [ ] Multi-region dataset expansion (beyond Belgium)
- [ ] Data augmentation strategies
- [ ] Train/validation split generation
- [ ] Post-processing for topology refinement
- [ ] Custom metrics for road connectivity
- [ ] Export trained model for deployment
- [ ] Docker container for reproducibility
- [ ] Colab notebook for demos
- [ ] Integration with other tile providers (Google, Mapbox, etc.)

### Known Issues
- Some patches may have low road coverage (<1%)
- Black tiles from failed downloads can appear in composite images
- No explicit validation set (rely on cross-validation)

---

**Author:** Francesco Girardello  
**Project:** PhD Research - Road Segmentation from Satellite Imagery  
**Framework:** nnU-Net v2

