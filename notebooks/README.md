# Notebooks

Jupyter notebooks used for exploration, validation and diagnostics during development. They are not part of the production pipeline but document the analysis process.

| Notebook | Description |
|---|---|
| `veo_banyoles_pipeline.ipynb` | End-to-end pipeline run on the Banyoles match (main validation notebook) |
| `analisis_formacion.ipynb` | Formation detection and positional block analysis |
| `analisis_ids.ipynb` | Track ID fragmentation analysis and Re-ID evaluation |
| `analisis_pkl.ipynb` | Inspection of saved detection/tracking `.pkl` files |
| `clasificacion_pipeline.ipynb` | Team classification pipeline exploration (HSV histograms, Hellinger distance) |
| `comparacion_trackers.ipynb` | Comparison between tracker versions (Kalman variants, Re-ID on/off) |
| `detections_viewer.ipynb` | Frame-by-frame detection viewer for debugging |
| `diagnostico_tracking.ipynb` | Tracking diagnostics: lost tracks, ID switches, seam zone issues |
| `homography_field.ipynb` | Homography estimation and reprojection error analysis |

## Usage

```bash
pip install jupyter
jupyter notebook
```

Notebooks expect the production pipeline files (`pipeline_core.py`, `veo_pipeline.py`) to be importable from the parent directory.
