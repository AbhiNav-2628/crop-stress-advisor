# Limitations and Future Scope

## Scientific limitations
1. **No field validation.** Nothing has been tested against ground-truth observations of real
   crops. Real-world accuracy is unknown.
2. **Synthetic training data.** The model learns the generator's assumptions. Agreement with those
   assumptions says nothing about agreement with nature.
3. **Cause attribution is fundamentally hard.** Water deficit, nitrogen deficiency, pests, disease,
   phenology, varietal differences, waterlogging, salinity and herbicide injury can all produce
   similar index changes. The categories are *potential* causes.
4. **Pest vs disease** are separated only by a weak, assumed spectral cue (NDRE/NDVI of weak
   patches). Even on synthetic data they are often confused.
5. **No crop-specific calibration.** Thresholds and the model ignore crop, variety, growth stage,
   row spacing and sensor characteristics.
6. **Missing modalities.** No thermal (canopy temperature / CWSI), soil moisture, weather, field
   history or time series, all of which matter, especially for water stress.
7. **Mild stress can be missed.** `plot_mild_ambiguous.tif` (mild water stress) is classified
   *healthy*.
8. **Spatial scale.** One prediction per plot or sample. There is no automatic zoning within a field.
9. **Generic NDVI readings** (e.g. "> 0.6 dense canopy") are crop-independent rules of thumb.

## Technical limitations
- Without the optional rasterio package, some compressed or tiled GeoTIFF variants cannot be read,
  and the CRS is not decoded.
- Inputs must already be calibrated reflectance and co-registered. There is no radiometric or
  geometric correction.
- The out-of-distribution check is a per-feature z-score. It does not catch unusual *combinations*
  of in-range values.
- Probabilities are not calibrated. The confidence thresholds (0.60 / 0.15) are design choices.
- The PDF uses built-in Latin-1 fonts, so non-Latin characters in file names are replaced.
- Large rasters (thousands × thousands of pixels) are processed in memory and may be slow; there
  is no tiling.

## Responsible-AI considerations
- Advisories are deliberately conservative and partly illustrative. They need review by agronomists
  for any real deployment.
- Uploaded GeoTIFFs may contain location data. Reports can therefore be location-sensitive.

## Future scope
1. Collect labelled field data (plots with agronomist-confirmed status), evaluate the system,
   recalibrate thresholds per crop, and refit with `scripts/train_demo_model.py` adapted to the real data.
2. Add thermal imagery and soil-moisture inputs for water-stress specificity.
3. Use multi-date time series (stress onset, rate of change).
4. Within-field zoning: tile rasters into zones and classify each, producing stress maps.
5. Probability calibration (e.g. isotonic) and multivariate OOD detection (e.g. Mahalanobis).
6. Optional deep models (e.g. a CNN on index patches) through the same `StressClassifier`
   interface, only once real labelled data exists.
7. Regional-language advisories and agronomist-reviewed, crop-specific rule sets.
