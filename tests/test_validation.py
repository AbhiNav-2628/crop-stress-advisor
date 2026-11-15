import io

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.config import DEMO_DATA_DIR
from src.data.ingestion import RasterInput, read_csv_table, read_raster
from src.data.preprocessing import sample_from_band_files, sample_from_raster, samples_from_table
from src.data.validation import ValidationError, validate_band_mapping


def raster(array, nodata=None):
    return RasterInput(np.asarray(array, dtype="float64"), nodata, "test.tif", "test")


def test_mapping_requires_red_and_nir():
    with pytest.raises(ValidationError, match="Red and NIR"):
        validate_band_mapping({"red": 1, "nir": None}, 5)


def test_mapping_out_of_range_band():
    with pytest.raises(ValidationError, match="only has 3"):
        validate_band_mapping({"red": 1, "nir": 5}, 3)


def test_mapping_duplicate_band_numbers():
    with pytest.raises(ValidationError, match="more than one"):
        validate_band_mapping({"red": 2, "nir": 2}, 5)


def test_mapping_ignores_unprovided_bands():
    assert validate_band_mapping({"red": 1, "nir": 2, "blue": 0, "green": None}, 2) == {"red": 1, "nir": 2}


def test_misaligned_band_files_rejected(settings):
    files = {"red": raster(np.full((1, 4, 4), 0.1)), "nir": raster(np.full((1, 5, 4), 0.5))}
    with pytest.raises(ValidationError, match="not aligned"):
        sample_from_band_files(files, settings)


def test_multiband_file_rejected_as_single_band(settings):
    files = {"red": raster(np.full((2, 4, 4), 0.1)), "nir": raster(np.full((1, 4, 4), 0.5))}
    with pytest.raises(ValidationError, match="single-band"):
        sample_from_band_files(files, settings)


def test_digital_numbers_detected(settings):
    cube = np.full((5, 8, 8), 3000.0)
    with pytest.raises(ValidationError, match="digital numbers"):
        sample_from_raster(raster(cube), {"red": 3, "nir": 5}, settings)


def test_scale_factor_fixes_digital_numbers(settings):
    cube = np.stack([np.full((8, 8), v) for v in (300, 800, 400, 3000, 5000)]).astype(float)
    sample = sample_from_raster(raster(cube), {"red": 3, "nir": 5}, settings, scale_factor=10000)
    assert sample.bands["nir"].max() == pytest.approx(0.5)


def test_nodata_and_nan_pixels_masked(settings):
    cube = np.full((5, 4, 4), 0.2)
    cube[:, 0, 0] = -9999
    cube[2, 1, 1] = np.nan
    sample = sample_from_raster(raster(cube, nodata=-9999), {"red": 3, "nir": 5}, settings)
    assert sample.n_valid == 14
    assert not sample.valid_mask[0, 0] and not sample.valid_mask[1, 1]


def test_all_pixels_invalid_rejected(settings):
    cube = np.full((5, 4, 4), -9999.0)
    with pytest.raises(ValidationError, match="No valid observations"):
        sample_from_raster(raster(cube, nodata=-9999), {"red": 3, "nir": 5}, settings)


def test_csv_missing_required_column(settings):
    frame = pd.DataFrame({"red": [0.1], "green": [0.1]})
    with pytest.raises(ValidationError, match="nir"):
        samples_from_table(frame, "x.csv", settings)


def test_csv_column_aliases_and_grouping(settings):
    frame = pd.DataFrame({"Plot_ID": ["a", "a", "b"], "R": [0.05, 0.06, 0.1], "NIR": [0.5, 0.45, 0.3]})
    samples = samples_from_table(frame, "x.csv", settings)
    assert [s.sample_id for s in samples] == ["a", "b"]
    assert samples[0].n_total == 2


def test_csv_non_numeric_column_rejected(settings):
    frame = pd.DataFrame({"red": ["a", "b"], "nir": [0.5, 0.4]})
    with pytest.raises(ValidationError, match="no numeric values"):
        samples_from_table(frame, "x.csv", settings)


def test_csv_partially_bad_values_are_masked(settings):
    frame = pd.DataFrame({"red": [0.05, "oops", 0.06], "nir": [0.5, 0.4, -0.2]})
    sample = samples_from_table(frame, "x.csv", settings)[0]
    assert sample.n_valid == 1


def test_empty_and_header_only_csv_rejected():
    with pytest.raises(ValidationError, match="empty"):
        read_csv_table(b"")
    with pytest.raises(ValidationError, match="no data rows"):
        read_csv_table(b"red,nir\n")


def test_malformed_csv_rejected():
    with pytest.raises(ValidationError):
        read_csv_table(b'red,nir\n"0.1,0.5\n0.2,0.3,0.9,1\n')


def test_non_tiff_rejected():
    with pytest.raises(ValidationError, match="Could not read"):
        read_raster(DEMO_DATA_DIR / "invalid" / "not_a_tiff.tif")


def test_reads_pixel_interleaved_tiff():
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, np.random.default_rng(0).random((6, 7, 5)).astype("float32"),
                     photometric="minisblack", planarconfig="contig")
    result = read_raster(buffer.getvalue())
    assert result.array.shape == (5, 6, 7)


def test_reads_bundled_geotiff_with_nodata():
    result = read_raster(DEMO_DATA_DIR / "plot_cloud_masked.tif")
    assert result.band_count == 5
    assert result.nodata == -9999
    assert result.metadata["georeferenced"] is True
