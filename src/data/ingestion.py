"""Reading input files: multiband GeoTIFFs, single-band rasters and CSV tables.

Rasterio is used when it is installed because it understands every GeoTIFF
variant and georeferencing. Without it, the lightweight `tifffile` package is
used, which is enough for ordinary uncompressed or deflate-compressed TIFFs
like the bundled demonstration files.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from src.data.validation import ValidationError

try:  # optional dependency
    import rasterio
    from rasterio.io import MemoryFile

    RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RASTERIO_AVAILABLE = False

import tifffile

GDAL_NODATA_TAG = 42113
GEOTIFF_TIEPOINT_TAG = 33922


@dataclass
class RasterInput:
    """A raster held as a (bands, rows, cols) float array plus metadata."""

    array: np.ndarray
    nodata: float | None
    source_name: str
    reader: str
    metadata: dict = field(default_factory=dict)

    @property
    def band_count(self) -> int:
        return self.array.shape[0]


def _as_bytes(source: str | Path | BinaryIO | bytes) -> tuple[bytes, str]:
    """Accept a path, raw bytes or an uploaded file object and return bytes + a name."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    if isinstance(source, bytes):
        return source, "uploaded.tif"
    name = getattr(source, "name", "uploaded.tif")
    if hasattr(source, "seek"):
        source.seek(0)
    return source.read(), name


def _read_with_rasterio(data: bytes, name: str) -> RasterInput:
    with MemoryFile(data) as memfile, memfile.open() as dataset:
        array = dataset.read().astype("float64")
        metadata = {
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "georeferenced": dataset.crs is not None,
            "dtype": dataset.dtypes[0],
        }
        return RasterInput(array, dataset.nodata, name, "rasterio", metadata)


def _read_with_tifffile(data: bytes, name: str) -> RasterInput:
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = series.axes
        page = tif.pages[0]
        nodata_tag = page.tags.get(GDAL_NODATA_TAG)
        georeferenced = page.tags.get(GEOTIFF_TIEPOINT_TAG) is not None
        dtype = str(page.dtype)

    # Bring the array into (bands, rows, cols) order.
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim == 3 and axes.endswith("S"):
        array = np.moveaxis(array, -1, 0)  # pixel-interleaved (rows, cols, bands)
    elif array.ndim != 3:
        raise ValidationError(
            f"'{name}' has an unsupported layout (axes '{axes}', shape {array.shape}). "
            "Expected a single image with one or more bands."
        )

    nodata = None
    if nodata_tag is not None:
        try:
            nodata = float(str(nodata_tag.value).strip("\x00 "))
        except ValueError:
            nodata = None

    metadata = {"crs": None, "georeferenced": georeferenced, "dtype": dtype}
    return RasterInput(array.astype("float64"), nodata, name, "tifffile", metadata)


def read_raster(source: str | Path | BinaryIO | bytes) -> RasterInput:
    """Read a single- or multi-band TIFF/GeoTIFF into a RasterInput."""
    data, name = _as_bytes(source)
    if not data:
        raise ValidationError(f"'{name}' is empty.")
    try:
        if RASTERIO_AVAILABLE:
            return _read_with_rasterio(data, name)
        return _read_with_tifffile(data, name)
    except ValidationError:
        raise
    except Exception as error:  # corrupt or unsupported file
        raise ValidationError(
            f"Could not read '{name}' as a TIFF/GeoTIFF raster ({type(error).__name__}: {error})."
        ) from error


def read_csv_table(source: str | Path | BinaryIO | bytes) -> tuple[pd.DataFrame, str]:
    """Read a CSV file into a DataFrame. Returns the frame and the source name."""
    data, name = _as_bytes(source)
    if not data or not data.strip():
        raise ValidationError(f"'{name}' is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValidationError(f"'{name}' is not a readable CSV file: {error}") from error
    except pd.errors.EmptyDataError as error:
        raise ValidationError(f"'{name}' contains no data rows.") from error
    if frame.empty:
        raise ValidationError(f"'{name}' has a header but no data rows.")
    return frame, name
