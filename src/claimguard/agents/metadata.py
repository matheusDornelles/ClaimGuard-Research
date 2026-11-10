"""EXIF extraction and descriptive metadata completeness features."""

from __future__ import annotations

import io
from typing import Any

from PIL import ExifTags, Image

from claimguard.schemas import AgentResult, AgentStatus

TAG_NAMES = {tag_id: name for tag_id, name in ExifTags.TAGS.items()}
GPS_NAMES = {tag_id: name for tag_id, name in ExifTags.GPSTAGS.items()}


def analyze_metadata(image_bytes: bytes) -> AgentResult:
    """Extract useful EXIF fields without treating absent metadata as evidence of fraud."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        exif = image.getexif()
        try:
            exif.update(exif.get_ifd(ExifTags.IFD.Exif))
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError, TypeError):
            gps_info = None

    values = {TAG_NAMES.get(key, str(key)): value for key, value in exif.items()}
    camera_make = _clean(values.get("Make"))
    camera_model = _clean(values.get("Model"))
    timestamp = _clean(
        values.get("DateTimeOriginal") or values.get("DateTimeDigitized") or values.get("DateTime")
    )
    software = _clean(values.get("Software"))
    gps = _extract_gps(gps_info or values.get("GPSInfo"))

    evidence = [
        "EXIF fields describe metadata present in the submitted file; they do not prove "
        "when or where the depicted event occurred.",
        "Missing or stripped metadata is common and is not, on its own, evidence of "
        "manipulation or fraud.",
    ]
    if camera_make or camera_model:
        evidence.append(
            f"Camera information: {' '.join(part for part in (camera_make, camera_model) if part)}."
        )
    else:
        evidence.append("No camera make/model was recorded.")
    if timestamp:
        evidence.append(f"Recorded capture timestamp: {timestamp}.")
    else:
        evidence.append("No EXIF capture timestamp was recorded.")
    if gps:
        evidence.append(
            "GPS coordinates are present; location accuracy and authenticity cannot be "
            "verified from EXIF alone."
        )
    else:
        evidence.append("No GPS coordinates were recorded.")

    relevant = [camera_make, camera_model, timestamp, software]
    completeness = sum(bool(value) for value in relevant) / len(relevant)
    features: dict[str, float | int | bool | str | None] = {
        "exif_tag_count": len(values),
        "camera_make_present": bool(camera_make),
        "camera_model_present": bool(camera_model),
        "timestamp_present": bool(timestamp),
        "software_present": bool(software),
        "gps_present": gps is not None,
        "metadata_completeness": round(completeness, 4),
        "camera_make": camera_make,
        "camera_model": camera_model,
        "timestamp": timestamp,
        "software": software,
    }

    return AgentResult(
        agent="metadata",
        status=AgentStatus.AVAILABLE,
        score=round(completeness, 4),
        score_name="metadata completeness",
        score_semantics=(
            "Descriptive fraction of selected EXIF fields present; "
            "not a trust or fraud probability."
        ),
        confidence=None,
        evidence=evidence,
        features=features,
        model="Pillow EXIF parser; deterministic metadata features",
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\x00", "")
    return text or None


def _extract_gps(raw_gps: Any) -> tuple[float, float] | None:
    if not raw_gps:
        return None
    try:
        gps = {GPS_NAMES.get(key, str(key)): value for key, value in raw_gps.items()}
        latitude = _to_degrees(gps["GPSLatitude"])
        longitude = _to_degrees(gps["GPSLongitude"])
        if gps.get("GPSLatitudeRef") == "S":
            latitude *= -1
        if gps.get("GPSLongitudeRef") == "W":
            longitude *= -1
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return round(latitude, 6), round(longitude, 6)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError):
        return None
    return None


def _to_degrees(value: Any) -> float:
    def rational(part: Any) -> float:
        if hasattr(part, "numerator") and hasattr(part, "denominator"):
            return float(part.numerator) / float(part.denominator)
        if isinstance(part, tuple):
            return float(part[0]) / float(part[1])
        return float(part)

    degrees, minutes, seconds = (rational(part) for part in value)
    return degrees + minutes / 60 + seconds / 3600
