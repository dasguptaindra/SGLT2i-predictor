"""Explicit mixed PaDEL/Mordred descriptors for the supplied trained model."""
from pathlib import Path
import csv
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import padelpy
from mordred import Calculator, descriptors
from mordred.error import Missing, Error
from rdkit import Chem

FEATURE_SOURCES = {
    "MAXaaN": "Mordred",
    "minaaN": "PaDEL",
    "nN": "Mordred",
    "nFARing": "Mordred",
    "naHRing": "Mordred",
    "MAXsCl": "Mordred",
    "NaaN": "Mordred",
    "nHBAcc_Lipinski": "PaDEL",
    "BCUTs-1h": "Mordred",
    "nFAHRing": "Mordred",
    "ATSC2c": "Mordred",
    "MDEC-33": "Mordred",
    "MATS2c": "Mordred",
}


def padel_values(smiles):
    if shutil.which("java") is None:
        raise RuntimeError("Java is missing. Deploy packages.txt with default-jre-headless, or install Java and add it to PATH locally.")
    # Restrict computation to the two needed descriptor families.
    template = Path(padelpy.__file__).parent / "PaDEL-Descriptor" / "descriptors.xml"
    tree = ET.parse(template)
    for group in tree.getroot().findall("Group"):
        for item in group.findall("Descriptor"):
            enabled = group.get("name") == "2D" and item.get("name") in {"EStateAtomType", "HBondAcceptorCount"}
            item.set("value", str(enabled).lower())
    with tempfile.TemporaryDirectory(prefix="sglt2_padel_") as folder:
        folder = Path(folder)
        smi = folder / "query.smi"
        output = folder / "descriptors.csv"
        settings = folder / "selected.xml"
        smi.write_text(smiles + "\tquery\n", encoding="utf-8")
        tree.write(settings, encoding="utf-8", xml_declaration=True)
        padelpy.padeldescriptor(
            mol_dir=str(smi), d_file=str(output), descriptortypes=str(settings),
            d_2d=True, d_3d=False, fingerprints=False, convert3d=False,
            threads=1, maxruntime=60000, sp_timeout=90, headless=True,
            removesalt=False, standardizenitro=False,
            standardizetautomers=False, detectaromaticity=False,
        )
        if not output.exists():
            raise ValueError("PaDEL did not produce descriptor output.")
        with output.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError("PaDEL did not return exactly one molecule.")
        return rows[0]


def calculate_descriptors(smiles, feature_names):
    smiles = smiles.strip()
    if not smiles or any(c.isspace() for c in smiles):
        raise ValueError("Enter one SMILES string without names, spaces or line breaks.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty molecule.")
    if set(feature_names) != set(FEATURE_SOURCES) or len(feature_names) != len(FEATURE_SOURCES):
        raise ValueError("The model features do not match the configured descriptor sources.")
    # Preserve the submitted structure; do not silently strip salts or normalize tautomers.
    pa = padel_values(smiles)
    available = {str(d): d for d in Calculator(descriptors, ignore_3D=True).descriptors}
    selected = [available[n] for n in feature_names if FEATURE_SOURCES[n] == "Mordred"]
    mo = {str(k): v for k, v in Calculator(selected)(mol).items()}
    row, errors, filled = {}, [], []
    for name in feature_names:
        source = pa if FEATURE_SOURCES[name] == "PaDEL" else mo
        if name not in source:
            errors.append(f"{name}: absent output column")
            continue
        value = source[name]
        if isinstance(value, Error):
            errors.append(f"{name}: {value}")
            continue
        # User confirmed missing descriptor values were replaced with zero in training.
        if isinstance(value, Missing) or value is None or (
            isinstance(value, str) and value.strip().lower() in {"", "nan", "na", "null"}
        ):
            row[name] = 0.0
            filled.append(name)
            continue
        try:
            value = float(value)
            if not np.isfinite(value):
                if np.isnan(value):
                    row[name] = 0.0
                    filled.append(name)
                    continue
                raise ValueError("infinite")
            row[name] = value
        except (ValueError, TypeError):
            errors.append(f"{name} ({FEATURE_SOURCES[name]})")
    if errors:
        raise ValueError(
            "Undefined or failed descriptors: " + ", ".join(errors)
            + ". Prediction stopped because the descriptor calculation failed."
        )
    frame = pd.DataFrame([row], columns=list(feature_names))
    frame.attrs["zero_filled"] = filled
    return frame
