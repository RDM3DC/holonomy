"""
AMA v0.1 - Analytic-Native Mesh Archive
========================================

A compact export format for phase-resolved simulations.
Stores only the "interesting" data: phase spines, conductance geometry,
and holonomy falsifier residuals.

The format is designed for:
  - Minimal redundancy (no repeated state-space)
  - Direct visualization of topological features
  - Interop with PR-Root holonomy computations
"""

import json
import struct
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class AMAExporter:
    """
    Export simulation data to Analytic-Native .ama format.
    
    The .ama file contains:
      - Header: metadata, domain info, field registry
      - Fields: compressed numpy arrays with semantic tags
      - Topology: detected defect locations and winding data
    """
    
    VERSION = "0.1"
    MAGIC = b"AMA\x00"
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.header = {
            "version": self.VERSION,
            "created": datetime.now().isoformat(),
            "generator": "PR-Root Holonomy Engine",
            "domain": None,
            "fields": [],
            "topology": {
                "defects": [],
                "total_winding": 0,
                "abelian_fraction": 1.0
            }
        }
        self.field_data: Dict[str, np.ndarray] = {}
        self._finalized = False
    
    def set_domain(self, shape: Tuple[int, ...], extents: List[List[float]], 
                   units: str = "arbitrary"):
        """
        Define the computational domain.
        
        Args:
            shape: Grid dimensions (e.g., (32, 32))
            extents: Physical bounds [[xmin, xmax], [ymin, ymax], ...]
            units: Physical units string
        """
        self.header["domain"] = {
            "shape": list(shape),
            "extents": extents,
            "units": units,
            "ndim": len(shape)
        }
    
    def add_field(self, name: str, data: np.ndarray, kind: str = "scalar",
                  colormap: str = "viridis", description: str = ""):
        """
        Add a field to the archive.
        
        Args:
            name: Field identifier
            data: NumPy array of field values
            kind: Semantic type - "scalar", "angle", "metric", "vector"
            colormap: Suggested visualization colormap
            description: Human-readable description
        """
        if self._finalized:
            raise RuntimeError("Cannot add fields after finalize()")
        
        field_meta = {
            "name": name,
            "kind": kind,
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "colormap": colormap,
            "description": description,
            "stats": {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data))
            }
        }
        
        # For angle fields, compute winding statistics
        if kind == "angle":
            field_meta["stats"]["total_rotation"] = float(np.sum(np.abs(np.gradient(data))))
        
        # For metric fields (falsifiers), compute defect locations
        if kind == "metric":
            threshold = np.mean(data) + 2 * np.std(data)
            defect_mask = data > threshold
            defect_locs = np.argwhere(defect_mask)
            field_meta["defect_count"] = len(defect_locs)
            
            # Store defect locations in topology
            for loc in defect_locs[:100]:  # Limit to 100 defects
                self.header["topology"]["defects"].append({
                    "location": loc.tolist(),
                    "strength": float(data[tuple(loc)])
                })
        
        self.header["fields"].append(field_meta)
        self.field_data[name] = data
    
    def set_topology(self, total_winding: int, abelian_fraction: float):
        """
        Set global topology metadata.
        
        Args:
            total_winding: Net winding number of the configuration
            abelian_fraction: Fraction of domain where scalar ops suffice
        """
        self.header["topology"]["total_winding"] = total_winding
        self.header["topology"]["abelian_fraction"] = abelian_fraction
    
    def finalize(self):
        """
        Write the .ama file to disk.
        """
        if self._finalized:
            return
        
        # Compute abelian fraction from falsifier if present
        for field in self.header["fields"]:
            if field["kind"] == "metric":
                data = self.field_data[field["name"]]
                threshold = np.mean(data) + 2 * np.std(data)
                abelian_frac = 1.0 - np.mean(data > threshold)
                self.header["topology"]["abelian_fraction"] = float(abelian_frac)
                break
        
        # Create output directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with gzip.open(self.filepath, 'wb') as f:
            # Magic number
            f.write(self.MAGIC)
            
            # Header as JSON
            header_json = json.dumps(self.header, indent=2).encode('utf-8')
            f.write(struct.pack('<I', len(header_json)))
            f.write(header_json)
            
            # Field data as compressed numpy
            for name, data in self.field_data.items():
                # Field name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                
                # Data as raw bytes
                data_bytes = data.astype(np.float32).tobytes()
                f.write(struct.pack('<I', len(data_bytes)))
                f.write(data_bytes)
        
        self._finalized = True
        print(f"[AMA] Exported: {self.filepath}")
        print(f"[AMA] Fields: {len(self.header['fields'])}")
        print(f"[AMA] Abelian fraction: {self.header['topology']['abelian_fraction']:.2%}")
        print(f"[AMA] Defects detected: {len(self.header['topology']['defects'])}")


class AMAReader:
    """
    Read .ama files back into memory.
    """
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.header: Dict[str, Any] = {}
        self.fields: Dict[str, np.ndarray] = {}
        self._load()
    
    def _load(self):
        with gzip.open(self.filepath, 'rb') as f:
            # Verify magic
            magic = f.read(4)
            if magic != AMAExporter.MAGIC:
                raise ValueError(f"Invalid AMA file: bad magic {magic}")
            
            # Read header
            header_len = struct.unpack('<I', f.read(4))[0]
            header_json = f.read(header_len).decode('utf-8')
            self.header = json.loads(header_json)
            
            # Read fields
            while True:
                name_len_bytes = f.read(4)
                if not name_len_bytes:
                    break
                name_len = struct.unpack('<I', name_len_bytes)[0]
                name = f.read(name_len).decode('utf-8')
                
                data_len = struct.unpack('<I', f.read(4))[0]
                data_bytes = f.read(data_len)
                
                # Find field metadata
                field_meta = next(fm for fm in self.header['fields'] if fm['name'] == name)
                shape = tuple(field_meta['shape'])
                
                data = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
                self.fields[name] = data
    
    def get_field(self, name: str) -> np.ndarray:
        return self.fields[name]
    
    def get_defects(self) -> List[Dict]:
        return self.header['topology']['defects']
    
    def get_abelian_fraction(self) -> float:
        return self.header['topology']['abelian_fraction']
    
    def summary(self):
        print(f"[AMA] File: {self.filepath}")
        print(f"[AMA] Version: {self.header['version']}")
        print(f"[AMA] Domain: {self.header['domain']['shape']}")
        print(f"[AMA] Fields: {[f['name'] for f in self.header['fields']]}")
        print(f"[AMA] Abelian: {self.get_abelian_fraction():.2%}")
        print(f"[AMA] Defects: {len(self.get_defects())}")
