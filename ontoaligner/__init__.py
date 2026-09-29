from pathlib import Path

# Load version from VERSION file
__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

from .pipeline import OntoAlignerPipeline
from ontoaligner import ontology, base, encoder, aligner, utils, postprocess

__all__ = [
    "ontology",
    "base",
    "encoder",
    "aligner",
    "utils",
    "postprocess",
    "OntoAlignerPipeline"
]
