"""Optional DeepOnto enrichment for concept-level LLM context.

This module deliberately keeps DeepOnto out of the normal ontology parser.  An
enricher is created once per ontology and adds a ``verbalized_axioms`` list to
the existing class dictionaries before they are passed to an encoder.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
import warnings


class DeepOntoAxiomEnricher:
    """Attach selected, asserted DeepOnto axiom verbalizations to concepts."""

    DEFAULT_AXIOM_TYPES = {
        "EquivalentClasses",
        "SubClassOf",
    }

    def __init__(
        self,
        ontology_path: str,
        max_axioms_per_class: int = 15,
        include_named_subclass_axioms: bool = False,
        axiom_types: Optional[Iterable[str]] = None,
        jvm_memory: str = "8g",
        vocab: Optional[Dict[str, str]] = None,
        ontology: Any = None,
        verbalizer: Any = None,
    ) -> None:
        if max_axioms_per_class < 1:
            raise ValueError("max_axioms_per_class must be at least 1.")

        requested_path = Path(ontology_path).expanduser()
        resolved_path = requested_path.resolve()
        if ontology is None and not resolved_path.is_file():
            raise FileNotFoundError(
                "DeepOnto ontology file not found. "
                f"Requested path: {ontology_path!r}; "
                f"resolved path: {resolved_path}. "
                "Use an absolute path or run from the repository root."
            )

        self.ontology_path = str(resolved_path)
        self.max_axioms_per_class = max_axioms_per_class
        self.include_named_subclass_axioms = include_named_subclass_axioms
        self.axiom_types = set(axiom_types or self.DEFAULT_AXIOM_TYPES)
        self.vocab = dict(vocab or {})

        if ontology is None or verbalizer is None:
            # Lazy imports keep DeepOnto optional for users who do not enable
            # axiom verbalization. DeepOnto's ontology module prompts for JVM
            # memory when imported before a JVM exists, which cannot work in a
            # non-interactive Slurm job. Start the JVM before importing that
            # module so DeepOnto never reaches its interactive prompt.
            import deeponto
            import jpype

            if not jpype.isJVMStarted():
                deeponto.init_jvm(jvm_memory)

            from deeponto.onto import Ontology, OntologyVerbaliser

            ontology = Ontology(self.ontology_path)
            try:
                parameters = inspect.signature(OntologyVerbaliser).parameters
            except (TypeError, ValueError):
                parameters = {}
            if vocab is not None and "vocab" in parameters:
                try:
                    verbalizer = OntologyVerbaliser(ontology, vocab=vocab)
                except TypeError as error:
                    # Some wrapped/older DeepOnto classes expose a misleading
                    # Python signature even though their constructor rejects
                    # ``vocab``. Fall back only for that compatibility error;
                    # unrelated constructor failures must remain visible.
                    if "unexpected keyword argument 'vocab'" not in str(error):
                        raise
                    verbalizer = OntologyVerbaliser(ontology)
            else:
                # Older DeepOnto releases do not accept ``vocab``. Keep the
                # mapping locally and replace IRIs in their verbalizations.
                verbalizer = OntologyVerbaliser(ontology)

        self.ontology = ontology
        self.verbalizer = verbalizer

    @staticmethod
    def build_vocab(concepts: Iterable[Dict[str, Any]]) -> Dict[str, str]:
        """Build a DeepOnto IRI-to-label vocabulary from parsed concepts."""
        vocab: Dict[str, str] = {}
        for concept in concepts:
            iri = concept.get("iri")
            label = concept.get("label") or concept.get("name")
            if iri and label:
                vocab[str(iri)] = str(label)
        return vocab

    @staticmethod
    def _axiom_type_name(axiom: Any) -> str:
        axiom_type = axiom.getAxiomType()
        if hasattr(axiom_type, "getName"):
            return str(axiom_type.getName())
        return str(axiom_type).replace(" ", "")

    @staticmethod
    def _verbalization_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(
                value.get("verbalisation")
                or value.get("verbalization")
                or value.get("verbal")
                or value.get("text")
                or ""
            )
        return str(value)

    def _verbalize_expression(self, expression: Any) -> str:
        method = getattr(self.verbalizer, "verbalise_class_expression", None)
        if not callable(method):
            raise AttributeError(
                "The installed DeepOnto OntologyVerbaliser exposes neither "
                "verbalise_axiom() nor verbalise_class_expression()."
            )
        return self._apply_vocab(self._verbalization_text(method(expression)))

    def _apply_vocab(self, text: str) -> str:
        """Replace entity IRIs in verbalized text with parsed ontology labels."""
        for iri, label in sorted(
            self.vocab.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(iri, label)
        return text

    def _verbalize_axiom(self, axiom: Any) -> str:
        """Verbalize an axiom across supported DeepOnto API versions."""
        method = getattr(self.verbalizer, "verbalise_axiom", None)
        if callable(method):
            return self._apply_vocab(self._verbalization_text(method(axiom)))

        axiom_type = self._axiom_type_name(axiom)
        if axiom_type == "SubClassOf":
            subclass = self._verbalize_expression(axiom.getSubClass())
            superclass = self._verbalize_expression(axiom.getSuperClass())
            return f"{subclass} is a subclass of {superclass}"

        if axiom_type == "EquivalentClasses":
            getter = getattr(axiom, "getClassExpressionsAsList", None)
            if callable(getter):
                expressions = list(getter())
            else:
                getter = getattr(axiom, "getClassExpressions", None)
                if not callable(getter):
                    raise AttributeError(
                        "EquivalentClasses axiom exposes no class expressions."
                    )
                expressions = list(getter())
            verbalized = [self._verbalize_expression(item) for item in expressions]
            return " is equivalent to ".join(item for item in verbalized if item)

        return ""

    def _resolve_class(self, class_iri: str) -> Any:
        owl_classes = getattr(self.ontology, "owl_classes", None)
        if isinstance(owl_classes, dict):
            return owl_classes.get(class_iri)

        getter = getattr(self.ontology, "get_owl_class", None)
        if callable(getter):
            return getter(class_iri)

        raise AttributeError(
            "The installed DeepOnto Ontology exposes neither an owl_classes "
            "index nor get_owl_class()."
        )

    def _asserted_axioms(self, owl_class: Any) -> Iterable[Any]:
        owl_ontology = getattr(self.ontology, "owl_onto", None)
        if owl_ontology is None:
            owl_ontology = getattr(self.ontology, "ontology", None)
        if owl_ontology is None or not hasattr(owl_ontology, "getAxioms"):
            raise AttributeError(
                "The installed DeepOnto Ontology does not expose its OWLAPI "
                "ontology through owl_onto or ontology."
            )
        return owl_ontology.getAxioms(owl_class)

    def _keep_axiom(self, axiom: Any, owl_class: Any) -> bool:
        axiom_type = self._axiom_type_name(axiom)
        if axiom_type not in self.axiom_types:
            return False

        if axiom_type == "SubClassOf":
            if axiom.getSubClass() != owl_class:
                return False
            if not self.include_named_subclass_axioms:
                super_class = axiom.getSuperClass()
                if hasattr(super_class, "isAnonymous") and not super_class.isAnonymous():
                    return False

        return True

    def verbalize_class_axioms(self, class_iri: str) -> List[str]:
        """Verbalize selected asserted axioms centered on ``class_iri``."""
        owl_class = self._resolve_class(class_iri)
        if owl_class is None:
            return []

        results: List[str] = []
        seen: Set[str] = set()
        for axiom in self._asserted_axioms(owl_class):
            if not self._keep_axiom(axiom, owl_class):
                continue

            #value = self._verbalize_axiom(axiom)

            try:
                value = self._verbalize_axiom(axiom)
            except RuntimeError as error:
                if "is not in one of the supported types" not in str(error):
                    raise
            
                warnings.warn(
                    f"Skipping unsupported DeepOnto axiom: {axiom}",
                    RuntimeWarning,
                )
                continue
            text = " ".join(value.split()).strip()
            normalized = text.casefold()
            if not text or normalized in seen:
                continue

            seen.add(normalized)
            results.append(text)
            if len(results) >= self.max_axioms_per_class:
                break

        return results



    def enrich(
        self,
        concepts: List[Dict[str, Any]],
        selected_iris: Optional[Iterable[str]] = None,
    ) -> None:
        """Add verbalized axioms in place, optionally only for selected IRIs."""
        selected = set(selected_iris) if selected_iris is not None else None
        for concept in concepts:
            iri = concept["iri"]
            if selected is not None and iri not in selected:
                concept["verbalized_axioms"] = []
                continue
            concept["verbalized_axioms"] = self.verbalize_class_axioms(iri)
