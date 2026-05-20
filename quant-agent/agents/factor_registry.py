from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from agents.factor_templates import FactorDirection, FactorTemplate

FactorSourceType = Literal["template", "composite", "generated", "manual"]
DEFAULT_COMPOSITE_METHOD = "weighted_sum"
DEFAULT_COMPOSITE_NORMALIZE = "none"
SUPPORTED_COMPOSITE_METHODS = frozenset({DEFAULT_COMPOSITE_METHOD})
SUPPORTED_COMPOSITE_NORMALIZERS = frozenset({"none", "rank_pct", "zscore"})
_COMPOSITE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class CompositeFactorComponent:
    """One weighted input into a composite factor."""

    factor: str
    weight: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CompositeFactorComponent:
        factor = _required_str(payload, "factor")
        weight = _required_float(payload, "weight")
        return cls(factor=factor, weight=weight)

    @property
    def factor_column(self) -> str:
        return factor_column_reference(self.factor)

    def to_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class CompositeFactorSpec:
    """Validated definition for a weighted composite factor."""

    name: str
    components: tuple[CompositeFactorComponent, ...]
    method: str = DEFAULT_COMPOSITE_METHOD
    normalize: str = DEFAULT_COMPOSITE_NORMALIZE

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CompositeFactorSpec:
        name = _validate_composite_name(_required_str(payload, "name"))
        method = _optional_str(payload, "method", DEFAULT_COMPOSITE_METHOD)
        if method not in SUPPORTED_COMPOSITE_METHODS:
            msg = (
                f"composite factor {name} method must be one of: "
                f"{', '.join(sorted(SUPPORTED_COMPOSITE_METHODS))}."
            )
            raise ValueError(msg)
        normalize = _optional_str(payload, "normalize", DEFAULT_COMPOSITE_NORMALIZE)
        if normalize not in SUPPORTED_COMPOSITE_NORMALIZERS:
            msg = (
                f"composite factor {name} normalize must be one of: "
                f"{', '.join(sorted(SUPPORTED_COMPOSITE_NORMALIZERS))}."
            )
            raise ValueError(msg)

        raw_components = payload.get("components")
        if (
            isinstance(raw_components, str)
            or not isinstance(raw_components, Sequence)
            or not raw_components
        ):
            msg = f"composite factor {name} components must be a non-empty sequence."
            raise ValueError(msg)

        components = []
        for item in raw_components:
            if not isinstance(item, Mapping):
                msg = f"composite factor {name} components must contain objects."
                raise ValueError(msg)
            components.append(CompositeFactorComponent.from_mapping(item))

        component_refs = [component.factor_column for component in components]
        if len(component_refs) != len(set(component_refs)):
            msg = f"composite factor {name} contains duplicate components."
            raise ValueError(msg)
        if not any(component.weight != 0.0 for component in components):
            msg = f"composite factor {name} must contain at least one non-zero weight."
            raise ValueError(msg)

        return cls(
            name=name,
            components=tuple(components),
            method=method,
            normalize=normalize,
        )

    @property
    def factor_column(self) -> str:
        return factor_column_reference(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "factor_column": self.factor_column,
            "method": self.method,
            "normalize": self.normalize,
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """Research metadata for one concrete factor matrix column."""

    factor_id: str
    factor_column: str
    name: str
    source_type: FactorSourceType
    formula: str
    hypothesis: str
    category: str
    direction: FactorDirection
    lookback_days: int
    data_lag_days: int
    required_columns: tuple[str, ...] = ()
    parameters: Mapping[str, Any] | None = None
    signal_tags: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    components: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_template(cls, template: FactorTemplate) -> FactorDefinition:
        return cls(
            factor_id=template.template_id,
            factor_column=factor_column_reference(template.template_id),
            name=template.name,
            source_type="template",
            formula=template.expression,
            hypothesis=template.description,
            category=template.category,
            direction=template.direction,
            lookback_days=template.lookback_days,
            data_lag_days=0,
            required_columns=template.required_columns,
            parameters=template.parameters,
            signal_tags=template.signal_tags,
            risk_flags=template.risk_flags,
        )

    @classmethod
    def from_composite(
        cls,
        spec: CompositeFactorSpec,
        registry: FactorDefinitionRegistry,
    ) -> FactorDefinition:
        component_definitions = []
        for component in spec.components:
            try:
                component_definitions.append(registry.get(component.factor_column))
            except KeyError as exc:
                msg = (
                    f"Composite factor {spec.name} references missing factor column: "
                    f"{component.factor_column}."
                )
                raise ValueError(msg) from exc
        component_records = []
        formula_terms = []
        for component, definition in zip(
            spec.components,
            component_definitions,
            strict=True,
        ):
            component_records.append(
                {
                    "factor": component.factor,
                    "factor_column": component.factor_column,
                    "weight": component.weight,
                    "source_type": definition.source_type,
                    "category": definition.category,
                    "direction": definition.direction,
                    "lookback_days": definition.lookback_days,
                    "data_lag_days": definition.data_lag_days,
                }
            )
            input_name = component.factor_column.removeprefix("factor__")
            if spec.normalize != DEFAULT_COMPOSITE_NORMALIZE:
                input_name = f"{spec.normalize}({input_name})"
            formula_terms.append(f"{component.weight:g} * {input_name}")

        required_columns = sorted(
            {
                required_column
                for definition in component_definitions
                for required_column in definition.required_columns
            }
        )
        signal_tags = sorted(
            {
                signal_tag
                for definition in component_definitions
                for signal_tag in definition.signal_tags
            }
        )
        risk_flags = sorted(
            {
                risk_flag
                for definition in component_definitions
                for risk_flag in definition.risk_flags
            }
            | {"composite_weight_risk"}
        )
        component_names = ", ".join(
            definition.factor_column.removeprefix("factor__")
            for definition in component_definitions
        )
        return cls(
            factor_id=spec.name,
            factor_column=spec.factor_column,
            name=spec.name,
            source_type="composite",
            formula=" + ".join(formula_terms),
            hypothesis=f"Weighted composite signal built from {component_names}.",
            category="composite",
            direction="positive",
            lookback_days=max(
                definition.lookback_days
                for definition in component_definitions
            ),
            data_lag_days=max(
                definition.data_lag_days
                for definition in component_definitions
            ),
            required_columns=tuple(required_columns),
            parameters={
                "method": spec.method,
                "normalize": spec.normalize,
            },
            signal_tags=tuple(signal_tags),
            risk_flags=tuple(risk_flags),
            components=tuple(component_records),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_column": self.factor_column,
            "name": self.name,
            "source_type": self.source_type,
            "formula": self.formula,
            "hypothesis": self.hypothesis,
            "category": self.category,
            "direction": self.direction,
            "lookback_days": self.lookback_days,
            "data_lag_days": self.data_lag_days,
            "required_columns": list(self.required_columns),
            "parameters": dict(self.parameters or {}),
            "signal_tags": list(self.signal_tags),
            "risk_flags": list(self.risk_flags),
            "components": [dict(component) for component in self.components],
        }


class FactorDefinitionRegistry:
    """Read-only lookup for concrete factor matrix column definitions."""

    def __init__(self, definitions: Sequence[FactorDefinition] = ()) -> None:
        self._definitions = tuple(definitions)
        self._by_column: dict[str, FactorDefinition] = {}
        for definition in self._definitions:
            if definition.factor_column in self._by_column:
                msg = f"Duplicate factor definition column: {definition.factor_column}."
                raise ValueError(msg)
            self._by_column[definition.factor_column] = definition

    @classmethod
    def from_templates_and_composites(
        cls,
        templates: Sequence[FactorTemplate],
        composite_factors: Sequence[CompositeFactorSpec] = (),
    ) -> FactorDefinitionRegistry:
        registry = cls(
            [FactorDefinition.from_template(template) for template in templates]
        )
        definitions = list(registry.all())
        for composite_factor in composite_factors:
            composite_definition = FactorDefinition.from_composite(
                composite_factor,
                registry,
            )
            definitions.append(composite_definition)
            registry = cls(definitions)
        return registry

    def all(self) -> tuple[FactorDefinition, ...]:
        return self._definitions

    def get(self, factor_column: str) -> FactorDefinition:
        try:
            return self._by_column[factor_column]
        except KeyError as exc:
            msg = f"Unknown factor definition column: {factor_column}."
            raise KeyError(msg) from exc

    def to_dicts(self) -> list[dict[str, Any]]:
        return [definition.to_dict() for definition in self._definitions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_count": len(self._definitions),
            "factor_columns": [
                definition.factor_column
                for definition in self._definitions
            ],
            "definitions": self.to_dicts(),
        }


def factor_column_reference(value: str) -> str:
    factor = value.strip()
    if factor.startswith("factor__"):
        return factor
    return f"factor__{factor}"


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{key} must be a finite number."
        raise ValueError(msg)
    number = float(value)
    if not np.isfinite(number):
        msg = f"{key} must be a finite number."
        raise ValueError(msg)
    return number


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _validate_composite_name(value: str) -> str:
    if not _COMPOSITE_NAME_PATTERN.fullmatch(value):
        msg = (
            "composite factor name must contain only letters, numbers, "
            "and underscores."
        )
        raise ValueError(msg)
    return value
