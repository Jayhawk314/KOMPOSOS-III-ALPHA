"""
KOMPOSOS-III Categorical Engine (Layer A)

Category theory foundations:
- Base categories with objects and morphisms
- Sheaves for multi-source data consistency
- Kan extensions for prediction (Lan) and synthesis (Ran)
- Para bicategory for parametric maps
- Grothendieck fibrations with Cartesian lifts
- Lenses and optics for forward/backward duality
"""

from .category import Object, Morphism, Category
from .kan_extensions import Functor, LeftKanExtension, RightKanExtension

__all__ = [
    "Object",
    "Morphism",
    "Category",
    "Functor",
    "LeftKanExtension",
    "RightKanExtension",
]
