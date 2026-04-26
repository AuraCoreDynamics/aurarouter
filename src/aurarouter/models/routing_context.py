from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RoutingContext:
    """Metadata about the routing process, including RAG context sources."""
    retrieval_used: bool = False
    sources: list[dict] = field(default_factory=list)
    # Extra fields for ZReach integration
    author_id: Optional[str] = None
    project_id: Optional[str] = None
