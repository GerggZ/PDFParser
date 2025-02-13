from dataclasses import dataclass, field



@dataclass
class SectionHeader:
    text: str
    color: tuple[int, int, int] = field(default=(1, 1, 1))

