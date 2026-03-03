from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PublicStateKey:
    """
    Lossless key for the current public state in v1 scaffolding.

    The PRD calls for lossless current-round indexing. This key is deterministic
    and intentionally explicit so changes are auditable.
    """

    street: str
    board_cards: tuple[str, ...]
    to_act: int
    pot: int
    current_bet: int
    stacks: tuple[int, ...]
    contributed_street: tuple[int, ...]
    active_mask: tuple[int, ...]
    action_history: tuple[str, ...]

    def to_token(self) -> str:
        board = "".join(self.board_cards)
        stacks = ",".join(str(v) for v in self.stacks)
        street_contrib = ",".join(str(v) for v in self.contributed_street)
        active = "".join(str(v) for v in self.active_mask)
        history = ";".join(self.action_history)
        return (
            f"street={self.street}|board={board}|to_act={self.to_act}|pot={self.pot}|bet={self.current_bet}|"
            f"stacks={stacks}|c={street_contrib}|active={active}|hist={history}"
        )


def encode_infoset_key(
    seat: int,
    private_bucket: int,
    public_state: PublicStateKey,
) -> str:
    return f"p{seat}|b{private_bucket}|{public_state.to_token()}"


def normalize_action_history(tokens: Iterable[str]) -> tuple[str, ...]:
    """Canonical action-history representation used in infoset keys."""

    out = []
    for token in tokens:
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        out.append(cleaned)
    return tuple(out)
