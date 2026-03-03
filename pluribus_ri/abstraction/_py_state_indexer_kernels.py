from pluribus_ri.core import NoLimitHoldemEngine

def build_public_state_token(
    engine: NoLimitHoldemEngine,
    history_scope: str,
) -> str:
    if engine.to_act is None and not engine.hand_complete:
        raise ValueError("engine has no active seat to act")

    street_name = engine.street.value
    if history_scope == "street":
        action_history = ";".join(
            f"p{a.seat}:{a.kind}:{a.amount}" for a in engine.action_log if a.street == street_name
        )
    elif history_scope == "all":
        action_history = ";".join(
            f"{a.street}:p{a.seat}:{a.kind}:{a.amount}" for a in engine.action_log
        )
    else:
        raise ValueError(f"unsupported history scope: {history_scope}")

    board = "".join(engine.board)
    stacks = ",".join(str(int(player.stack)) for player in engine.players)
    street_contrib = ",".join(str(int(player.contributed_street)) for player in engine.players)
    active = "".join("1" if player.in_hand else "0" for player in engine.players)
    to_act = -1 if engine.to_act is None else int(engine.to_act)
    return (
        f"street={street_name}|board={board}|to_act={to_act}|pot={int(engine.total_pot)}|bet={int(engine.current_bet)}|"
        f"stacks={stacks}|c={street_contrib}|active={active}|hist={action_history}"
    )
