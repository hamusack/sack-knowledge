def profile(config, name, *, default_external=False):
    if name is None:
        return ("external" if default_external else "local"), default_external
    entries = config.get("profiles", {})
    entry = entries.get(name) if isinstance(entries, dict) else None
    if isinstance(entry, dict) and type(entry.get("external")) is bool:
        return name, entry["external"]
    return "external", True


def visible_record(alias="r", *, external=False):
    permission = "AND a.external_ai=1" if external else ""
    unrestricted = "" if external else f"{alias}.source <> 'discord' OR "
    return f"""{alias}.deleted=0 AND ({unrestricted}EXISTS (
        SELECT 1 FROM allow_list a
        WHERE a.source={alias}.source AND a.target={alias}.account AND a.local_storage=1 {permission}
    ))"""


def visible_chunk(alias="c", *, external=False):
    anchor = f"""EXISTS (
        SELECT 1 FROM records anchor
        WHERE anchor.id={alias}.first_record_id AND {visible_record("anchor", external=external)}
    )"""
    if not external:
        return anchor
    members = f"SELECT 1 FROM chunk_records member WHERE member.chunk_id={alias}.id"
    return f"""{anchor} AND EXISTS ({members}) AND NOT EXISTS (
        {members} AND NOT EXISTS (
            SELECT 1 FROM records candidate
            WHERE candidate.id=member.record_id AND {visible_record("candidate", external=True)}
        )
    )"""
