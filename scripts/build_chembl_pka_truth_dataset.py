import argparse
import json
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore


BASE_API = "https://www.ebi.ac.uk/chembl/api/data"
FORMAL_CHARGE_PATTERN = re.compile(r"\[[^\]]*[+-][0-9]*[^\]]*\]")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        as_str = str(value).strip()
        if not as_str:
            return None
        return float(as_str)
    except Exception:
        return None


def has_formal_charge_token(smiles: Any) -> bool:
    if smiles is None:
        return False
    text = str(smiles)
    if not text:
        return False
    return bool(FORMAL_CHARGE_PATTERN.search(text))


def mode_or_none(values: Sequence[Any]) -> Optional[Any]:
    filtered = [v for v in values if v is not None and str(v).strip() != ""]
    if not filtered:
        return None
    counts: Dict[str, int] = {}
    value_map: Dict[str, Any] = {}
    for value in filtered:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
        value_map[key] = value
    top_key = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return value_map[top_key]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = [(v, max(w, 0.0)) for v, w in zip(values, weights)]
    total = sum(w for _, w in pairs)
    if total <= 0:
        return statistics.median(values)
    pairs.sort(key=lambda pair: pair[0])
    running = 0.0
    threshold = total / 2.0
    for value, weight in pairs:
        running += weight
        if running >= threshold:
            return value
    return pairs[-1][0]


def api_get_json(url: str, timeout: float, max_retries: int, sleep_seconds: float) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "pkate-chembl-pka/1.0"})
        try:
            with urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except HTTPError as exc:
            status = getattr(exc, "code", None)
            if status in {429, 500, 502, 503, 504} and attempt <= max_retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and str(retry_after).isdigit():
                    delay = float(retry_after)
                else:
                    delay = sleep_seconds + min(30.0, (2 ** (attempt - 1))) + random.random()
                time.sleep(delay)
                continue
            raise
        except (URLError, TimeoutError):
            if attempt <= max_retries:
                delay = sleep_seconds + min(30.0, (2 ** (attempt - 1))) + random.random()
                time.sleep(delay)
                continue
            raise


def chunked(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def build_url(path: str, params: Dict[str, Any]) -> str:
    return f"{BASE_API}/{path}?{urlencode(params, doseq=True)}"


def normalize_next_url(next_url: Optional[str], default_limit: int) -> Optional[str]:
    if not next_url:
        return None
    parsed = urlparse(next_url)
    if parsed.netloc and "ebi.ac.uk" in parsed.netloc:
        return next_url
    query = parse_qs(parsed.query)
    flat: Dict[str, Any] = {k: v if len(v) > 1 else v[0] for k, v in query.items()}
    if "limit" not in flat:
        flat["limit"] = str(default_limit)
    return build_url(parsed.path.strip("/").split("/")[-1], flat)


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def fetch_status(timeout: float, max_retries: int, sleep_seconds: float) -> Optional[str]:
    try:
        payload = api_get_json(build_url("status.json", {}), timeout, max_retries, sleep_seconds)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("chembl_db_version", "chembl_release", "release"):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def fetch_activity_rows(
    start_url: str,
    timeout: float,
    max_retries: int,
    sleep_seconds: float,
    checkpoint_path: str,
    checkpoint_state: Optional[Dict[str, Any]],
    max_pages: int,
    max_records: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    page_count = 0
    next_url: Optional[str] = start_url
    if checkpoint_state and checkpoint_state.get("next_url"):
        next_url = str(checkpoint_state["next_url"])
        page_count = int(checkpoint_state.get("pages_fetched", 0))

    while next_url:
        if max_pages > 0 and page_count >= max_pages:
            break
        payload = api_get_json(next_url, timeout, max_retries, sleep_seconds)
        activities = payload.get("activities", [])
        if not isinstance(activities, list):
            activities = []

        for activity in activities:
            if max_records > 0 and len(rows) >= max_records:
                break
            rows.append(activity)
        page_count += 1

        current_next = payload.get("page_meta", {}).get("next")
        normalized_next = normalize_next_url(current_next, default_limit=int(parse_qs(urlparse(start_url).query).get("limit", [1000])[0]))
        checkpoint_payload = {
            "next_url": normalized_next,
            "pages_fetched": page_count,
            "records_fetched": len(rows),
            "saved_at": utc_now_iso(),
        }
        save_checkpoint(checkpoint_path, checkpoint_payload)

        if max_records > 0 and len(rows) >= max_records:
            break

        next_url = normalized_next
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    final_state = {
        "next_url": next_url,
        "pages_fetched": page_count,
        "records_fetched": len(rows),
        "saved_at": utc_now_iso(),
    }
    save_checkpoint(checkpoint_path, final_state)
    return rows, final_state


def fetch_entity_map(
    endpoint: str,
    id_field: str,
    ids: Sequence[str],
    timeout: float,
    max_retries: int,
    sleep_seconds: float,
    chunk_size: int,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    unique_ids = sorted({str(item).strip() for item in ids if str(item).strip()})
    for part in chunked(unique_ids, chunk_size):
        query = {f"{id_field}__in": ",".join(part), "limit": max(1000, len(part))}
        url = build_url(f"{endpoint}.json", query)
        payload = api_get_json(url, timeout, max_retries, sleep_seconds)
        key = f"{endpoint}s"
        items = payload.get(key, [])
        if not isinstance(items, list):
            items = []
        for item in items:
            entity_id = item.get(id_field)
            if entity_id:
                result[str(entity_id)] = item
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return result


def build_raw_dataframe(
    activities: Sequence[Dict[str, Any]],
    assay_map: Dict[str, Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    molecule_map: Dict[str, Dict[str, Any]],
    chembl_release: Optional[str],
) -> pd.DataFrame:
    pulled_at = utc_now_iso()
    rows: List[Dict[str, Any]] = []
    for activity in activities:
        assay_id = str(activity.get("assay_chembl_id") or "").strip()
        molecule_id = str(activity.get("molecule_chembl_id") or "").strip()
        assay = assay_map.get(assay_id, {})
        target_id = str(activity.get("target_chembl_id") or assay.get("target_chembl_id") or "").strip()
        target = target_map.get(target_id, {})
        molecule = molecule_map.get(molecule_id, {})

        molecule_structures = molecule.get("molecule_structures") or {}
        molecule_hierarchy = molecule.get("molecule_hierarchy") or {}
        canonical_smiles = molecule_structures.get("canonical_smiles")
        parent_id = molecule_hierarchy.get("parent_chembl_id") or molecule_id

        rows.append(
            {
                "activity_id": activity.get("activity_id"),
                "record_id": activity.get("record_id"),
                "molecule_chembl_id": molecule_id or None,
                "parent_molecule_chembl_id": parent_id,
                "assay_chembl_id": assay_id or None,
                "target_chembl_id": target_id or None,
                "standard_type": activity.get("standard_type"),
                "standard_relation": activity.get("standard_relation"),
                "standard_value": activity.get("standard_value"),
                "standard_upper_value": activity.get("standard_upper_value"),
                "standard_units": activity.get("standard_units"),
                "relation": activity.get("relation"),
                "value": activity.get("value"),
                "units": activity.get("units"),
                "text_value": activity.get("text_value"),
                "data_validity_comment": activity.get("data_validity_comment"),
                "data_validity_description": activity.get("data_validity_description"),
                "potential_duplicate": activity.get("potential_duplicate"),
                "standard_flag": activity.get("standard_flag"),
                "src_id": activity.get("src_id"),
                "document_chembl_id": activity.get("document_chembl_id"),
                "document_journal": activity.get("document_journal"),
                "document_year": activity.get("document_year"),
                "assay_type": assay.get("assay_type") or activity.get("assay_type"),
                "bao_format": assay.get("bao_format"),
                "bao_endpoint": assay.get("bao_endpoint"),
                "assay_organism": assay.get("assay_organism"),
                "assay_confidence_score": assay.get("confidence_score"),
                "target_pref_name": target.get("pref_name"),
                "target_organism": target.get("organism"),
                "target_type": target.get("target_type"),
                "canonical_smiles": canonical_smiles,
                "standard_inchi_key": molecule_structures.get("standard_inchi_key"),
                "pulled_at": pulled_at,
                "chembl_release": chembl_release,
            }
        )
    return pd.DataFrame(rows)


def clean_observations(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["standard_value_num"] = pd.to_numeric(df["standard_value"], errors="coerce")

    relation_series = (
        df["standard_relation"].fillna(df["relation"]).fillna("").astype(str).str.strip()
    )
    relation_series = relation_series.replace({"": "="})
    df["relation_normalized"] = relation_series

    df["is_exact"] = df["relation_normalized"].isin(["="])
    df["is_censored"] = df["relation_normalized"].isin(["<", "<=", ">", ">="])

    df["lower_bound"] = pd.NA
    df["upper_bound"] = pd.NA
    greater_mask = df["relation_normalized"].isin([">", ">="])
    less_mask = df["relation_normalized"].isin(["<", "<="])
    equal_mask = df["relation_normalized"].eq("=")

    df.loc[greater_mask, "lower_bound"] = df.loc[greater_mask, "standard_value_num"]
    df.loc[less_mask, "upper_bound"] = df.loc[less_mask, "standard_value_num"]
    df.loc[equal_mask, "lower_bound"] = df.loc[equal_mask, "standard_value_num"]
    df.loc[equal_mask, "upper_bound"] = df.loc[equal_mask, "standard_value_num"]

    quality = pd.Series(1.0, index=df.index)
    potential_dup = pd.to_numeric(df["potential_duplicate"], errors="coerce").fillna(0)
    standard_flag = pd.to_numeric(df["standard_flag"], errors="coerce").fillna(0)

    quality = quality - (potential_dup > 0).astype(float) * 0.25
    quality = quality - (df["data_validity_comment"].fillna("").astype(str).str.strip() != "").astype(float) * 0.35
    quality = quality - (standard_flag <= 0).astype(float) * 0.2

    conf = pd.to_numeric(df["assay_confidence_score"], errors="coerce")
    quality = quality + conf.fillna(5.0) / 50.0
    quality = quality.clip(lower=0.0, upper=1.0)
    df["quality_score"] = quality

    df["drop_reason"] = ""
    df.loc[df["standard_value_num"].isna(), "drop_reason"] = "missing_numeric_value"
    df.loc[~df["relation_normalized"].isin(["=", "<", "<=", ">", ">="]), "drop_reason"] = "unsupported_relation"
    out_of_range = (df["standard_value_num"] < -5) | (df["standard_value_num"] > 20)
    df.loc[out_of_range & df["drop_reason"].eq(""), "drop_reason"] = "outside_reasonable_range"
    df.loc[df["canonical_smiles"].isna() & df["drop_reason"].eq(""), "drop_reason"] = "missing_canonical_smiles"
    df.loc[df["parent_molecule_chembl_id"].isna() & df["drop_reason"].eq(""), "drop_reason"] = "missing_parent_molecule"

    df["keep_observation"] = df["drop_reason"].eq("")
    df["obs_id"] = [f"obs_{index+1}" for index in range(len(df))]
    df["molecule_key"] = df["parent_molecule_chembl_id"].fillna(df["molecule_chembl_id"])
    df["canonical_smiles_api"] = df["canonical_smiles"]
    df["has_formal_charge_token"] = df["canonical_smiles_api"].apply(has_formal_charge_token)
    df["is_multicomponent_smiles"] = df["canonical_smiles_api"].fillna("").astype(str).str.contains(".", regex=False)
    return df


def summarize_molecules(clean_df: pd.DataFrame) -> pd.DataFrame:
    kept = clean_df[clean_df["keep_observation"]].copy()
    if kept.empty:
        return pd.DataFrame(
            columns=[
                "molecule_key",
                "canonical_smiles_api",
                "prevalent_pka",
                "prevalent_method",
                "prevalent_interval_low",
                "prevalent_interval_high",
                "n_obs_total",
                "n_obs_exact",
                "n_obs_censored",
                "iqr",
                "mad",
                "stddev",
                "max_gap",
                "conflict_flag",
                "conflict_note",
                "unique_smiles_forms",
                "n_obs_charged_form",
                "n_obs_neutral_form",
                "mixed_charge_forms",
                "dominant_relation",
                "dominant_assay_type",
                "dominant_src_id",
                "provenance_activity_ids",
                "provenance_document_ids",
            ]
        )

    output_rows: List[Dict[str, Any]] = []
    for molecule_key, group in kept.groupby("molecule_key", dropna=False):
        values = group["standard_value_num"].astype(float).tolist()
        exact_group = group[group["is_exact"]]
        censored_group = group[group["is_censored"]]
        smiles = mode_or_none(group["canonical_smiles_api"].tolist())

        exact_values = exact_group["standard_value_num"].astype(float).tolist()
        exact_weights = exact_group["quality_score"].astype(float).tolist()

        prevalent_pka: Optional[float] = None
        prevalent_method = ""
        prevalent_low: Optional[float] = None
        prevalent_high: Optional[float] = None
        conflict_flag = False
        conflict_note = ""

        if exact_values:
            prevalent_pka = weighted_median(exact_values, exact_weights)
            prevalent_low = percentile(exact_values, 0.25)
            prevalent_high = percentile(exact_values, 0.75)
            prevalent_method = "exact_weighted_median"

        lower_bounds = [safe_float(v) for v in censored_group["lower_bound"].tolist()]
        upper_bounds = [safe_float(v) for v in censored_group["upper_bound"].tolist()]
        lower_bounds = [v for v in lower_bounds if v is not None]
        upper_bounds = [v for v in upper_bounds if v is not None]

        if not exact_values and (lower_bounds or upper_bounds):
            lower = max(lower_bounds) if lower_bounds else None
            upper = min(upper_bounds) if upper_bounds else None
            prevalent_low = lower
            prevalent_high = upper
            if lower is not None and upper is not None and lower <= upper:
                prevalent_pka = (lower + upper) / 2.0
                prevalent_method = "censored_interval"
            else:
                prevalent_method = "censored_conflict"
                conflict_flag = True
                conflict_note = "inconsistent_censored_bounds"

        if exact_values and (lower_bounds or upper_bounds):
            feasible_low = max(lower_bounds) if lower_bounds else None
            feasible_high = min(upper_bounds) if upper_bounds else None
            if prevalent_pka is not None:
                if feasible_low is not None and prevalent_pka < feasible_low:
                    conflict_flag = True
                if feasible_high is not None and prevalent_pka > feasible_high:
                    conflict_flag = True
                if conflict_flag and conflict_note == "":
                    conflict_note = "exact_vs_censored_disagreement"

        stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
        med = statistics.median(values) if values else None
        mad = statistics.median([abs(v - med) for v in values]) if values and med is not None else None
        max_gap = (max(values) - min(values)) if values else None

        if max_gap is not None and max_gap > 6.0:
            conflict_flag = True
            if conflict_note == "":
                conflict_note = "wide_value_spread"

        n_obs_charged_form = int(group["has_formal_charge_token"].sum())
        n_obs_neutral_form = int((~group["has_formal_charge_token"]).sum())
        unique_smiles_forms = int(group["canonical_smiles_api"].nunique(dropna=True))
        mixed_charge_forms = bool(n_obs_charged_form > 0 and n_obs_neutral_form > 0)

        output_rows.append(
            {
                "molecule_key": molecule_key,
                "canonical_smiles_api": smiles,
                "prevalent_pka": prevalent_pka,
                "prevalent_method": prevalent_method,
                "prevalent_interval_low": prevalent_low,
                "prevalent_interval_high": prevalent_high,
                "n_obs_total": int(len(group)),
                "n_obs_exact": int(len(exact_group)),
                "n_obs_censored": int(len(censored_group)),
                "iqr": iqr,
                "mad": mad,
                "stddev": stddev,
                "max_gap": max_gap,
                "conflict_flag": bool(conflict_flag),
                "conflict_note": conflict_note,
                "unique_smiles_forms": unique_smiles_forms,
                "n_obs_charged_form": n_obs_charged_form,
                "n_obs_neutral_form": n_obs_neutral_form,
                "mixed_charge_forms": mixed_charge_forms,
                "dominant_relation": mode_or_none(group["relation_normalized"].tolist()),
                "dominant_assay_type": mode_or_none(group["assay_type"].tolist()),
                "dominant_src_id": mode_or_none(group["src_id"].tolist()),
                "provenance_activity_ids": "|".join(sorted({str(v) for v in group["activity_id"].dropna().tolist()})),
                "provenance_document_ids": "|".join(sorted({str(v) for v in group["document_chembl_id"].dropna().tolist()})),
            }
        )

    return pd.DataFrame(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh ChEMBL-derived pKa truth dataset with API canonical SMILES and provenance tables."
    )
    parser.add_argument("--out-raw", default="data/raw/chembl_pka_activity_raw.csv", help="Path for raw activity table CSV")
    parser.add_argument("--out-clean", default="data/processed/chembl_pka_observations_clean.csv", help="Path for cleaned observation table CSV")
    parser.add_argument("--out-summary", default="data/processed/chembl_pka_molecule_summary.csv", help="Path for molecule summary table CSV")
    parser.add_argument("--checkpoint", default="data/processed/chembl_pka_pull_checkpoint.json", help="Path for checkpoint JSON")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    parser.add_argument("--limit", type=int, default=1000, help="API page size")
    parser.add_argument("--max-pages", type=int, default=0, help="Maximum pages to fetch (0 = no page cap)")
    parser.add_argument("--max-records", type=int, default=0, help="Maximum activity records to fetch (0 = no record cap)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries for transient errors")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between requests")
    parser.add_argument("--chunk-size", type=int, default=100, help="Batch size for metadata lookups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for out_path in [args.out_raw, args.out_clean, args.out_summary]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    checkpoint_state = load_checkpoint(args.checkpoint) if args.resume else None

    base_params = {
        "standard_type__iexact": "pKa",
        "standard_value__isnull": "false",
        "molecule_chembl_id__isnull": "false",
        "order_by": "activity_id",
        "limit": args.limit,
    }
    start_url = build_url("activity.json", base_params)

    print("Fetching ChEMBL pKa activities...")
    activities, pull_state = fetch_activity_rows(
        start_url=start_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        checkpoint_path=args.checkpoint,
        checkpoint_state=checkpoint_state,
        max_pages=args.max_pages,
        max_records=args.max_records,
    )
    print(f"Fetched {len(activities)} activity rows across {pull_state['pages_fetched']} pages")

    if not activities:
        raise RuntimeError("No activities were fetched from ChEMBL with current filters.")

    assay_ids = [str(row.get("assay_chembl_id") or "") for row in activities]
    target_ids = [str(row.get("target_chembl_id") or "") for row in activities]
    molecule_ids = [str(row.get("molecule_chembl_id") or "") for row in activities]

    print("Fetching assay metadata...")
    assay_map = fetch_entity_map(
        endpoint="assay",
        id_field="assay_chembl_id",
        ids=assay_ids,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        chunk_size=args.chunk_size,
    )

    inferred_target_ids = target_ids + [str(item.get("target_chembl_id") or "") for item in assay_map.values()]
    print("Fetching target metadata...")
    target_map = fetch_entity_map(
        endpoint="target",
        id_field="target_chembl_id",
        ids=inferred_target_ids,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        chunk_size=args.chunk_size,
    )

    print("Fetching molecule metadata (canonical SMILES from API)...")
    molecule_map = fetch_entity_map(
        endpoint="molecule",
        id_field="molecule_chembl_id",
        ids=molecule_ids,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        chunk_size=args.chunk_size,
    )

    chembl_release = fetch_status(args.timeout, args.max_retries, args.sleep_seconds)

    raw_df = build_raw_dataframe(
        activities=activities,
        assay_map=assay_map,
        target_map=target_map,
        molecule_map=molecule_map,
        chembl_release=chembl_release,
    )
    raw_df.to_csv(args.out_raw, index=False)
    print(f"Saved raw activity table: {args.out_raw} ({len(raw_df)} rows)")

    clean_df = clean_observations(raw_df)
    clean_df.to_csv(args.out_clean, index=False)
    kept_count = int(clean_df["keep_observation"].sum())
    print(f"Saved cleaned observations: {args.out_clean} ({len(clean_df)} rows, kept={kept_count})")

    summary_df = summarize_molecules(clean_df)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"Saved molecule summary: {args.out_summary} ({len(summary_df)} rows)")

    print("Drop reasons:")
    print(clean_df["drop_reason"].replace("", "kept").value_counts().to_string())


if __name__ == "__main__":
    main()
