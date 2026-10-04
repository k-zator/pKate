#!/usr/bin/env python3
"""Interactively audit experimental-pKa/structure/transition inconsistencies.

The reviewer is deliberately non-destructive: decisions are written to a
curation overlay and the source SDF files are never edited.  A later release
rebuild can quarantine rejected rows or apply an explicitly supplied
replacement SMILES with full provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw, rdFMCS  # type: ignore

from site_resolution_review import RawMolCache


DEFAULT_COMPARISON = (
    "data/processed/stage3_failure_visualization_canonical/"
    "stage3_deployment_view_of_heldout_errors.csv"
)
DEFAULT_NETWORKS = "data/processed/pka_molecule_microstate_network_stage3_predictions.csv"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_DECISIONS = "data/curation/pka_structure_consistency_review_decisions.csv"
DEFAULT_QUEUE = "data/processed/dataset_audit/pka_structure_consistency_review_queue.csv"
DEFAULT_IMAGE_DIR = "data/processed/dataset_audit/pka_structure_consistency_review_images"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766

TERMINAL_DECISIONS = {
    "keep_current_structure_and_transition",
    "exclude_measurement_structure_pair",
    "exclude_entire_source_record",
    "replace_input_smiles",
    "correct_transition_definition",
}
ALL_DECISIONS = TERMINAL_DECISIONS | {"defer"}
DECISION_COLUMNS = [
    "review_id", "source_file", "record_index", "pka_value", "molecule_id",
    "site_id", "original_smiles", "review_decision", "replacement_smiles",
    "replacement_transition_mode", "review_note", "reviewed_at",
]


def _json_list(value: object) -> List:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _canonical_unmapped_smiles(mapped_or_plain_smiles: object) -> str:
    mol = Chem.MolFromSmiles(str(mapped_or_plain_smiles))
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _heavy_elements(mol: Chem.Mol) -> Counter:
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def validate_replacement_smiles(original_smiles: str, replacement_smiles: str) -> str:
    """Return canonical replacement or raise when it changes heavy-atom identity."""
    original = Chem.MolFromSmiles(str(original_smiles))
    replacement = Chem.MolFromSmiles(str(replacement_smiles))
    if original is None:
        raise ValueError("The recorded original SMILES is not parseable")
    if replacement is None:
        raise ValueError("Replacement SMILES is not parseable by RDKit")
    if _heavy_elements(original) != _heavy_elements(replacement):
        raise ValueError("Replacement changes the heavy-atom element inventory")
    original_heavy = Chem.RemoveHs(original).GetNumAtoms()
    replacement_heavy = Chem.RemoveHs(replacement).GetNumAtoms()
    if original_heavy != replacement_heavy:
        raise ValueError("Replacement changes the number of heavy atoms")
    mcs = rdFMCS.FindMCS(
        [Chem.RemoveHs(original), Chem.RemoveHs(replacement)],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=5,
    )
    if bool(mcs.canceled) or int(mcs.numAtoms) != int(original_heavy):
        raise ValueError(
            "Replacement changes heavy-atom connectivity; only protonation, charge, "
            "bond-order, and tautomer edits are accepted here"
        )
    return Chem.MolToSmiles(replacement, canonical=True, isomericSmiles=True)


def _review_id(source_file: str, record_index: int, pka: float, molecule_id: str, site_id: str) -> str:
    key = f"{source_file}|{record_index}|{float(pka):.8g}|{molecule_id}|{site_id}"
    return "pka_structure_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _implied_form(pka: float, ph: float) -> Tuple[float, str]:
    exponent = max(-300.0, min(300.0, float(ph) - float(pka)))
    probability = 1.0 / (1.0 + 10.0 ** exponent)
    if probability > 0.5:
        return probability, "acid_form"
    if probability < 0.5:
        return probability, "base_form"
    return probability, "equal_acid_and_base"


def _conditional_node(nodes: List[Dict], site_id: str, target_form: str) -> Optional[Dict]:
    matching = [
        node for node in nodes
        if str(node.get("site_forms", {}).get(site_id, "")) == target_form
    ]
    if not matching:
        return None
    return max(
        matching,
        key=lambda node: float(node.get("stage3_configuration_population_at_ph", 0.0)),
    )


def _optional_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_decisions(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        frame = pd.read_csv(path, low_memory=False)
    else:
        frame = pd.DataFrame(columns=DECISION_COLUMNS)
    for column in DECISION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[DECISION_COLUMNS].copy()


def _decision_lookup(decisions: pd.DataFrame) -> Dict[str, pd.Series]:
    return {str(row["review_id"]): row for _, row in decisions.iterrows()}


def build_review_queue(
    comparison_path: str = DEFAULT_COMPARISON,
    networks_path: str = DEFAULT_NETWORKS,
    raw_dir: str = DEFAULT_RAW_DIR,
    decisions_path: str = DEFAULT_DECISIONS,
) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_path, low_memory=False)
    comparison = comparison[
        comparison["stage3_form_vs_experimental_pka_check"].eq("disagrees")
    ].copy()
    networks = pd.read_csv(networks_path, low_memory=False).set_index("molecule_id")
    decisions = _decision_lookup(_load_decisions(decisions_path))
    raw_cache = RawMolCache(raw_dir)
    rows: List[Dict] = []

    for _, failure in comparison.iterrows():
        molecule_id = str(failure["molecule_id"])
        site_id = str(failure["site_id"])
        if molecule_id not in networks.index:
            continue
        network = networks.loc[molecule_id]
        sites = _json_list(network["sites_json"])
        nodes = _json_list(network["stage3_microstate_nodes_json"])
        site = next((value for value in sites if str(value.get("site_id")) == site_id), None)
        if site is None:
            continue
        measurements = list(site.get("experimental_measurements") or [])
        for measurement in measurements:
            pka = float(measurement["experimental_pka"])
            ph = float(failure["population_ph"])
            implied_probability, implied_form = _implied_form(pka, ph)
            predicted_form = str(failure["stage3_predicted_site_form_at_ph"])
            if implied_form not in {"acid_form", "base_form"} or implied_form == predicted_form:
                continue
            source_file = str(measurement.get("source_file", ""))
            record_index = int(measurement.get("record_index"))
            raw_mol = raw_cache.get(source_file, record_index)
            original_smiles = Chem.MolToSmiles(raw_mol, canonical=True, isomericSmiles=True)
            proposed_node = _conditional_node(nodes, site_id, implied_form)
            proposed_mapped = "" if proposed_node is None else str(
                proposed_node.get("reference_atom_mapped_smiles", "")
            )
            proposed_smiles = _canonical_unmapped_smiles(proposed_mapped)
            review_id = _review_id(source_file, record_index, pka, molecule_id, site_id)
            prior = decisions.get(review_id)
            one_body_pka = float(failure.get(
                "deployed_assigned_one_body_pka",
                failure["deployed_assigned_local_pka"],
            ))
            deployed_macro_pka = float(failure["deployed_assigned_macro_pka"])
            rows.append({
                "review_id": review_id,
                "review_status": "pending" if prior is None else str(prior.get("review_decision", "pending")),
                "review_note": "" if prior is None else str(prior.get("review_note", "")),
                "source_file": source_file,
                "record_index": record_index,
                "pka_value": pka,
                "population_ph": ph,
                "molecule_id": molecule_id,
                "site_id": site_id,
                "site_label": str(failure["site_label"]),
                "site_family": str(failure["site_family"]),
                "validation_scope": str(failure["validation_scope"]),
                "original_smiles": original_smiles,
                "representative_atom_mapped_smiles": str(failure["atom_mapped_smiles"]),
                "input_member_form": str(failure.get("input_member_form", "")),
                "experimental_pka_implied_protonated_probability": implied_probability,
                "experimental_pka_implied_form": implied_form,
                "stage2_predicted_macro_pka": deployed_macro_pka,
                "stage3_model_implied_macro_pka": _optional_float(
                    failure.get("stage3_reconstructed_macro_pka")
                ),
                "stage3_one_body_pka": one_body_pka,
                "stage3_contextual_edge_pka_at_dominant_background": _optional_float(
                    failure.get("stage3_contextual_edge_pka_at_dominant_background")
                ),
                "stage3_contextual_edge_pka_min": _optional_float(
                    failure.get("stage3_contextual_edge_pka_min")
                ),
                "stage3_contextual_edge_pka_max": _optional_float(
                    failure.get("stage3_contextual_edge_pka_max")
                ),
                # Compatibility alias; this is a one-body coefficient, not a
                # directly observed local pKa in a coupled molecule.
                "stage3_effective_local_pka": one_body_pka,
                "stage3_protonated_probability": float(failure["stage3_protonated_probability_at_ph"]),
                "stage3_predicted_form": predicted_form,
                "stage3_dominant_atom_mapped_smiles": str(failure["dominant_atom_mapped_smiles"]),
                "proposed_pka_consistent_atom_mapped_smiles": proposed_mapped,
                "proposed_pka_consistent_smiles": proposed_smiles,
                "site_atom_maps_json": str(failure["site_atom_maps_json"]),
                "site_center_maps_json": str(failure["site_center_maps_json"]),
                "source_assay_id": str(measurement.get("source_assay_id", "")),
                "source_document_id": str(measurement.get("source_document_id", "")),
                "source_molecule_id": str(measurement.get("source_molecule_id", "")),
                "source_site_evidence": str(measurement.get("source_site_evidence", "")),
                "source_site_evidence_confidence": str(
                    measurement.get("source_site_evidence_confidence", "")
                ),
                "transition_identity_confidence": str(
                    measurement.get("transition_identity_confidence", "")
                ),
                "current_network_mapping_basis": str(measurement.get("network_mapping_basis", "")),
                "review_warning": (
                    "A high pKa on an N-H heterocycle may describe neutral-to-anion "
                    "deprotonation rather than neutral-to-cation protonation. Replacing "
                    "the drawn charge alone does not correct a wrong transition definition."
                ),
                "absolute_experimental_vs_deployed_macro_gap": abs(
                    pka - deployed_macro_pka
                ),
                "absolute_experimental_vs_stage3_local_gap": abs(
                    pka - one_body_pka
                ),
            })
    queue = pd.DataFrame(rows)
    if not queue.empty:
        queue = queue.sort_values(
            ["absolute_experimental_vs_deployed_macro_gap", "source_file", "record_index"],
            ascending=[False, True, True],
        ).drop_duplicates("review_id").reset_index(drop=True)
    return queue


def _parse_maps(raw: object) -> List[int]:
    return [int(value) for value in _json_list(raw)]


def _drawing(smiles: object, site_maps: object, center_maps: object) -> Tuple[Optional[Chem.Mol], List[int], Dict]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, [], {}
    sites = set(_parse_maps(site_maps))
    centers = set(_parse_maps(center_maps))
    colors = {}
    for atom in mol.GetAtoms():
        number = int(atom.GetAtomMapNum())
        if number in sites:
            colors[atom.GetIdx()] = (0.95, 0.75, 0.25)
        if number in centers:
            colors[atom.GetIdx()] = (0.90, 0.20, 0.20)
    return mol, sorted(colors), colors


def render_review_images(queue: pd.DataFrame, image_dir: str) -> None:
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    for _, row in queue.iterrows():
        specs = [
            (
                row["representative_atom_mapped_smiles"],
                "SOURCE DRAWING (not an observed equilibrium state)\n"
                f"drawn member={row['input_member_form']} | pKa={float(row['pka_value']):.2f}",
            ),
            (
                row["proposed_pka_consistent_atom_mapped_smiles"],
                f"HH FORM IF CURRENT PAIR IS CORRECT @ pH {float(row['population_ph']):.1f}\n"
                f"{row['experimental_pka_implied_form']} | P(protonated)="
                f"{float(row['experimental_pka_implied_protonated_probability']):.3f}",
            ),
            (
                row["stage3_dominant_atom_mapped_smiles"],
                f"STAGE 3 DOMINANT AT pH {float(row['population_ph']):.1f}\n"
                f"{row['stage3_predicted_form']} | P(prot)="
                f"{float(row['stage3_protonated_probability']):.3f}\n"
                f"macro/one-body/edge pKa="
                f"{float(row['stage2_predicted_macro_pka']):.2f}/"
                f"{float(row['stage3_one_body_pka']):.2f}/"
                f"{row['stage3_contextual_edge_pka_at_dominant_background']}",
            ),
        ]
        mols, legends, highlights, colors = [], [], [], []
        for smiles, legend in specs:
            mol, atom_indices, atom_colors = _drawing(
                smiles, row["site_atom_maps_json"], row["site_center_maps_json"]
            )
            if mol is None:
                continue
            mols.append(mol)
            legends.append(legend)
            highlights.append(atom_indices)
            colors.append(atom_colors)
        if not mols:
            continue
        image = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(560, 470), legends=legends,
            highlightAtomLists=highlights, highlightAtomColors=colors, useSVG=False,
        )
        image.save(str(Path(image_dir) / f"{row['review_id']}.png"))


def _save_decision(path: str, row: pd.Series, action: str, note: str, replacement: str, mode: str) -> None:
    if action not in ALL_DECISIONS:
        raise ValueError(f"Unsupported action: {action}")
    if action == "replace_input_smiles":
        replacement = validate_replacement_smiles(str(row["original_smiles"]), replacement)
    else:
        replacement = ""
    decisions = _load_decisions(path)
    decisions = decisions[decisions["review_id"].astype(str) != str(row["review_id"])].copy()
    payload = {
        "review_id": row["review_id"],
        "source_file": row["source_file"],
        "record_index": int(row["record_index"]),
        "pka_value": float(row["pka_value"]),
        "molecule_id": row["molecule_id"],
        "site_id": row["site_id"],
        "original_smiles": row["original_smiles"],
        "review_decision": action,
        "replacement_smiles": replacement,
        "replacement_transition_mode": mode if action == "correct_transition_definition" else "",
        "review_note": note,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    decisions = pd.concat([decisions, pd.DataFrame([payload])], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(path, index=False)


def _page(queue: pd.DataFrame, row: Optional[pd.Series], status: Dict[str, str], error: str = "") -> str:
    total = len(queue)
    terminal = sum(status.get(str(value), "") in TERMINAL_DECISIONS for value in queue["review_id"])
    deferred = sum(status.get(str(value), "") == "defer" for value in queue["review_id"])
    if row is None:
        body = (
            "<h1>pKa–structure review complete</h1>"
            f"<p>{terminal}/{total} terminal decisions; {deferred} deferred.</p>"
        )
    else:
        rid = html.escape(str(row["review_id"]))
        warning = html.escape(str(row["review_warning"]))
        body = f"""
        <h1>{rid}</h1>
        <p><b>{html.escape(str(row['site_label']))}</b> · experimental pKa
        <b>{float(row['pka_value']):.2f}</b> · deployed macroscopic pKa
        <b>{float(row['stage2_predicted_macro_pka']):.2f}</b> · Stage 3 one-body coefficient
        <b>{float(row['stage3_one_body_pka']):.2f}</b></p>
        <img src="/image/{quote(str(row['review_id']))}.png" alt="Structure comparison">
        <div class="warning"><b>Transition warning:</b> {warning}</div>
        <dl>
          <dt>Original SMILES</dt><dd class="mono">{html.escape(str(row['original_smiles']))}</dd>
          <dt>Current-pair HH-form SMILES</dt><dd class="mono">{html.escape(str(row['proposed_pka_consistent_smiles']))}</dd>
          <dt>Source</dt><dd>{html.escape(str(row['source_file']))}#{int(row['record_index'])}</dd>
          <dt>ChEMBL molecule / assay / document</dt><dd>{html.escape(str(row['source_molecule_id']))} / {html.escape(str(row['source_assay_id']))} / {html.escape(str(row['source_document_id']))}</dd>
          <dt>Source site evidence</dt><dd>{html.escape(str(row['source_site_evidence']))} ({html.escape(str(row['source_site_evidence_confidence']))})</dd>
          <dt>Current mapping basis</dt><dd>{html.escape(str(row['current_network_mapping_basis']))}</dd>
          <dt>pKa semantics</dt><dd>Experimental pKa is compared with the deployed macroscopic pKa. The one-body coefficient and context-specific edge pKa are model terms, not interchangeable experimental observables.</dd>
        </dl>
        <form method="post">
          <input type="hidden" name="review_id" value="{rid}">
          <label>Review note<input name="note" value="{html.escape(str(row.get('review_note', '')))}"></label>
          <div class="actions">
            <button name="action" value="keep_current_structure_and_transition" class="keep">Keep structure + current transition</button>
            <button name="action" value="exclude_measurement_structure_pair" class="bad">Discard this structure–pKa pair</button>
            <button name="action" value="exclude_entire_source_record" class="bad">Discard all pKas for this source record</button>
          </div>
          <label>Replacement SMILES<input class="mono" name="replacement_smiles" value="{html.escape(str(row['proposed_pka_consistent_smiles']))}"></label>
          <button name="action" value="replace_input_smiles" class="replace">Use replacement SMILES</button>
          <label>Correct transition definition
            <select name="transition_mode">
              <option value="acidic_deprotonation">neutral/protonated acid → conjugate base</option>
              <option value="basic_protonation">conjugate acid → neutral/basic form</option>
              <option value="other_site_or_direction">other site or direction; see note</option>
            </select>
          </label>
          <button name="action" value="correct_transition_definition" class="transition">Keep structure; correct transition/site</button>
          <button name="action" value="defer">Defer</button>
        </form>
        """
    error_html = f"<div class='error'>{html.escape(error)}</div>" if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>pKa–structure audit</title>
    <style>
    body{{font:16px system-ui;margin:0 auto;padding:1.2rem;max-width:1700px;color:#17202a}}
    header{{position:sticky;top:0;background:#17202a;color:white;padding:.8rem;z-index:2}}
    img{{width:100%;border:1px solid #bbb;background:white}}
    dl{{display:grid;grid-template-columns:18rem 1fr;gap:.35rem 1rem}}
    dt{{font-weight:700}} dd{{margin:0}} .mono{{font-family:monospace}}
    label{{display:block;margin:.8rem 0;font-weight:700}} input,select{{width:100%;padding:.55rem;margin-top:.25rem}}
    button{{padding:.7rem 1rem;margin:.35rem;border:0;border-radius:.3rem;cursor:pointer}}
    .keep{{background:#247a3c;color:white}} .bad{{background:#9b2c2c;color:white}}
    .replace{{background:#2266aa;color:white}} .transition{{background:#6b46a3;color:white}}
    .warning{{background:#fff4ce;border-left:6px solid #d69e00;padding:.8rem;margin:.8rem 0}}
    .error{{background:#ffd8d8;color:#7a1111;padding:.8rem}}
    </style></head><body><header>{terminal}/{total} decided · {deferred} deferred</header>
    {error_html}{body}</body></html>"""


def serve_web_reviewer(queue: pd.DataFrame, decisions_path: str, image_dir: str, host: str, port: int) -> None:
    rows = {str(row["review_id"]): row for _, row in queue.iterrows()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"review-web: {fmt % args}")

        def _status(self) -> Dict[str, str]:
            return {
                str(row["review_id"]): str(row["review_decision"])
                for _, row in _load_decisions(decisions_path).iterrows()
            }

        def _next(self, requested: str = "") -> Optional[pd.Series]:
            status = self._status()
            if requested in rows:
                return rows[requested]
            for review_id, candidate in rows.items():
                # A defer is non-terminal in the durable audit, but it must
                # advance the current click-through pass instead of trapping
                # the reviewer on the same record.
                if status.get(review_id, "") not in ALL_DECISIONS:
                    return candidate
            return None

        def _send(self, content: str, code: int = 200, mime: str = "text/html; charset=utf-8") -> None:
            raw = content.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.startswith("/image/"):
                review_id = os.path.basename(parsed.path)[:-4]
                path = Path(image_dir) / f"{review_id}.png"
                if not path.exists():
                    self._send("missing", 404, "text/plain")
                    return
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            requested = parse_qs(parsed.query).get("review_id", [""])[0]
            self._send(_page(queue, self._next(requested), self._status()))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            review_id = form.get("review_id", [""])[0]
            row = rows.get(review_id)
            if row is None:
                self._send("unknown review", 400, "text/plain")
                return
            try:
                _save_decision(
                    decisions_path,
                    row,
                    form.get("action", [""])[0],
                    form.get("note", [""])[0],
                    form.get("replacement_smiles", [""])[0],
                    form.get("transition_mode", [""])[0],
                )
            except ValueError as exc:
                self._send(_page(queue, row, self._status(), str(exc)), 400)
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    print(f"pKa–structure reviewer: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--networks", default=DEFAULT_NETWORKS)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    queue = build_review_queue(
        comparison_path=args.comparison,
        networks_path=args.networks,
        raw_dir=args.raw_dir,
        decisions_path=args.decisions,
    )
    Path(args.queue).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.queue, index=False)
    if not os.path.exists(args.decisions):
        Path(args.decisions).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=DECISION_COLUMNS).to_csv(args.decisions, index=False)
    render_review_images(queue, args.image_dir)
    print(f"review_rows={len(queue)}")
    print(f"queue={args.queue}")
    print(f"decisions={args.decisions}")
    if not args.build_only:
        serve_web_reviewer(queue, args.decisions, args.image_dir, args.host, args.port)


if __name__ == "__main__":
    main()
