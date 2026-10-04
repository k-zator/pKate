#!/usr/bin/env python3
"""Local guided form for collecting full-range pKa transition requirements.

The user is not expected to know microscopic pKas or thermodynamic metadata.
"I don't know; research this" is a complete, useful response.  Answers are
autosaved to a separate CSV and every transition remains disabled until a
later chemistry/provenance validation step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_TEMPLATE = "data/curation/full_pka_range_transition_inputs.csv"
DEFAULT_RESPONSES = "data/curation/full_pka_range_information_responses.csv"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8771

BASE_COLUMNS = [
    "transition_id", "site_family", "acid_form_label", "base_form_label",
    "acid_charge_relative_to_neutral", "base_charge_relative_to_neutral",
    "measurement_kind", "reference_pka", "reference_uncertainty", "solvent",
    "temperature_K", "ionic_strength_M", "allowed_protonation_atom_rule",
    "tautomer_or_microstate_populations", "source_citation", "enable", "notes",
]
RESPONSE_COLUMNS = [
    "support_decision", "pka_input_mode", "reference_pka_low",
    "reference_pka_high", "conditions_status", "applicability_domain",
    "counterion_policy", "research_requested", "user_notes",
    "submission_status", "activation_readiness", "missing_for_activation",
    "updated_at",
]
OUTPUT_COLUMNS = BASE_COLUMNS + RESPONSE_COLUMNS

MUTABLE_COLUMNS = {
    "support_decision", "pka_input_mode", "reference_pka",
    "reference_pka_low", "reference_pka_high", "reference_uncertainty",
    "conditions_status", "solvent", "temperature_K", "ionic_strength_M",
    "measurement_kind", "tautomer_or_microstate_populations",
    "source_citation", "applicability_domain", "counterion_policy",
    "research_requested", "user_notes", "submission_status",
}

GUIDANCE: Dict[str, Dict[str, str]] = {
    "123triazole_cation_to_neutral": {
        "priority": "high",
        "question": "Should the model handle protonated 1,2,3-triazoles in strongly acidic solution?",
        "why": "This is proton addition to a ring N. It is separate from losing the existing N-H proton.",
    },
    "123triazole_neutral_to_anion": {
        "priority": "high",
        "question": "Should the model handle deprotonated 1,2,3-triazoles?",
        "why": "This removes the neutral ring N-H proton and produces an anionic tautomer ensemble.",
    },
    "124triazole_cation_to_neutral": {
        "priority": "high",
        "question": "Should the model handle protonated 1,2,4-triazoles in acidic solution?",
        "why": "The protonated-to-neutral transition needs its own pKaH evidence.",
    },
    "124triazole_neutral_to_anion": {
        "priority": "high",
        "question": "Should the model handle deprotonated 1,2,4-triazoles?",
        "why": "Neutral-to-anion acidity is a second transition, not the inverse of pKaH data.",
    },
    "n_substituted_tetrazole_cation_to_neutral": {
        "priority": "high",
        "question": "Do N-substituted tetrazoles need acidic-solution protonation coverage?",
        "why": "N-substitution removes the usual tetrazole N-H acidity but leaves possible ring-N basicity.",
    },
    "sulfonamide_neutral_to_anion": {
        "priority": "high",
        "question": "Should sulfonamide N-H deprotonation be part of the normal model?",
        "why": "Many medicinal sulfonamides can ionize in or near the aqueous range, but substitution changes pKa substantially.",
    },
    "sulfamide_neutral_to_anion": {
        "priority": "medium",
        "question": "Should sulfamide N-H deprotonation be supported?",
        "why": "A molecule may contain two non-equivalent N-H sites and sequential pKas.",
    },
    "sulfamate_neutral_to_anion": {
        "priority": "medium",
        "question": "Should sulfamate N-H deprotonation be supported?",
        "why": "N-H and O-H loss must be identified separately; one pKa cannot safely stand for both.",
    },
    "oxadiazole_cation_to_neutral": {
        "priority": "medium",
        "question": "Do you need protonated oxadiazoles in the requested pH range?",
        "why": "These rings are recognized, but their weak basicity is isomer-dependent.",
    },
    "thiadiazole_cation_to_neutral": {
        "priority": "medium",
        "question": "Do you need protonated thiadiazoles in the requested pH range?",
        "why": "Each ring isomer needs a defensible pKaH domain rather than a shared guess.",
    },
    "indazole_cation_to_neutral": {
        "priority": "high",
        "question": "Should indazole protonation to a cation be supported?",
        "why": "Indazole is amphoteric; this is its basic transition.",
    },
    "indazole_neutral_to_anion": {
        "priority": "high",
        "question": "Should indazole N-H deprotonation to an anion be supported?",
        "why": "This is the acidic transition and needs evidence separate from indazole pKaH.",
    },
    "indole_neutral_to_anion": {
        "priority": "low",
        "question": "Do you need indole N-H deprotonation at very high pH?",
        "why": "This is usually beyond ordinary aqueous/biological use.",
    },
    "amide_neutral_to_anion": {
        "priority": "low",
        "question": "Do you need ordinary amide N-H deprotonation?",
        "why": "This is normally an extreme-range transition, not ordinary amide protonation near ambient pH.",
    },
    "ketone_neutral_to_enolate": {
        "priority": "low",
        "question": "Should the model cover removal of an alpha-C-H proton to form enolates?",
        "why": "This is carbon acidity and can create several regioisomeric/enolate states.",
    },
    "protonated_ketone_to_neutral": {
        "priority": "low",
        "question": "Should protonated carbonyl oxygen be represented in strong acid?",
        "why": "O-protonation is a different transition from alpha-carbon deprotonation.",
    },
    "nitrile_cation_to_neutral": {
        "priority": "low",
        "question": "Should nitrile N-protonation in strong acid be supported?",
        "why": "This is normally outside the current aqueous pKa 0–14 scope.",
    },
}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: _text(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _default_response(row: Mapping[str, str]) -> Dict[str, str]:
    result = {column: _text(row.get(column, "")) for column in OUTPUT_COLUMNS}
    result.update({
        "enable": "false",
        "measurement_kind": "unknown",
        "support_decision": "unsure",
        "pka_input_mode": "unknown",
        "conditions_status": "unknown",
        "research_requested": "false",
        "submission_status": "draft",
        "activation_readiness": "awaiting_user_scope",
        "missing_for_activation": "support decision",
    })
    return result


def _enabled(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "yes"}


def _overlay_validated_template(
    row: Dict[str, str], base: Mapping[str, str]
) -> Dict[str, str]:
    if not _enabled(base.get("enable")):
        row["enable"] = "false"
        return row
    for column in (
        "measurement_kind", "reference_pka", "reference_uncertainty",
        "solvent", "temperature_K", "ionic_strength_M",
        "tautomer_or_microstate_populations", "source_citation",
    ):
        row[column] = _text(base.get(column))
    row["enable"] = "true"
    row["activation_readiness"] = "validated_and_enabled"
    row["missing_for_activation"] = ""
    row["research_requested"] = "false"
    if row["solvent"].lower() == "water" and row["temperature_K"]:
        row["conditions_status"] = "aqueous_ambient"
    return row


def load_rows(template_path: str, responses_path: str) -> List[Dict[str, str]]:
    template = _read_csv(template_path)
    if not template:
        raise ValueError(f"No transition rows found in {template_path}")
    missing = set(BASE_COLUMNS) - set(template[0])
    if missing:
        raise ValueError(f"Template is missing columns: {sorted(missing)}")
    prior = {row.get("transition_id", ""): row for row in _read_csv(responses_path)}
    result = []
    for base in template:
        transition_id = base["transition_id"]
        row = _default_response(base)
        if transition_id in prior:
            for column in OUTPUT_COLUMNS:
                if column in prior[transition_id]:
                    row[column] = prior[transition_id][column]
            # Structural identity always comes from the versioned template.
            for column in BASE_COLUMNS:
                if column not in MUTABLE_COLUMNS:
                    row[column] = base[column]
        row = _overlay_validated_template(row, base)
        guide = GUIDANCE.get(transition_id, {})
        row["display_priority"] = guide.get("priority", "medium")
        row["plain_question"] = guide.get(
            "question", f"Should the model support {transition_id.replace('_', ' ')}?"
        )
        row["plain_why"] = guide.get("why", _text(base.get("notes")))
        result.append(row)
    return result


def _number(value: object) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Not a number: {text}") from exc


def normalize_answer(base: Mapping[str, str], answer: Mapping[str, object]) -> Dict[str, str]:
    row = {column: _text(base.get(column, "")) for column in OUTPUT_COLUMNS}
    for column in MUTABLE_COLUMNS:
        if column in answer:
            row[column] = _text(answer[column])

    support = row["support_decision"] or "unsure"
    if support not in {"want_support", "not_needed", "unsure"}:
        raise ValueError("Invalid support decision")
    mode = row["pka_input_mode"] or "unknown"
    if mode not in {"unknown", "exact", "range"}:
        raise ValueError("Invalid pKa input mode")
    if row["measurement_kind"] not in {"", "unknown", "macro", "micro"}:
        raise ValueError("Measurement kind must be macro, micro, or unknown")

    if mode == "unknown":
        row["reference_pka"] = ""
        row["reference_pka_low"] = ""
        row["reference_pka_high"] = ""
        row["reference_uncertainty"] = ""
    elif mode == "exact":
        if _number(row["reference_pka"]) is None:
            raise ValueError("Enter the pKa number, or choose ‘I don't know’")
        row["reference_pka_low"] = ""
        row["reference_pka_high"] = ""
        _number(row["reference_uncertainty"])
    else:
        low = _number(row["reference_pka_low"])
        high = _number(row["reference_pka_high"])
        if low is None or high is None or low > high:
            raise ValueError("Enter a valid pKa range with low ≤ high")
        row["reference_pka"] = f"{(low + high) / 2.0:g}"
        row["reference_uncertainty"] = f"{(high - low) / 2.0:g}"

    if row["conditions_status"] == "aqueous_ambient":
        row["solvent"] = row["solvent"] or "water"
        row["temperature_K"] = row["temperature_K"] or "298.15"
    _number(row["temperature_K"])
    _number(row["ionic_strength_M"])

    row["research_requested"] = (
        "true" if _text(answer.get("research_requested", row["research_requested"])).lower()
        in {"1", "true", "yes", "on"} else "false"
    )
    missing: List[str] = []
    if support != "want_support":
        readiness = "not_requested" if support == "not_needed" else "awaiting_user_scope"
        if support == "unsure":
            missing.append("support decision")
    else:
        if mode == "unknown":
            missing.append("pKa value or research")
        if not row["solvent"]:
            missing.append("solvent")
        if not row["temperature_K"]:
            missing.append("temperature")
        if row["measurement_kind"] in {"", "unknown"}:
            missing.append("macro/micro identity")
        if not row["source_citation"]:
            missing.append("source")
        readiness = "ready_for_codex_validation" if not missing else "needs_research_or_metadata"
    if row["research_requested"] == "true" and missing:
        readiness = "research_requested"

    # Human submission is evidence collection, not chemistry approval.
    row["enable"] = "false"
    row["activation_readiness"] = readiness
    row["missing_for_activation"] = "; ".join(missing)
    row["updated_at"] = datetime.now(timezone.utc).isoformat()
    return row


def write_rows(path: str, rows: Sequence[Mapping[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, output)


def summarize(rows: Iterable[Mapping[str, str]]) -> Dict[str, int]:
    rows = list(rows)
    return {
        "total": len(rows),
        "submitted": sum(row.get("submission_status") == "submitted" for row in rows),
        "want_support": sum(row.get("support_decision") == "want_support" for row in rows),
        "not_needed": sum(row.get("support_decision") == "not_needed" for row in rows),
        "unsure": sum(row.get("support_decision") == "unsure" for row in rows),
        "research_requested": sum(row.get("research_requested") == "true" for row in rows),
        "ready_for_validation": sum(
            row.get("activation_readiness") == "ready_for_codex_validation" for row in rows
        ),
        "validated_and_enabled": sum(
            row.get("activation_readiness") == "validated_and_enabled" for row in rows
        ),
    }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pKa information feeder</title>
<style>
:root{--bg:#f5f1e8;--ink:#20231f;--card:#fffdf8;--accent:#176b5b;--soft:#dcebe5;--warn:#9b5d16;--line:#d7d1c4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,sans-serif}
main{max-width:920px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:18px;align-items:start}
h1{font-size:1.65rem;margin:.2rem 0}.lede{max-width:680px;color:#4d514b}.progress{white-space:nowrap;background:var(--soft);padding:10px 14px;border-radius:999px}
.toolbar,.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 7px 24px #554b3520}
.tag{display:inline-block;border-radius:999px;padding:3px 9px;font-size:.78rem;text-transform:uppercase;background:#eee}.high{background:#f6d6cc}.medium{background:#f4e5bd}.low{background:#dcebe5}
h2{font-size:1.45rem;margin:.7rem 0}.why{padding:12px 14px;background:#f0eee7;border-left:4px solid var(--accent);border-radius:7px}.reaction{font-family:ui-monospace,monospace;background:#252a26;color:#f7f5ee;padding:12px;border-radius:9px;overflow:auto}
fieldset{border:0;border-top:1px solid var(--line);margin:20px 0 0;padding:18px 0 0}legend{font-weight:700;padding-right:10px}.choice{display:block;padding:8px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}label span{display:block;font-size:.86rem;color:#5b6059;margin-bottom:4px}
input,select,textarea,button{font:inherit}input,select,textarea{width:100%;padding:10px;border:1px solid #aaa398;border-radius:8px;background:white}textarea{min-height:80px}button{border:0;border-radius:9px;padding:10px 15px;cursor:pointer}.primary{background:var(--accent);color:white}.secondary{background:#e8e3d8}.danger{color:var(--warn)}
.hint{font-size:.88rem;color:#666}.hidden{display:none}.status{min-height:1.5em;color:var(--accent);font-weight:600}.missing{color:var(--warn)}details{margin-top:16px}code{font-size:.88rem}.spacer{flex:1}
@media(max-width:650px){.top{display:block}.progress{display:inline-block}.grid{grid-template-columns:1fr}.card{padding:17px}}
</style></head><body><main>
<div class="top"><div><h1>Full-range pKa information feeder</h1><div class="lede">You only need to say what you want and what you know. <b>“I don’t know—research this” is a valid completed answer.</b> Nothing entered here is automatically activated for training.</div></div><div class="progress" id="progress">Loading…</div></div>
<div class="toolbar"><label><span>Show</span><select id="filter"><option value="all">All transitions</option><option value="high">High priority</option><option value="unsubmitted">Still unanswered</option><option value="research">Research requested</option></select></label><span class="spacer"></span><button class="secondary" id="download">Download responses CSV</button></div>
<section class="card" id="card"><div id="loading">Loading questions…</div><div id="form" class="hidden">
<span class="tag" id="priority"></span><span class="tag" id="submitTag"></span><h2 id="question"></h2><div class="why" id="why"></div><p class="reaction" id="reaction"></p>
<fieldset><legend>1. Do you want this chemistry supported?</legend>
<label class="choice"><input type="radio" name="support" value="want_support"> Yes / probably useful</label>
<label class="choice"><input type="radio" name="support" value="not_needed"> No, outside my intended use</label>
<label class="choice"><input type="radio" name="support" value="unsure"> I’m not sure</label></fieldset>
<fieldset><legend>2. What do you know about its pKa?</legend><div class="grid"><label><span>Answer type</span><select id="pkaMode"><option value="unknown">I don’t know</option><option value="exact">A value / rough estimate</option><option value="range">Only a plausible range</option></select></label><label id="exactBox"><span>pKa value</span><input id="pka" inputmode="decimal" placeholder="e.g. 6.8"></label><label id="uncertaintyBox"><span>Uncertainty, if known (± pKa)</span><input id="uncertainty" inputmode="decimal" placeholder="optional"></label><label id="lowBox"><span>Lowest plausible pKa</span><input id="pkaLow" inputmode="decimal"></label><label id="highBox"><span>Highest plausible pKa</span><input id="pkaHigh" inputmode="decimal"></label></div>
<label class="choice"><input type="checkbox" id="research"> I don’t know enough—ask Codex to research/verify this transition</label><div class="hint">A remembered textbook number is still useful. Put that in and say “from memory” below; it will remain unvalidated.</div></fieldset>
<fieldset><legend>3. Where did the value apply?</legend><div class="grid"><label><span>Conditions</span><select id="conditions"><option value="unknown">Unknown</option><option value="aqueous_ambient">Water, around room temperature</option><option value="custom">Other / known conditions</option></select></label><label><span>Macro or microscopic?</span><select id="kind"><option value="unknown">I don’t know</option><option value="macro">Macroscopic / molecule-level</option><option value="micro">Microscopic / atom-specific</option></select></label><label><span>Solvent</span><input id="solvent" placeholder="water, DMSO, mixture…"></label><label><span>Temperature (K)</span><input id="temperature" inputmode="decimal" placeholder="298.15"></label><label><span>Ionic strength (M)</span><input id="ionic" inputmode="decimal" placeholder="optional"></label><label><span>Source or where you remember it from</span><input id="source" placeholder="DOI, URL, book, or ‘memory’"></label></div></fieldset>
<details><summary>Optional advanced information</summary><div class="grid"><label><span>Which molecules/substituents does it apply to?</span><textarea id="domain"></textarea></label><label><span>Known tautomer populations or relative energies</span><textarea id="tautomers"></textarea></label><label><span>Salt, counterion, or metal-coordination policy</span><textarea id="counterion"></textarea></label><label><span>Anything else you want me to know</span><textarea id="notes"></textarea></label></div></details>
<p id="missing" class="missing"></p><p id="status" class="status"></p>
<div class="nav"><button class="secondary" id="prev">← Previous</button><button class="secondary" id="save">Save draft</button><span class="spacer"></span><button class="primary" id="submit">Submit answer & next →</button></div>
</div></section></main>
<script>
let all=[], visible=[], index=0, timer=null;
const $=id=>document.getElementById(id), val=id=>$(id).value, set=(id,v)=>$(id).value=v||'';
async function init(){const x=await fetch('/api/state').then(r=>r.json());all=x.rows; applyFilter(); $('loading').classList.add('hidden');$('form').classList.remove('hidden');}
function applyFilter(){const f=val('filter');visible=all.filter(r=>f==='all'||r.display_priority===f||(f==='unsubmitted'&&r.submission_status!=='submitted')||(f==='research'&&r.research_requested==='true'));if(!visible.length){visible=all}index=Math.min(index,visible.length-1);render();}
function current(){return visible[index]}
function render(){if(!visible.length)return;const r=current();localStorage.setItem('pka-feeder-id',r.transition_id);$('priority').textContent=r.display_priority+' priority';$('priority').className='tag '+r.display_priority;$('submitTag').textContent=r.submission_status||'draft';$('question').textContent=r.plain_question;$('why').textContent=r.plain_why;$('reaction').textContent=`${r.acid_form_label} (${r.acid_charge_relative_to_neutral})  ⇌  ${r.base_form_label} (${r.base_charge_relative_to_neutral}) + H⁺`;
document.querySelectorAll('[name=support]').forEach(x=>x.checked=x.value===r.support_decision);set('pkaMode',r.pka_input_mode);set('pka',r.reference_pka);set('uncertainty',r.reference_uncertainty);set('pkaLow',r.reference_pka_low);set('pkaHigh',r.reference_pka_high);$('research').checked=r.research_requested==='true';set('conditions',r.conditions_status);set('kind',r.measurement_kind||'unknown');set('solvent',r.solvent);set('temperature',r.temperature_K);set('ionic',r.ionic_strength_M);set('source',r.source_citation);set('domain',r.applicability_domain);set('tautomers',r.tautomer_or_microstate_populations);set('counterion',r.counterion_policy);set('notes',r.user_notes);$('missing').textContent=r.missing_for_activation?'Still needed before activation: '+r.missing_for_activation:'';$('status').textContent='';togglePka();updateProgress();}
function togglePka(){const m=val('pkaMode');$('exactBox').classList.toggle('hidden',m!=='exact');$('uncertaintyBox').classList.toggle('hidden',m!=='exact');$('lowBox').classList.toggle('hidden',m!=='range');$('highBox').classList.toggle('hidden',m!=='range');if(m==='unknown')$('research').checked=true;}
function payload(submit=false){return {transition_id:current().transition_id,support_decision:(document.querySelector('[name=support]:checked')||{}).value||'unsure',pka_input_mode:val('pkaMode'),reference_pka:val('pka'),reference_uncertainty:val('uncertainty'),reference_pka_low:val('pkaLow'),reference_pka_high:val('pkaHigh'),research_requested:$('research').checked,conditions_status:val('conditions'),measurement_kind:val('kind'),solvent:val('solvent'),temperature_K:val('temperature'),ionic_strength_M:val('ionic'),source_citation:val('source'),applicability_domain:val('domain'),tautomer_or_microstate_populations:val('tautomers'),counterion_policy:val('counterion'),user_notes:val('notes'),submission_status:submit?'submitted':current().submission_status};}
async function save(submit=false,quiet=false){clearTimeout(timer);const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload(submit))});const x=await res.json();if(!res.ok){$('status').textContent=x.error;throw Error(x.error)}const i=all.findIndex(r=>r.transition_id===x.row.transition_id);all[i]=x.row;const vi=visible.findIndex(r=>r.transition_id===x.row.transition_id);if(vi>=0)visible[vi]=x.row;if(!quiet)$('status').textContent=submit?'Answer submitted and safely stored.':'Draft autosaved.';updateProgress();return x;}
function autosave(){clearTimeout(timer);timer=setTimeout(()=>save(false,true).catch(()=>{}),650)}
function move(n){index=(index+n+visible.length)%visible.length;render()}
function updateProgress(){const n=all.filter(r=>r.submission_status==='submitted').length;$('progress').textContent=`${n} / ${all.length} answered`}
$('filter').addEventListener('change',applyFilter);$('pkaMode').addEventListener('change',()=>{togglePka();autosave()});document.querySelectorAll('input,select,textarea').forEach(x=>{if(x.id!=='filter'&&x.id!=='pkaMode')x.addEventListener('change',autosave)});$('save').onclick=()=>save(false);$('submit').onclick=async()=>{await save(true);move(1)};$('prev').onclick=()=>move(-1);$('download').onclick=()=>window.location='/api/download.csv';init().catch(e=>$('loading').textContent='Could not load: '+e.message);
</script></body></html>"""


class FeederState:
    def __init__(self, template_path: str, responses_path: str):
        self.template_path = template_path
        self.responses_path = responses_path
        self.lock = threading.Lock()
        self.template_rows = {
            row["transition_id"]: row for row in _read_csv(template_path)
        }
        self.rows = load_rows(template_path, responses_path)
        if not os.path.exists(responses_path):
            write_rows(responses_path, self.rows)

    def public_rows(self) -> List[Dict[str, str]]:
        return [dict(row) for row in self.rows]

    def save(self, payload: Mapping[str, object]) -> Dict[str, str]:
        transition_id = _text(payload.get("transition_id"))
        with self.lock:
            index = next(
                (i for i, row in enumerate(self.rows) if row["transition_id"] == transition_id),
                None,
            )
            if index is None:
                raise ValueError(f"Unknown transition_id: {transition_id}")
            display = {
                key: self.rows[index].get(key, "")
                for key in ("display_priority", "plain_question", "plain_why")
            }
            updated = normalize_answer(self.rows[index], payload)
            updated = _overlay_validated_template(
                updated, self.template_rows.get(transition_id, {})
            )
            updated.update(display)
            self.rows[index] = updated
            write_rows(self.responses_path, self.rows)
            return dict(updated)


def make_handler(state: FeederState):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            self._send(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                rows = state.public_rows()
                self._json(200, {"rows": rows, "summary": summarize(rows)})
            elif path == "/api/download.csv":
                body = Path(state.responses_path).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{Path(state.responses_path).name}"',
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected one answer object")
                row = state.save(payload)
                self._json(200, {"row": row, "summary": summarize(state.public_rows())})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guided full-range pKa information feeder.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--responses", default=DEFAULT_RESPONSES)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = FeederState(args.template, args.responses)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"pKa information feeder: {url}")
    print(f"autosave_file={args.responses}")
    print("Press Ctrl-C to stop; submitted answers are already saved.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
