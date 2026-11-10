"""Versioned, offline-testable controls embedded in Person1_Full_Run_v2.ipynb.

These checks establish parsing/structural properties, not mathematical correctness.
"""
import ast
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

CONTROL_VERSION = 'person1-v3.2'
PROMPT_SPEC_VERSION = 'math-worked-solution-v2'
PROMPT_TEMPLATE = (
    'Question: {problem}\n'
    'Solve the problem step by step. End with: Final answer: <answer>.\n'
    'Solution:\n'
)
NUMBER_PATTERN = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+))?%?'
NUMBER_RE = re.compile(NUMBER_PATTERN)
SUSPECT_ANSWER_RE = re.compile(r'\(,|,\)|,,|,=|^,|,$|^=|\(\)')
CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def clean_text(value):
    if value is None or type(value).__name__ in {'NAType', 'NaTType'}:
        return ''
    if not isinstance(value, str):
        try:
            if math.isnan(value): return ''
        except (TypeError, ValueError): pass
    # NFC preserves mathematical superscripts and other compatibility characters.
    text = unicodedata.normalize('NFC', str(value)).replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'\n{3,}', '\n\n', re.sub(r'[ \t]+', ' ', text)).strip()


def raw_text(value):
    """Preserve original string bytes; represent missing scalar text as empty."""
    return value if isinstance(value, str) else clean_text(value)


def compact_answer(value):
    # Never silently truncate an answer (v1 truncated at 500 characters).
    text = clean_text(value)
    return text[:-1].rstrip() if text.endswith('.') else text


def normalize_problem_for_matching(value):
    return re.sub(r'\s+', ' ', clean_text(value).casefold()).strip()


def sha256_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def squash_answer(value):
    return re.sub(r'\s+|\\left|\\right|\$|\\!|\\,|\\;|\\displaystyle', '', compact_answer(value).casefold())


def scalar_fraction(value):
    """Safe exact arithmetic only; no eval, symbolic parsing or approximate equality."""
    text = squash_answer(value)
    text = re.sub(r'^([a-z])=', '', text)
    text = re.sub(r'\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', r'(\1/\2)', text)
    # Only thousands separators are numeric commas; (1,2) is not scalar 12.
    comma_checked = re.sub(r'(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)', lambda match: match.group().replace(',', ''), text)
    if ',' in comma_checked: return None
    text = comma_checked
    percent = text.endswith('%')
    if percent:
        text = text[:-1]
    if not text or len(text) > 100 or not re.fullmatch(r'[\d.+*/() -]+', text):
        return None
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand)
            return result if isinstance(node.op, ast.UAdd) else -result
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('unsupported scalar')
    try:
        value = visit(ast.parse(text, mode='eval').body)
        return value / 100 if percent else value
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return None


def normalize_number(value):
    fraction = scalar_fraction(value)
    if fraction is None:
        return value
    return str(fraction.numerator) if fraction.denominator == 1 else f'{fraction.numerator}/{fraction.denominator}'


def answer_key(value):
    fraction = scalar_fraction(value)
    return 'rational:' + str(fraction) if fraction is not None else 'text:' + squash_answer(value)


def answers_equivalent(left, right, shared_checker=None):
    """True/False for exact rational scalars; None means equivalence is unproven."""
    if squash_answer(left) == squash_answer(right):
        return True
    a, b = scalar_fraction(left), scalar_fraction(right)
    if a is not None and b is not None:
        return a == b
    if shared_checker is not None:
        result = shared_checker(left, right)
        if result not in (True, False, None):
            raise ValueError('Shared checker must return True, False or None')
        return result
    return None


def suspected_multi_part(problem):
    text = clean_text(problem).casefold()
    # Conservative scope: multi-part examples await a reviewed full answer tuple.
    return bool(re.search(
        r'\b(?:both|respectively)\b|\b(?:radius|diameter|length|width|area|perimeter|volume)'
        r'\s+(?:and|,)\s+(?:the\s+)?(?:radius|diameter|length|width|area|perimeter|volume)\b'
        r'|\bhow much did each\b|\b(?:find|calculate|determine|what are)\b[^?\n]*\band\b'
        r'|(?:\(a\)|\(b\)|\bi\)|\bii\))', text))


def boxed_spans(solution):
    spans = []
    for marker in re.finditer(r'\\(?:boxed|fbox)\s*\{', solution):
        brace, depth = marker.end() - 1, 0
        for pos in range(brace, len(solution)):
            if solution[pos] == '{': depth += 1
            elif solution[pos] == '}':
                depth -= 1
                if depth == 0:
                    spans.append((marker.start(), pos + 1, compact_answer(solution[brace + 1:pos])))
                    break
    return spans


def last_boxed(solution):
    spans = boxed_spans(solution or '')
    return spans[-1][2] if spans else ''


FINAL_MARKER_RE = re.compile(r'(?im)^[ \t]*final answer[ \t]*[:=]')
TERMINAL_MARKER_RE = re.compile(r'(?im)^[ \t]*(final answer[ \t]*[:=]|####)[ \t]*([^\n]+?)[ \t]*(?:\n[ \t]*)*\Z')
MISSING_ANSWER_VALUES = {'', 'proof', 'notfound', 'nan', 'none', 'null', '<na>', 'nat', 'n/a'}


def answer_type(value):
    """Bounded shape validation; supported symbolic syntax is not a correctness proof."""
    text = compact_answer(value)
    if text.casefold() in MISSING_ANSWER_VALUES: return 'missing'
    if len(text) > 2000: return 'unsupported_oversized'
    if re.search(r'\b(?:or|and|respectively)\b|\\pm|±', text, re.I): return 'unsupported_multi_answer'
    without_numeric_commas = re.sub(r'\d{1,3}(?:,\d{3})+', '', text)
    if ',' in without_numeric_commas or ';' in text: return 'unsupported_multi_answer'
    if scalar_fraction(text) is not None: return 'numeric_scalar'
    # An equality chain must be reduced/reviewed explicitly, never accepted as its first number.
    if '=' in text and not re.fullmatch(r'[a-zA-Z]\s*=\s*[^=]+', text):
        return 'unsupported_equality_chain'
    if re.fullmatch(r'\$?' + NUMBER_PATTERN + r'\s+[A-Za-z][A-Za-z²³^ /-]*', text) and not re.search(
            r'\b(?:times|plus|minus|divided|pi|sqrt)\b', text, re.I):
        return 'quantity_scalar'
    core = text.strip('$')
    for left, right in [(r'\(', r'\)'), (r'\[', r'\]')]:
        if core.startswith(left) and core.endswith(right): core = core[len(left):-len(right)]
    stack = []
    for char in core:
        if char in '({[': stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != {')':'(', '}':'{', ']':'['}[char]: return 'unsupported_unbalanced'
    if stack: return 'unsupported_unbalanced'
    commands = re.findall(r'\\([A-Za-z]+)', core)
    allowed = {'frac','dfrac','tfrac','sqrt','pi','alpha','beta','gamma','theta','lambda','sigma','omega',
               'infty','sin','cos','tan','log','ln','exp','left','right','cdot','times','boxed','fbox'}
    if any(command not in allowed for command in commands): return 'unsupported_symbolic_command'
    residue = re.sub(r'\\[A-Za-z]+', '', core)
    if not re.fullmatch(r'[A-Za-z0-9\s.+*/^_(){}\[\]=\-√π]+', residue): return 'unsupported_structure'
    identifiers = re.findall(r'[A-Za-z]+', residue)
    if any(len(name) > 1 and name not in {'sqrt','pi','sin','cos','tan','log','ln','exp','abs'} for name in identifiers):
        return 'unsupported_prose'
    if not commands and not identifiers and not re.search(r'[√π]', core): return 'unsupported_numeric_expression'
    return 'symbolic_scalar'


@dataclass(frozen=True)
class AnswerExtraction:
    candidate: str
    mode: str
    evidence_text: str = ''
    evidence_start: int = -1
    evidence_end: int = -1
    confidence: str = 'none'
    ambiguity_reason: str = ''
    answer_type: str = 'missing'
    evidence_basis: str = 'canonical_solution'

    def to_dict(self): return asdict(self)


@lru_cache(maxsize=256)
def extract_orca_final_answer(problem, solution):
    """Only the terminal conclusion can supply a target. Never scan earlier givens."""
    solution = clean_text(solution)
    if not solution:
        return '', 'missing'
    if suspected_multi_part(problem):
        return '', 'ambiguous_multi_part'
    # Explicit markers must form the trailing block; earlier markers are ignored.
    markers = list(FINAL_MARKER_RE.finditer(solution))
    marker = TERMINAL_MARKER_RE.search(solution)
    if len(markers) > 1: return '', 'ambiguous_multiple_final_markers'
    if markers and marker is None: return '', 'ambiguous_nonterminal_final_marker'
    if marker:
        value = compact_answer(marker.group(2))
        kind = answer_type(value)
        if kind == 'missing':
            return '', 'missing'
        if kind.startswith('unsupported'): return '', 'ambiguous_' + kind
        return value, 'explicit_final_marker' if marker.group(1) != '####' else 'gsm_marker'
    boxes = boxed_spans(solution)
    if boxes:
        # A box inside the derivation is not a final answer; multiple boxes need review.
        if len(boxes) > 1:
            return '', 'ambiguous_multiple_boxes'
        _, end, value = boxes[-1]
        trailing = solution[end:]
        if re.fullmatch(r'[\s.$\\\[\]()!]*', trailing):
            return (value, 'boxed') if not answer_type(value).startswith('unsupported') else ('', 'ambiguous_boxed_structure')
    paragraph = solution.split('\n\n')[-1]
    sentence = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)[-1].strip()
    # Anchored answer clause, e.g. '6 is subtracted ... result would be 6'.
    clause = re.search(
        r'(?i)\b(?:answer|result|number|value|total|cost|price|distance|area|radius|amount)'
        r'(?:\s+\w+){0,5}?\s+(?:is|equals|would be|will be)\s*[:=]?\s*\$?('
        + NUMBER_PATTERN + r')(?:\s+[A-Za-z][A-Za-z²³^ /-]*)?[.!]?$', sentence)
    if clause:
        # Numeric clauses with operator words require explicit review, not unit stripping.
        if re.search(r'\b(?:times|plus|minus|divided|pi|sqrt)\b', sentence[clause.start(1):], re.I):
            return '', 'ambiguous_unmarked_math_expression'
        return normalize_number(clause.group(1)), 'terminal_answer_clause'
    if re.search(r'\\(?:sqrt|frac|dfrac|tfrac|pi)\b|[√π]|[\w)]\s*\^|\b[a-zA-Z]\s*[+\-]\s*\d|\b(?:times|plus|minus|divided|pi|sqrt)\b', sentence, re.I):
        return '', 'ambiguous_unmarked_math_expression'
    numbers = NUMBER_RE.findall(sentence)
    # Repeated numbers and arithmetic expressions remain ambiguous, never select 'novel' ones.
    if len(numbers) == 1:
        # A prose quantity can be a conclusion when the immediately preceding
        # derivation ends in the same scalar. Earlier givens are never evidence.
        # Keep ratios, symbolic expressions, alternatives and uncorroborated
        # single numbers in quarantine. This checks extraction, not all reasoning.
        before = solution[:solution.rfind(sentence)].rstrip()
        if '=' in before and not re.search(r'\b(?:times|plus|minus|divided|pi|sqrt|not|cannot)\b', sentence, re.I):
            rhs = before.rsplit('=', 1)[1].splitlines()[0].strip()
            rhs = rhs.replace(r'\]', '').replace(r'\)', '').strip(' $.')
            rhs = re.sub(r'^Rs\.\s*', '', rhs)
            match = re.fullmatch('(' + NUMBER_PATTERN + r')(?:\s+[A-Za-z][A-Za-z /-]*)?', rhs)
            if match and answers_equivalent(match.group(1), numbers[0]) is True:
                return normalize_number(numbers[0]), 'corroborated_terminal_quantity'
        return normalize_number(numbers[0]), 'terminal_single_number_review'
    return '', 'ambiguous_terminal' if numbers else 'missing'


def extract_answer_metadata(problem, solution):
    text = clean_text(solution)
    answer, mode = extract_orca_final_answer(problem, text)
    marker = TERMINAL_MARKER_RE.search(text)
    boxes = boxed_spans(text)
    if marker:
        start, end = marker.span()
    elif mode == 'boxed' and boxes:
        start, end, _ = boxes[-1]
    else:
        start = text.rfind('\n\n') + 2 if '\n\n' in text else 0
        end = len(text)
    confidence = 'marked_terminal' if mode in {'explicit_final_marker','gsm_marker','boxed'} else (
        'terminal_clause' if mode in {'terminal_answer_clause', 'corroborated_terminal_quantity'} else 'heuristic' if answer else 'none')
    return AnswerExtraction(answer, mode, text[start:end], start, end, confidence,
                            mode.removeprefix('ambiguous_') if mode.startswith('ambiguous') else '', answer_type(answer))


def extraction_consistency_status(extraction, final_answer):
    if not extraction.candidate or extraction.confidence == 'heuristic':
        return 'unresolved_requires_review'
    equivalent = answers_equivalent(extraction.candidate, final_answer)
    return 'consistent_terminal_evidence' if equivalent is True else 'contradictory' if equivalent is False else 'unresolved_requires_review'


def diagram_flags(problem, solution):
    text = f'{problem}\n{solution}'.casefold()
    flags = []
    if re.search(r'\[/?asy\]|\\begin\{(?:asy|tikzpicture)\}|<svg\b', text):
        flags.append('inline_diagram_code')
    if re.search(r'!\[[^\]]*\]\s*(?:\(|\[)|<img\b|\\includegraphics\b'
                 r'|\b(?:figure|diagram)\s+(?:below|above)|shown in the (?:figure|diagram)', text):
        flags.append('missing_or_external_diagram')
    return flags


def has_image_reference(problem, solution):
    return bool(diagram_flags(problem, solution))


def looks_answer_only(solution, final_answer):
    text = clean_text(solution)
    if answers_equivalent(text, final_answer) is True:
        return True
    prose = re.sub(r'\\[a-zA-Z]+|[^a-zA-Z ]', ' ', text)
    words = prose.split()
    return len(words) < 8 and len(text.splitlines()) <= 2 and text.count('=') <= 1


def target_quality_flags(problem, solution, final_answer, answer_status):
    flags = []
    if answer_status.startswith('ambiguous'):
        flags.append('ambiguous_final_answer')
    kind = answer_type(final_answer)
    if kind == 'missing': flags.append('missing_or_sentinel_final_answer')
    elif kind.startswith('unsupported'): flags.append('unsupported_final_answer_structure')
    if answer_status == 'terminal_single_number_review':
        flags.append('heuristic_terminal_answer_requires_individual_review')
    if suspected_multi_part(problem):
        flags.append('suspected_multi_part_requires_review')
    if solution and looks_answer_only(solution, final_answer):
        flags.append('suspected_answer_only_solution')
    # A recognized terminal scalar disagreement is a separate consistency defect.
    terminal, status = extract_orca_final_answer(problem, solution)
    if terminal and status in {'explicit_final_marker', 'gsm_marker', 'boxed', 'terminal_answer_clause', 'corroborated_terminal_quantity'}:
        if answers_equivalent(terminal, final_answer) is False:
            flags.append('terminal_answer_disagrees_with_target')
    markers = list(FINAL_MARKER_RE.finditer(solution))
    if markers:
        terminal_marker = TERMINAL_MARKER_RE.search(solution)
        if len(markers) > 1 or terminal_marker is None:
            flags.append('nonterminal_or_multiple_final_markers')
        elif answers_equivalent(terminal_marker.group(2), final_answer) is not True:
            flags.append('unverified_terminal_marker_target')
    return list(dict.fromkeys(flags))


def build_completion(solution, final_answer):
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError('A nonempty worked solution is required')
    # Preserve solution text and an already-correct terminal block; do not erase evidence.
    final_answer = compact_answer(final_answer)
    if answer_type(final_answer) == 'missing' or answer_type(final_answer).startswith('unsupported'):
        raise ValueError('Missing or unsupported final target requires quarantine')
    markers = list(FINAL_MARKER_RE.finditer(solution))
    marker = TERMINAL_MARKER_RE.search(solution)
    if len(markers) > 1 or (markers and marker is None):
        raise ValueError('Nonterminal or multiple final-answer markers require review')
    if marker:
        if answers_equivalent(marker.group(2), final_answer) is not True:
            raise ValueError('Existing final marker conflicts with or is unverified against target')
        if marker.group(1).casefold().startswith('final answer') and ':' in marker.group(1):
            return solution
        raise ValueError('Noncanonical final marker requires review before formatting')
    return f'{solution.rstrip()}\n\nFinal answer: {final_answer}'


REVIEW_METADATA_COLUMNS = ['reviewer', 'reviewed_utc', 'policy_version', 'resolution']


def save_review_sheet(frame, path):
    """Preserve decisions on identical inputs; reject stale evidence rather than overwrite it."""
    import pandas as pd
    frame = frame.copy()
    for col in REVIEW_METADATA_COLUMNS:
        if col not in frame: frame[col] = ''
    path = Path(path)
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False, dtype=str)
        data_cols = [c for c in frame if not c.startswith('manual_') and c not in REVIEW_METADATA_COLUMNS]
        expected = frame[data_cols].fillna('').astype(str).reset_index(drop=True)
        if not set(data_cols).issubset(previous) or not previous[data_cols].reset_index(drop=True).equals(expected):
            raise ValueError(f'Stale review inputs: {path}; start a new run, do not overwrite review evidence')
        return previous
    frame.to_csv(path, index=False)
    return frame


def review_evidence(path, policy_version):
    import pandas as pd
    sheet = pd.read_csv(path, keep_default_na=False, dtype=str)
    if sheet.empty:
        if 'manual_same_problem_yes_no' not in sheet:
            raise ValueError(f'Mandatory source/target review is empty: {path}')
        return file_sha256(path)  # no generated pairs; approval still binds this file
    required = [c for c in sheet if c.startswith('manual_') and
                (c.endswith('_yes_no') or c == 'manual_keep_quarantine')]
    if not required:
        raise ValueError(f'No structured review decisions: {path}')
    for col in REVIEW_METADATA_COLUMNS:
        if col not in sheet or sheet[col].str.strip().eq('').any():
            raise ValueError(f'Incomplete review metadata {col}: {path}')
    if not sheet['policy_version'].eq(policy_version).all():
        raise ValueError(f'Wrong policy version: {path}')
    for value in sheet['reviewed_utc']:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
            raise ValueError(f'Invalid review date: {path}')
    for col in required:
        allowed = {'keep', 'quarantine'} if col == 'manual_keep_quarantine' else {'yes', 'no'}
        if not sheet[col].str.casefold().isin(allowed).all():
            raise ValueError(f'Incomplete/invalid decision {col}: {path}')
    if not sheet['resolution'].isin({'accepted', 'quarantined', 'corrected_in_new_run'}).all():
        raise ValueError(f'Unresolved review finding: {path}')
    # Accepted source/target examples cannot have failed checks. Pair 'no' means distinct, valid.
    if 'manual_same_problem_yes_no' not in sheet:
        accepted = sheet['resolution'].eq('accepted')
        for col in required:
            if col.endswith('_yes_no') and col != 'manual_needs_missing_diagram_yes_no':
                if sheet.loc[accepted, col].str.casefold().ne('yes').any():
                    raise ValueError(f'Failed target check accepted without repair: {path}')
        if 'manual_needs_missing_diagram_yes_no' in sheet:
            if sheet.loc[accepted, 'manual_needs_missing_diagram_yes_no'].str.casefold().ne('no').any():
                raise ValueError(f'Missing diagram accepted: {path}')
        if 'manual_keep_quarantine' in sheet:
            if sheet.loc[accepted, 'manual_keep_quarantine'].str.casefold().ne('keep').any():
                raise ValueError(f'Quarantine decision incorrectly accepted: {path}')
    return file_sha256(path)


def validate_review_approval(report_dir, filenames, policy, run_fingerprint, artifact_sha256=None):
    approval_path = Path(report_dir) / 'review_approval.json'
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    if approval.get('policy') != policy or approval.get('run_fingerprint') != run_fingerprint:
        raise ValueError('Review approval is for different policy/run inputs')
    if not approval.get('reviewer') or not approval.get('reviewed_utc'):
        raise ValueError('Approval requires reviewer and date')
    stamp = datetime.fromisoformat(approval['reviewed_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Invalid approval date')
    if not approval.get('missed_match_audit', {}).get('notes'):
        raise ValueError('Approval must document independently sought missed matches')
    evidence = {name: review_evidence(Path(report_dir) / name, policy['version']) for name in filenames}
    if approval.get('review_file_sha256') != evidence:
        raise ValueError('Review approval hashes do not match saved evidence')
    if artifact_sha256 is not None and approval.get('artifact_sha256') != artifact_sha256:
        raise ValueError('Review approval does not bind the current exported artifact hashes')
    return evidence


def validate_review_selection(report_dir, filenames, policy, selected_frames):
    """Enforce decisions across training AND fresh pools, independent of similarity scores."""
    import pandas as pd
    columns = ['row_id', 'dataset_id', 'original_row_id']
    if not selected_frames or any(not set(columns).issubset(frame) for frame in selected_frames):
        raise ValueError('Review enforcement requires source identity for every selected pool')
    selected = pd.concat([frame[columns].astype(str) for frame in selected_frames], ignore_index=True)
    selected_ids = set(selected['row_id'])
    source_ids = set(zip(selected['dataset_id'], selected['original_row_id']))
    kinds = {'duplicate_instance', 'parameter_or_word_variant', 'template_family', 'unrelated'}
    for name in filenames:
        sheet = pd.read_csv(Path(report_dir) / name, keep_default_na=False, dtype=str)
        for row in sheet.to_dict('records'):
            resolution = row['resolution']
            if 'manual_same_problem_yes_no' not in row:
                excluded = resolution != 'accepted' or row.get('manual_keep_quarantine') == 'quarantine'
                if excluded and row.get('row_id') in selected_ids:
                    raise ValueError(f'Review finding still selected: {name}/{row["row_id"]}')
                if excluded and (row.get('dataset'), row.get('original_index')) in source_ids:
                    raise ValueError(f'Source inspection finding still selected: {name}')
                continue
            kind = row.get('manual_overlap_kind')
            same = row['manual_same_problem_yes_no'].casefold()
            action = row.get('manual_policy_action', '').casefold()
            if kind not in kinds or action not in {'keep', 'remove'}:
                raise ValueError(f'Pair review requires classification and explicit keep/remove action: {name}')
            if (same == 'yes') != (kind == 'duplicate_instance'):
                raise ValueError(f'Contradictory same-instance classification: {name}')
            must_remove = kind == 'duplicate_instance' or resolution != 'accepted'
            if kind == 'parameter_or_word_variant' and policy['overlap_scope'] == 'conservative_lexical_variants':
                must_remove = True
            if must_remove and action != 'remove':
                raise ValueError(f'Policy/review finding requires removal: {name}')
            if action == 'keep':
                continue
            protected = 'training_row_id' in row or row.get('search_method', '').startswith('protected_')
            if protected:
                training_id = row.get('training_row_id', row.get('row_id_b'))
                retained = training_id in selected_ids
            else:
                retained = row.get('row_id_a') in selected_ids and row.get('row_id_b') in selected_ids
            if retained:
                raise ValueError(f'Reviewed overlap remains selected despite removal decision: {name}')


def verify_export_hashes(manifest_dir, report_dir, run_tag, summaries, prompt_spec_hash,
                         fresh_summary=None, debug_summary=None):
    """Rehash release artifacts; changing a file after export invalidates the gate."""
    expected = {f'train_n{size}_{run_tag}.jsonl': summary['sha256'] for size, summary in summaries.items()}
    if fresh_summary is not None:
        expected[f'fresh_candidate_pool_R1000_{run_tag}.jsonl'] = fresh_summary['sha256']
    if debug_summary is not None:
        expected[f'debug_32_{run_tag}.jsonl'] = debug_summary['sha256']
    expected['prompt_specification.json'] = prompt_spec_hash
    for name, digest in expected.items():
        path = Path(report_dir if name == 'prompt_specification.json' else manifest_dir) / name
        if file_sha256(path) != digest:
            raise ValueError(f'Exported artifact changed after generation: {name}')
    return expected


def reconcile_training_tokens(expected, verified):
    """Require the exact actual training-manifest coverage before computing a budget."""
    import pandas as pd
    for frame, cols, name in [(expected, {'row_id', 'supervised_tokens'}, 'expected'),
                              (verified, {'row_id', 'actual_supervised_tokens'}, 'verified')]:
        if not cols.issubset(frame.columns) or frame.empty:
            raise ValueError(f'{name} token records are empty or missing columns')
        if frame['row_id'].isna().any() or frame['row_id'].astype(str).str.strip().eq('').any():
            raise ValueError(f'{name} has missing IDs')
        if not frame['row_id'].is_unique:
            raise ValueError(f'{name} has duplicate IDs')
    if set(expected['row_id']) != set(verified['row_id']):
        raise ValueError('Training token coverage mismatch: missing or unexpected IDs')
    result = expected[['row_id', 'supervised_tokens']].merge(
        verified[['row_id', 'actual_supervised_tokens']], on='row_id', how='outer', validate='one_to_one', indicator=True)
    values = pd.to_numeric(result['actual_supervised_tokens'], errors='raise')
    if values.isna().any() or (~values.map(lambda n: n > 0 and n == int(n))).any():
        raise ValueError('Actual label counts must be positive integers')
    result['actual_supervised_tokens'] = values.astype('int64')
    result['difference'] = result['actual_supervised_tokens'] - result['supervised_tokens']
    if result['difference'].ne(0).any():
        raise ValueError('Actual training labels disagree with preprocessing counts')
    return result.drop(columns='_merge')


def validate_problem_manifest(frame, label):
    required = {'set_id', 'problem_id', 'problem'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError(f'{label} must contain nonempty set_id/problem_id/problem records')
    for col in required:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{label} has missing {col}')
    if frame.duplicated(['set_id', 'problem_id']).any():
        raise ValueError(f'{label} has duplicate composite problem IDs')


def validate_parent_evidence(seen, actual_manifest, metadata, expected_manifest_sha256=None):
    if seen.empty or 'problem_group_id' not in seen or not seen['problem_group_id'].is_unique:
        raise ValueError('Parent-seen groups must be nonempty and unique')
    if seen['problem_group_id'].isna().any() or seen['problem_group_id'].astype(str).str.strip().eq('').any():
        raise ValueError('Parent-seen groups contain missing IDs')
    required = {'checkpoint_id', 'checkpoint_sha256', 'actual_manifest_sha256', 'confirmed_by', 'confirmed_utc'}
    if not required.issubset(metadata) or any(not metadata[k] for k in required):
        raise ValueError('Parent checkpoint confirmation is incomplete')
    if not re.fullmatch(r'[0-9a-f]{64}', metadata['checkpoint_sha256']):
        raise ValueError('Parent checkpoint requires a SHA-256 identity')
    stamp = datetime.fromisoformat(metadata['confirmed_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Parent confirmation requires a valid timestamp')
    if metadata['actual_manifest_sha256'] != file_sha256(actual_manifest):
        raise ValueError('Actual parent training manifest hash mismatch')
    if expected_manifest_sha256 is not None and metadata['actual_manifest_sha256'] != expected_manifest_sha256:
        raise ValueError('Parent checkpoint was not trained on the agreed corrected R1000 manifest')
    records = [json.loads(line) for line in Path(actual_manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or len({r['row_id'] for r in records}) != len(records):
        raise ValueError('Actual parent training manifest has empty/duplicate IDs')
    actual_groups = {r['problem_group_id'] for r in records}
    if set(seen['problem_group_id']) != actual_groups:
        raise ValueError('Parent groups do not cover actual training manifest')
    return records
