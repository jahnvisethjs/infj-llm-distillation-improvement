from capstone_person1.pipeline_controls import *
from capstone_person1.instance_matching import instance_hash, context_answer_key, context_equivalent

def stable_row_id(dataset_id, revision, original_index):
    return sha256_text(f'{dataset_id}|{revision}|train|{original_index}')

def canonicalize_orca(row, idx, *, spec, config):
    orca_revision, ORCA_DATASET_ID = spec['revision'], spec['dataset_id']
    original_idx = int(row.get('__original_index', idx))
    problem_raw = raw_text(row.get('question', ''))
    solution_raw = raw_text(row.get('answer', ''))
    problem = clean_text(problem_raw)
    solution = clean_text(solution_raw)
    extraction = extract_answer_metadata(problem, solution)
    final_answer, answer_status = extraction.candidate, extraction.mode
    flags, review_flags = [], []
    if not problem: flags.append('empty_problem')
    if not solution: flags.append('missing_solution')
    if answer_status == 'missing': flags.append('missing_final_answer')
    if answer_status.startswith('ambiguous'): flags.append('ambiguous_final_answer')
    if answer_status == 'terminal_single_number_review': review_flags.append('heuristic_answer_extraction_review')
    if CJK_RE.search(problem) or CJK_RE.search(solution): flags.append('non_english_text')
    flags.extend(diagram_flags(problem, solution))
    flags.extend(target_quality_flags(problem, solution, final_answer, answer_status))
    flags = list(dict.fromkeys(flags))
    problem_hash = instance_hash(problem)
    row_id = stable_row_id(ORCA_DATASET_ID, orca_revision, original_idx)
    return {
        'row_id': row_id, 'dataset_id': 'orca', 'dataset_revision': orca_revision,
        'original_row_id': str(original_idx), 'original_source': 'microsoft_orca_math',
        'problem_raw': problem_raw, 'solution_raw': solution_raw,
        'problem': problem, 'solution': solution, 'final_answer': final_answer,
        'answer_parse_status': answer_status, 'correctness_status': 'unknown',
        'answer_extraction_json': json.dumps(extraction.to_dict(), ensure_ascii=False),
        'answer_evidence_start': extraction.evidence_start, 'answer_evidence_end': extraction.evidence_end,
        'answer_evidence_text': extraction.evidence_text, 'answer_confidence': extraction.confidence,
        'answer_type': answer_type(final_answer), 'answer_ambiguity_reason': extraction.ambiguity_reason,
        'answer_evidence_basis': extraction.evidence_basis,
        'extraction_consistency_status': 'consistent_terminal_evidence' if extraction.candidate and extraction.confidence!='heuristic' and context_equivalent(problem,extraction.candidate,final_answer) is True else 'unresolved_requires_review',
        'solution_completeness_status': 'unknown',
        'source_fields_origin': row.get('__source_fields_origin', 'pinned_upstream'),
        'question_type': 'math-word-problem', 'problem_type': '', 'synthetic': True,
        'quality_flags': flags, 'review_flags': review_flags, 'eligible': len(flags) == 0,
        'answer_key': context_answer_key(problem, final_answer),
        'problem_hash': problem_hash, 'problem_group_id': 'exact_' + problem_hash[:24],
        'split': 'candidate' if len(flags) == 0 else 'quarantine',
        'problem_char_length': len(problem), 'solution_char_length': len(solution),
    }

def canonicalize_numina(row, idx, *, spec, config):
    numina_revision, NUMINA_DATASET_ID = spec['revision'], spec['dataset_id']
    EXCLUDE_SYNTHETIC_NUMINA = config['exclude_synthetic_numina']
    original_idx = int(row.get('__original_index', idx))
    problem_raw = raw_text(row.get('problem', ''))
    solution_raw = raw_text(row.get('solution', ''))
    problem = clean_text(problem_raw)
    solution = clean_text(solution_raw)
    final_answer = compact_answer(row.get('answer', ''))
    extraction = extract_answer_metadata(problem, solution)
    question_type = clean_text(row.get('question_type', ''))
    original_source = clean_text(row.get('source', ''))
    synthetic = bool(row.get('synthetic', False))
    problem_valid = clean_text(row.get('problem_is_valid', '')).casefold()
    solution_valid = clean_text(row.get('solution_is_valid', '')).casefold()
    answer_status = 'parsed' if final_answer.casefold() not in MISSING_ANSWER_VALUES else 'missing'
    flags, review_flags = [], []
    if not problem: flags.append('empty_problem')
    if not solution: flags.append('missing_solution')
    if question_type.casefold() != 'math-word-problem': flags.append('outside_math_word_problem_metadata_scope')
    if answer_status != 'parsed': flags.append('missing_or_sentinel_final_answer')
    if SUSPECT_ANSWER_RE.search(final_answer): flags.append('suspect_answer_format')
    if problem_valid not in {'yes', 'true'}: flags.append('numina_problem_not_valid')
    if solution_valid not in {'yes', 'true'}: flags.append('numina_solution_not_valid')
    if original_source.casefold().replace('-', '_').replace(' ', '_') == 'orca_math':
        flags.append('embedded_orca_source_removed')
    if EXCLUDE_SYNTHETIC_NUMINA and synthetic:
        flags.append('synthetic_removed_by_team_policy')
    flags.extend(diagram_flags(problem, solution))
    if CJK_RE.search(problem) or CJK_RE.search(solution): flags.append('non_english_text')
    if answer_status == 'parsed' and squash_answer(final_answer) not in squash_answer(solution):
        review_flags.append('final_answer_not_literal_in_solution')
    flags.extend(target_quality_flags(problem, solution, final_answer, answer_status))
    if not extraction.candidate or extraction.confidence == 'heuristic' or context_equivalent(problem, extraction.candidate, final_answer) is not True:
        flags.append('unproven_solution_target_consistency')
    flags = list(dict.fromkeys(flags))
    problem_hash = instance_hash(problem)
    row_id = stable_row_id(NUMINA_DATASET_ID, numina_revision, original_idx)
    return {
        'row_id': row_id, 'dataset_id': 'numina', 'dataset_revision': numina_revision,
        'original_row_id': str(original_idx), 'original_source': original_source,
        'problem_raw': problem_raw, 'solution_raw': solution_raw,
        'problem': problem, 'solution': solution, 'final_answer': final_answer,
        'answer_parse_status': answer_status, 'correctness_status': 'unknown',
        'answer_extraction_json': json.dumps(extraction.to_dict(), ensure_ascii=False),
        'answer_evidence_start': extraction.evidence_start, 'answer_evidence_end': extraction.evidence_end,
        'answer_evidence_text': extraction.evidence_text, 'answer_confidence': extraction.confidence,
        'answer_type': answer_type(final_answer), 'answer_ambiguity_reason': extraction.ambiguity_reason,
        'answer_evidence_basis': extraction.evidence_basis,
        'extraction_consistency_status': 'consistent_terminal_evidence' if extraction.candidate and extraction.confidence!='heuristic' and context_equivalent(problem,extraction.candidate,final_answer) is True else 'unresolved_requires_review',
        'solution_completeness_status': 'unknown',
        'source_fields_origin': row.get('__source_fields_origin', 'pinned_upstream'),
        'provided_answer_raw': raw_text(row.get('answer', '')),
        'numina_problem_validity_raw': raw_text(row.get('problem_is_valid', '')),
        'numina_solution_validity_raw': raw_text(row.get('solution_is_valid', '')),
        'question_type': question_type, 'problem_type': clean_text(row.get('problem_type', '')),
        'synthetic': synthetic, 'quality_flags': flags, 'review_flags': review_flags, 'eligible': len(flags) == 0,
        'answer_key': context_answer_key(problem, final_answer),
        'problem_hash': problem_hash, 'problem_group_id': 'exact_' + problem_hash[:24],
        'split': 'candidate' if len(flags) == 0 else 'quarantine',
        'problem_char_length': len(problem), 'solution_char_length': len(solution),
    }

