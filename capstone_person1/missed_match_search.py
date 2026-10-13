"""A second bounded retrieval channel; signatures suggest cases, never establish identity."""
import re
from collections import defaultdict
from capstone_person1.instance_matching import normalize_instance,word_shingles

def signature(text):
    text=normalize_instance(text).casefold().rstrip('.!? ')
    text=re.sub(r'\d+(?:\.\d+)?','<N>',text)
    return re.sub(r'(letters of|word)\s+[a-z]+',r'\1 <WORD>',text)

def retrieve_signature_pairs(frame,protected):
    buckets=defaultdict(list)
    for row in frame.itertuples(index=False): buckets[signature(row.problem)].append((row.row_id,row.problem))
    pairs,protected_pairs=[],[]
    def pair(left,right,protected=False):
        a,b=word_shingles(left[1]),word_shingles(right[1]);score=len(a&b)/len(a|b) if a|b else 1
        return dict(row_id_a=left[0],row_id_b=right[0],problem_a=left[1],problem_b=right[1],similarity=score,
                    retrieval='protected_parameter_signature' if protected else 'parameter_signature')
    for key,rows in sorted(buckets.items()):
        rows=sorted(rows)
        # A deterministic star bounds large template families without quadratic pairs.
        pairs.extend(pair(rows[0],row) for row in rows[1:])
    for row in protected.itertuples(index=False):
        right=('protected:'+row.set_id+':'+row.problem_id,row.problem)
        protected_pairs.extend(pair(left,right,True) for left in buckets.get(signature(row.problem),[]))
    return pairs,protected_pairs
