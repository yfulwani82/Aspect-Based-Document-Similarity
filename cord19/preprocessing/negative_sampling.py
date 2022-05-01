import logging
import math
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set

from fuzzywuzzy import fuzz

from cord19.utils import get_sorted_pair

logger = logging.getLogger(__name__)


def get_cocitations(cits: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    from_to_cits = defaultdict(set)
    for from_id, to_id in cits:
        from_to_cits[from_id].add(to_id)
    cocits_set = set()
    for from_cit, to_cits in from_to_cits.items():
        for a in to_cits:
            for b in to_cits:
                cocits_set.add(get_sorted_pair(a, b))
    logger.info(f'total co-citation count: {len(cocits_set):,}')
    return cocits_set

def get_authors(doi, doi2paper):
    if doi in doi2paper:
        paper = doi2paper[doi]

        if 'authors' in paper:
            last_names = [a.split()[-1].lower() for a in paper['authors'] if len(a.split()) > 0]
            return last_names
        else:
            return []

    # elif doi in doi2paper:
    #     paper = doi2paper[doi]
    #     last_names = [a['last'].lower() for a in paper['metadata']['authors']]
    #     return last_names
    else:
        raise ValueError(f'DOI not found: {doi}')


def have_no_shared_authors(a_doi, b_doi, doi2paper):
    try:
        a_authors = set(get_authors(a_doi, doi2paper))
        b_authors = set(get_authors(b_doi, doi2paper))

        overlap = a_authors & b_authors

        if len(overlap) == 0:
            return True
        else:
            return False

    except ValueError:
        return False


# has same venue
def get_venue(doi, doi2paper):
    if doi in doi2paper:
        paper = doi2paper[doi]
        return str(paper['venue']).lower().strip() if 'venue' in paper else None
    else:
        raise ValueError(f'DOI not found: {doi}')


def have_not_same_venue(a_doi, b_doi, doi2paper):
    a_venue = get_venue(a_doi, doi2paper)
    b_venue = get_venue(b_doi, doi2paper)

    if a_venue is None or b_venue is None or a_venue == "" or b_venue == "":
        # cant answer if venue is not set
        return False

    if fuzz.ratio(a_venue, b_venue) < 0.75:
        # fuzzy string matching score must be low!
        return True
    else:
        return False


def get_negative_pairs(doi2paper, candidate_doc_ids: List[str], positive_pairs, cits_set, cocits_set, negative_ratio=0.5, negative_count=0):
    # negative_label = 'none'
    # negative_needed = 10000 #105492  # len(df)

    if negative_count > 0:
        negative_needed = negative_count
    else:
        negative_needed = math.ceil(len(positive_pairs) * negative_ratio)

    negative_rows = []
    negative_pairs = set()
    tries = 0

    print(f'Negatives needed: {negative_needed:,} (ratio: {negative_ratio})')

    while len(negative_pairs) < negative_needed:
        a = random.choice(candidate_doc_ids)
        b = random.choice(candidate_doc_ids)

        if a == b or a == np.nan or b == np.nan or isinstance(a,float) or isinstance(b,float):
            tries += 1
            continue

        pair = tuple((a, b))

        if pair in negative_pairs:
            continue

        cit_pair = get_sorted_pair(a, b)

        if cit_pair in cits_set:
            tries += 1
            continue

        if cit_pair in cocits_set:
            tries += 1
            continue

        if not have_no_shared_authors(a, b, doi2paper):
            tries += 1
            continue

        if not have_not_same_venue(a, b, doi2paper):
            tries += 1
            continue

        # text = get_text_from_doi(a)
        # text_b = get_text_from_doi(b)
        # if text == '' or text_b == '':
        #     continue

        # None of the criteria above matches...
        negative_pairs.add(pair)
        # negative_rows.append((
        #     a,
        #     b,
        #     text,
        #     text_b,
        #     negative_label,
        # ))

    logger.info(f'Found {len(negative_pairs):,} negative rows (tried {tries:,} random samples)')

    return negative_pairs
