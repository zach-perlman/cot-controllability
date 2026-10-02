# exp07 search stopped; validation candidates (2026-10-02, before any validation generation)

Search: rounds round0 and round1 (search_log.jsonl), 8 candidates beyond baseline and upgraded, out of a budget of
at most 20. Stopped because no candidate's mean S(1000) over the 2 search models x 5 search rules differed from
upgraded's by more than the noise (paired question bootstrap, 95% CIs about +-5 points), the top 4 were within 1.5
points of each other, and the one combination tried (guide_in_system_long) scored below both of its parts. A gain
large enough for the search split to detect looked unlikely to be found in the remaining time.

Search scores (mean S(1000); difference from upgraded with 95% CI, paired over the 60 search questions):

| candidate | S(1000) | - upgraded |
|---|---|---|
| short_sentences | 51.3 | +4.1 [-1.1, +9.4] |
| rule_last | 50.5 | +3.4 [-1.3, +8.2] |
| long_examples | 50.3 | +3.2 [-1.8, +7.9] |
| guide_in_system | 49.7 | +2.6 [-2.8, +7.9] |
| opening | 49.3 | +2.1 [-2.7, +6.7] |
| upgraded | 47.2 | |
| six_examples | 47.0 | -0.1 [-5.3, +5.2] |
| guide_in_system_long | 46.4 | -0.6 [-6.0, +4.6] |
| examples_with_guide | 33.0 | -14.0 [-20.2, -8.1] |

Eligibility (manifest: empty, degenerate, meta and ended_clean_early within 5 points of upgraded's): all candidates
eligible; short_sentences has the largest ended_clean_early (3.0 vs upgraded's 1.2).

Validation (manifest rule: the 3 eligible candidates with the highest search score, plus upgraded, on the validation
split, search models and search rules): short_sentences, rule_last, long_examples, upgraded. The winner is the
highest validation score among them and goes to the test even if it does not beat upgraded.

UNVERIFIED.
