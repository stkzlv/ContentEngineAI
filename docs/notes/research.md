# Research Module Notes

<!--
Read before changing src/research; AGENTS.md links here.

Each entry records a defect and what it cost, so the shape that produced it is
recognisable the next time. Add to it the same way: what broke, why it was
invisible, and what now catches it.
-->

- **One spike week in Google Trends distorted every trend and peak month.** The first full run labelled nearly every keyword "falling" and gave most an April peak. Trends showed a spike in the weeks of 5-19 April 2026 across unrelated searches at once ("wireless earbuds" and "smart watch" both topped out on 12 April), a data anomaly rather than demand. Means carried it everywhere: the twelve-month mean was inflated, so the last quarter read as a fall, and April won the five-year monthly average. A comparison against the year's mean also made every holiday item fall each autumn. Shares now come from median weeks, the trend compares the last quarter with the same quarter a year earlier, and a peak month must be the top month in at least two complete years by a clear margin. Two traps came with the fix: a keyword with zero interest then and now must read as flat (1.0), not unknown, or every dead keyword leaves the drop list; and a sparse year (a typical month of zero, top month under one Trends unit) must not vote, or scattered single readings elect a peak month about half the time. Tests cover a spike week, a one-year spike, a flat year, a seasonal high, a dead keyword and sparse noise.
- **Rising searches next to broad seeds were noise.** Seeds such as "tech gadgets" returned "tech etruesports", "credit card rewards" and "august 2026 tech gadgets" as keywords to add. Rising searches now come from next to the strongest configured keywords, and the 20 fastest-rising are measured against the anchor (twelve months only, to spare Trends requests) and kept only at or above the keywords' median share in some country.
- **The uncovered-topic list held one stem.** Listed stem by stem and cut to ten, every candidate came from "how to turn off". The list now alternates between stems.
- **A sample rerun measured the last run's scripts.** The producer's script step reuses a script already on disk, so a second `sample` run into the same directory read the old scripts back with half their checks blank. Each variant's samples are cleared before a run, and a failed clear raises.
- **Comparing variants by totals rewarded dropping topics.** The recommendation compares the share of verified scripts with a wrong or outdated step, counting only scripts with at least one sourced verdict, and reports dropped and unjudged scripts beside it.
