---
name: criminal-behavioral-analysis
description: Behavioral/criminal profiling skill for the Profiler agent — read this when building a Crime Scene Report and Offender Typology from case images plus a Forensic agent report.
---

# Criminal Behavioral Analysis Skill

## When to Activate
- Building a behavioral/criminal profile from case images
- A Forensic agent's report is present in your prompt and needs to be turned into a profiling report

## Context You Already Have — read this first
- **You do NOT delegate to the Forensic agent yourself.** The orchestrator
  (main.py) already sent the case image(s) to the Forensic agent over A2A,
  received its report, and (if it found evidence of trauma) ran that
  report through a RAG pipeline against forensic_knowledge/ — all of that
  is already included directly in the message you receive. There is
  nothing to wait for and no separate file to read for it.
- **You have already been given 5 "short" reference images directly**
  (see `assets/short/`), covering the baseline classification frameworks:
  Palermo/Mastronardi's typology, the Holmes & Holmes typology, and the
  Organized/Disorganized/Mixed crime-scene classification. You always have
  this baseline even if you never call a tool.

## Step-by-Step Process
1. **Read the case image(s) and the Forensic agent's report** (and any RAG
   grounding excerpts) already present in your prompt.
2. **(Optional) Go deeper**: if the 5 short reference images aren't
   sufficient, use `read_file` on `references/REFERENCE.md` for the
   "Advanced version" — descriptions of the classification frameworks in
   more depth — or on the image descriptions under `assets/long/`. This
   step is optional; use your judgment.
3. **Build the Crime Scene Report**: analyze every detail across the
   images for irregularities that don't match how the scene would look in
   real life (e.g. a living room, park, or other setting), comparing
   against the frameworks in `references/REFERENCE.md`.
4. **Use the Forensic agent's report as evidence, not as your own
   conclusion**: you are not deriving cause/manner of death yourself —
   treat its findings (and any RAG-grounded reference material) as input
   to your behavioral profile.
5. **Write the final profile**, combining the Crime Scene Report and the
   Forensic agent's findings into the exact section order defined in
   `references/OUTPUT.md`.