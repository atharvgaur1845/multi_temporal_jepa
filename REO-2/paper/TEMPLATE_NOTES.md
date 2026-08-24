# Template mechanics

## Get the style file

`main.tex` will not compile until `neurips_2026.sty` sits next to it. Take it from the **NeurIPS 2026
Overleaf template** — not a copy of the 2024/2025 style, and not a hand-modified one. Off-template
risks desk rejection, and the workshop options below only exist in the current file.

```
REO-2/paper/
├── main.tex
├── references.bib
├── neurips_2026.sty      <- drop it here
└── figures/
```

## The two lines that make it a REO workshop paper

```latex
\workshoptitle{2nd Workshop on Advances in Representation Learning for Earth Observation}
\usepackage[dblblindworkshop]{neurips_2026}
```

`dblblindworkshop` (not `preprint`, not `final`) is what produces the anonymised workshop layout.

## Delete from the stock template

- `\newpage`
- `\input{checklist.tex}` — the NeurIPS checklist is **not required** for this workshop.

Leaving either in costs a page or appends a section that should not be there.

## Length

**4 pages**, references and appendix excluded. Do not shrink margins, do not `\vspace` your way under
the limit, do not drop to `\small` in the body — all three are visible and all three read as padding.
If it does not fit, cut a paragraph.

## Double-blind

- No `\author` block, no affiliation, no acknowledgements.
- **No `github.com/atharvgaur1845`** anywhere — not in text, not in a footnote, not in a comment that
  could survive into the PDF. Use `anonymous.4open.science`, or omit the code link entirely.
- Do not write "our previous work" in a way that identifies the author. Cite in third person.

## Strip PDF metadata — this is a real desk-reject cause

LaTeX leaks author names into PDF properties even with no `\author` block (from `\title`, from
`hyperref`, from the PDF producer string, and from Overleaf's own project metadata). `main.tex`
already sets:

```latex
\hypersetup{pdfauthor={},pdftitle={},pdfsubject={},pdfkeywords={},pdfcreator={},pdfproducer={}}
```

**Verify it worked, do not assume:**

```bash
pdfinfo main.pdf                       # poppler-utils
exiftool main.pdf | grep -i -E 'author|creator|producer|title|company'
```

If anything identifying survives:

```bash
exiftool -all:all= -overwrite_original main.pdf
pdfinfo main.pdf                       # confirm it is clean, then re-check the PDF still opens
```

Also check the **source bundle** if the venue asks for one: `.bib` comments, `\usepackage{}` comments,
`% author:` lines, and any `\todo{}` left in the file.

## Before upload

```bash
grep -n "PENDING" main.tex              # must return nothing
grep -rn "atharv\|github.com" main.tex references.bib   # must return nothing
```

`\PENDING{}` renders in red and is impossible to miss in the compiled PDF — that is the point. Every
one must be replaced by a number from a committed CSV or by a deleted claim.
