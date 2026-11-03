"""Build the one-page mentor brief: python docs/brief/build_brief.py docs/brief/mentor_brief.pdf"""
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

F = "C:/Windows/Fonts/"
for n, f in (("UI", "segoeui"), ("UI-B", "segoeuib"), ("UI-I", "segoeuii"), ("UI-BI", "segoeuiz"), ("UI-L", "segoeuil")):
    pdfmetrics.registerFont(TTFont(n, F + f + ".ttf"))
registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-BI")

INK, MUTED, ACCENT = colors.HexColor("#1f2933"), colors.HexColor("#5b6670"), colors.HexColor("#0f5c6e")
LIGHT, RULE = colors.HexColor("#e6f1f3"), colors.HexColor("#c9d3d8")

body = ParagraphStyle("b", fontName="UI", fontSize=8.6, leading=11.6, textColor=INK, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="UI-B")
h = ParagraphStyle("h", fontName="UI-B", fontSize=10, leading=13, textColor=ACCENT, spaceBefore=5, spaceAfter=2)
bul = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=1.5)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
H = lambda t: story.append(Paragraph(t, h))


def B(items):
    for t in items:
        story.append(Paragraph(t, bul, bulletText="•"))


def T(rows, widths, header=True):
    data = [[Paragraph(c, cellb if header and i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 3))


P("LEXSI LABS AI SAFETY &amp; ALIGNMENT HACKATHON · TRACK 2b: REPAIRING SAFETY DRIFT WITHOUT RETRAINING",
  ParagraphStyle("k", parent=body, fontName="UI-B", fontSize=7.4, textColor=ACCENT, spaceAfter=1))
P("Same model, different language, different safety",
  ParagraphStyle("t", fontName="UI-B", fontSize=16, leading=20, textColor=INK, spaceAfter=1))
P("Team: Aditya Arnav, Pallab Das · Mentor brief, 3 November 2026 · Core model: Tiny Aya Fire / Earth (3.35B), "
  "reference Tiny Aya Global · Repos: CuratorKIT, AuditKIT, CircuitKIT, SafeTune",
  ParagraphStyle("m", parent=body, fontName="UI-L", fontSize=8.2, textColor=MUTED, spaceAfter=4))

H("The problem")
P("Tiny Aya's regional models are tuned for the people who speak those languages: <b>Fire</b> for South Asia, "
  "<b>Earth</b> for Africa and West Asia. On average they are almost as safe as Tiny Aya Global, but in their "
  "<b>weakest language</b> they are 9–10 points less safe on MultiJail: 87.0% safe for Global, 78.1% for Fire, "
  "77.5% for Earth (Tiny Aya tech report, Table 7). A harmful request Global refuses may be answered by the model "
  "built for that language's speakers, and a check on averages would not notice.")

H("What we are doing")
T([["Step", "What happens", "Lexsi kit"],
   ["1. Data", "Harmful prompts (Aya red-teaming, MultiJail) and harmless look-alikes (XSTest) in English, Hindi, "
               "Bengali, Tamil, Hinglish, Arabic, Swahili. Splits are by item, so a translation of a test prompt can "
               "never be used for the repair.", "CuratorKIT"],
   ["2. Measure", "Global, Fire and Earth, per language, on three axes: refusal of harmful prompts, wrongful refusal "
                  "of harmless ones, and general ability (Global-MMLU-Lite). Two independent LLM judges plus a "
                  "hand-checked sample, with and without Tiny Aya's default system preamble.", "AuditKIT"],
   ["3. Locate", "EAP-IG finds the refusal circuit per language and per model, checked with the faithfulness "
                 "tests. Where did regional tuning change it?", "CircuitKIT"],
   ["4. Repair", "Four arms, compared head to head (below). Settings are tuned on the repair split only.", "SafeTune"],
   ["5. Verify", "Re-test all three axes, then again after 4-bit compression (NF4 and GGUF Q4_K_M).",
    "AuditKIT"]],
  [17 * mm, 133 * mm, 20 * mm])

H("The repair comparison")
T([["Arm", "Edit", "Why it is there"],
   ["ReSta", "Fire + α·(Global − Base), whole model", "Established weight-space safety restore"],
   ["<b>Circuit (C-ΔΘ)</b>", "<b>The same safety vector, only in the layers CircuitKIT flags</b>",
    "<b>Our main bet. Same edit as ReSta, so the difference isolates what the circuit map adds</b>"],
   ["Merge-back", "Blend Fire back toward Global", "Naive baseline; likely costs regional ability"],
   ["Steering", "Refusal direction added at run time", "Known to cause over-refusal; we check on Tiny Aya"]],
  [24 * mm, 66 * mm, 80 * mm])
P("All four are training-free weight or activation edits, matching the problem statement. A circuit-restricted LoRA "
  "is an optional fifth arm.")

H("How we will judge success (pre-registered, before any repair)")
B(["Close at least <b>half</b> of the worst-language refusal gap to Global.",
   "Wrongful refusal of harmless prompts rises by <b>at most 2 points</b> in any language; general ability falls by "
   "<b>at most 1 point</b>.",
   "All of the above <b>still holds at 4-bit</b>. Per-language 95% CIs and exact McNemar tests; every arm published, "
   "including losing ones, and negative results reported."])

H("Where we are today")
P("The private repository is set up. Data builds from the real sources, with the leakage check passing. Judges, "
  "statistics, the run ledger and all four kit integrations are written. 26 tests pass. Next: the day-1 GPU smoke "
  "test (reproduce the gap on about 200 prompts, a first EAP-IG run on Tiny Aya), then the full baseline "
  "(4–5 Nov). Along the way we found gaps in the kits, all of them English-only defaults "
  "(refusal detection, benchmark loaders, cross-language circuit tasks). We plan to contribute fixes upstream as "
  "pull requests.")

H("What we would value your view on")
B(["Do the success thresholds above look right to you?",
   "Is a public multilingual safety eval set (prompts and labels only, no harmful completions) acceptable under "
   "Cohere's acceptable-use policy?",
   "Have you seen pitfalls with refusal circuits or C-ΔΘ on Cohere2 models that we should plan for?"])

doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=13 * mm,
                        bottomMargin=11 * mm, title="Same model, different language, different safety",
                        author="Aditya Arnav, Pallab Das", subject="Mentor brief, Lexsi hackathon")
doc.build(story)
print("ok", sys.argv[1])
