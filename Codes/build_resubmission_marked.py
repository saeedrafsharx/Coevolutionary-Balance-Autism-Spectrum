"""Build a latexdiff source; obsolete original numeric tables are excluded."""
from pathlib import Path
import subprocess,re
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision/resubmission'
old=(R/'manuscript_marked.tex').read_text()
# Carry original exhibit references used only within deleted text, avoiding ?? labels.
refs=re.search(r'\\AtBeginDocument\{%.*?\n\}',old,re.S).group(0)
s=(R/'manuscript.tex').read_text().replace('\\newif\\ifillustrated','\\def\\SubmissionVersion{1}\n\\newif\\ifillustrated')
(R/'source_support/revision_main_for_diff.tex').write_text(s)
p=subprocess.run(['latexdiff','--disable-citation-markup','--floattype=FLOATSAFE',str(R/'source_support/original_main_for_diff.tex'),str(R/'source_support/revision_main_for_diff.tex')],capture_output=True,text=True,check=True)
s=p.stdout.replace('\\begin{document}',refs+'\n\\begin{document}',1)
(R/'manuscript_marked.tex').write_text(s)
print('Marked manuscript source regenerated.')
