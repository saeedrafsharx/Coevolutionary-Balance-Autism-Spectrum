"""Build marked narrative changes and retain all deleted original main tables."""
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

# Original table contents cannot safely receive word-level latexdiff markup
# around booktabs commands. Show every removed table intact in a red appendix.
original=(ROOT/'manuscript.tex').read_text().split('\\section*{Supporting information}')[0]
tables=re.findall(r'\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}',original,re.S)
assert len(tables)==12
appendix=r"""
\clearpage
\section*{Deleted original main-article tables}
The following twelve tables reproduce the removed original tables in full for
comparison with the replacement tables above. Red text marks material deleted
from the original submission. These numerical values and methodological accounts
are historical and do not describe the revised analysis. Table numbers below
refer to the original submission. This deletion record appears only in the
marked manuscript; it is not part of the clean revised article.
\begingroup
\singlespacing
\color{red!70!black}
\setcounter{table}{0}
\renewcommand{\thetable}{Original \arabic{table}}
\renewcommand{\theHtable}{deletedoriginal.\arabic{table}}
"""
for table in tables:
 table=re.sub(r'\\begin\{table\}(?:\[[^\]]*\])?',lambda m:r'\begin{table}[H]\color{red!70!black}',table,count=1)
 table=re.sub(r'\\label\{([^}]+)\}',lambda m:r'\label{deletedoriginal:'+m[1]+'}',table)
 # Fit wide old numerical tables without changing their contents.
 table=re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})',lambda m:r'\resizebox{\linewidth}{!}{%'+ '\n'+m[1]+'\n}',table,flags=re.S)
 appendix+='\n'+table+'\n\\clearpage\n'
appendix+='\\endgroup\n'
s=s.replace('\\end{document}',appendix+'\n\\end{document}')
(R/'source_support/deleted_original_tables.tex').write_text(appendix)
(R/'manuscript_marked.tex').write_text(s)
print('Marked manuscript source regenerated.')
