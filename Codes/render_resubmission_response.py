"""Resolve response locations against the clean manuscript and render LaTeX."""
from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]/'revision/resubmission'
s=(R/'response_to_reviewers.md').read_text();aux=(R/'manuscript_submission.aux').read_text()
loc={k:(int(line),int(page)) for k,line,page in re.findall(r'\\newlabel\{lin:([^}]+)\}\{\{(\d+)\}\{(\d+)\}',aux)}
map1={1:['results_intra','results_inter','results_ML'],2:['participants','groupmodels'],3:['groupmodels','ML','selection'],4:['efficiency','nulls'],5:['results_null','nulls'],6:['power'],7:['preprocessing','falff_provenance'],8:['states'],9:['sex'],10:['data']}
def location(keys):return '; '.join(f'p. {loc[k][1]}, line {loc[k][0]}' for k in keys)+'.'
s=s.replace('{LOCATION_EDITOR}',location(['preprocessing','falff_provenance']))
parts=re.split(r'(?m)(^## Reviewer [12]\n|^### [^\n]+\n)',s);reviewer=0;keys=[];result=''
for part in parts:
 if part.startswith('## Reviewer '):reviewer=int(part.split()[2]);result+=part
 elif part.startswith('### '):
  n=re.match(r'### (\d+)\.',part)
  keys=map1[int(n.group(1))] if reviewer==1 and n else (['sex'] if not n else {1:['falff_provenance'],2:['states'],3:['ML']}[int(n.group(1))]);result+=part
 elif reviewer and '**Location:**' in part:result+=re.sub(r'\*\*Location:\*\*[^\n]*','**Location:** '+location(keys),part)
 else:result+=part
(R/'response_to_reviewers.md').write_text(result)
def esc(t):
 t=t.replace('−','-').replace('–','--').replace('—','---').replace('≥',r'$\geq$').replace('±',r'$\pm$').replace('×',r'$\times$').replace('α',r'$\alpha$').replace('²',r'$^2$').replace('Mišić','Misic')
 return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#'}.get(ch,ch) for ch in t)
lines=[]
for para in result.split('\n\n'):
 para=' '.join(para.splitlines())
 if not para:continue
 m=re.match(r'^(#{1,3}) (.*)',para)
 if m:lines.append('\\'+('subsection' if len(m[1])==3 else 'section')+'*{'+esc(m[2])+'}')
 else:
  t=esc(para);t=re.sub(r'\*\*(.*?)\*\*',r'\\textbf{\1}',t);t=re.sub(r'`([^`]+)`',r'\\texttt{\1}',t);lines.append(t+'\n')
tex=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip,microtype,lmodern}
\emergencystretch=3em
\sloppy
\begin{document}
'''+ '\n\n'.join(lines)+'\n\\end{document}\n'
(R/'response_to_reviewers.tex').write_text(tex)
print('Response locations updated and LaTeX rendered.')
