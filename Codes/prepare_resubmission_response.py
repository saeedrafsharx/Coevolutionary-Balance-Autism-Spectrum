from pathlib import Path
import re
R=Path('revision/resubmission'); old=Path('revision/revised_paper/response_to_reviewers.md').read_text()
s=old[:old.index('## Items required before this response can be finalized')]
s=s.replace(' — working draft','')
a=s.index('**Author-review draft');b=s.index('Dear Editor',a)
s=s[:a]+'''**Author review version.** The response is complete point by point, but the historical fALFF provenance concern remains unresolved (Editor, R1.7 and R2.1). Do not submit it as a fully resolved response until that scientific issue is settled. Locations refer to manuscript_submission.pdf; line numbers identify the start of the relevant passage. Reviewer points are reproduced from Comments.pdf below.

'''+s[b:]
s=s.replace('The working Methods','The Methods').replace('All revised main figures','All 11 revised main figures')
s=s.replace('The combined primary-summary figure has two\npanels whose descriptions match their contents.', 'Global distributions, ROC curves and classifier uncertainty now have separate, matched captions.')
s=s.replace('BH correction spans all 284 control/pipeline/null comparisons.', 'BH correction in the revised main text spans all 560 participant/pipeline/null comparisons, covering both diagnostic groups. The separately documented control-only family remains identifiable in S2.')
s=s.replace('**Addressed in the draft, with a final literature check pending.**','**Addressed.**')
s=s.replace('The authors should check whether additional recent mega-analytic or sex-specific\nfALFF work warrants inclusion before finalizing the reference set.','Holiga et al. (2019) supplies a complementary large-cohort example of reproducible connectivity changes with overlapping individual distributions; Vasa and Mišić (2022) clarifies the role of null constraints. These references prevent interpreting all multisite findings as uniformly nonreproducible.')
a=s.index('**Local materials updated; public release pending.**');b=s.index('### Sex and generalizability',a)
s=s[:a]+'''**Updated source and results supplied.** S2 Data includes complete machine-readable results, covariates, feature matrices, out-of-fold predictions, selection records and provenance. S3 Code includes the revision source, execution instructions and analysis specifications. The local release README clearly distinguishes historical notebooks from current analyses and states the remaining fALFF limitation. No external repository has been updated during this revision preparation, and we do not claim that stale remote content has already been replaced. If the earlier public URL is retained in the submission, its README still needs to be updated by the authors to point to this exact release.

'''+s[b:]
s=s.replace('S2 machine-readable ZIP','S2 Data and S3 Code ZIP archives')
s=s.replace('A final\n   production/style check remains required after author approval.', 'The submission-format file has inline captions and separate Fig1--Fig11 TIFFs; a companion illustrated PDF is supplied for reading. Main and supporting PDFs have been compiled and cross-references checked.')
s=s.replace('A final author/language pass will follow scientific approval.', 'Terminology, captions and interpretation have been checked against the final analysis tables.')
# Add selection completion within response 3 rather than unrelated appended issue.
needle='Both pipelines\nuse the same primary participants and nodes.'
s=s.replace(needle,needle+''' Feature selection was already nested in the baseline: ANOVA retained 22 features in 11 outer fits, 44 in 17, and 88 in 22; reconstruction reproduces all saved predictions. A separately labeled exploratory follow-up compares ANOVA, mutual information and L1-ranked selection with training-only redundancy pruning. Four model/pipeline point estimates improve and six decline, with every paired interval including zero. All ten comparisons, ROC curves, confusion matrices and selected feature names are supplied, without choosing an outer-test winner.''')
# Actual comment text from supplied PDF, not web quotes.
c=Path('revision/resubmission/source_support/reviewer_comments.txt').read_text().replace('\x0c',' ')
r1=c[c.index('Reviewer #1: 1. Several'):c.index('Reviewer #2: This manuscript')]
r2=c[c.index('Reviewer #2: This manuscript'):c.index('6. PLOS authors have')]
comments={}
for n in range(1,11):
 start=re.search((r'Reviewer #1: 1\.' if n==1 else rf'(?m)^\s*{n}\.\s*'),r1)
 end=re.search(rf'(?m)^\s*{n+1}\.\s*',r1[start.end():]) if n<10 else re.search('SEX AND GENERALIZABILITY',r1[start.end():])
 text=r1[start.end():start.end()+end.start()] if end else r1[start.end():]
 comments['r1_'+str(n)]=' '.join(text.split())
comments['sex']=' '.join(r1[r1.index('SEX AND GENERALIZABILITY.')+len('SEX AND GENERALIZABILITY.'):].split())
for n in range(1,4):
 start=re.search(rf'(?m)^\s*{n}\.\s*',r2); end=re.search(rf'(?m)^\s*{n+1}\.\s*',r2[start.end():]) if n<3 else None
 comments['r2_'+str(n)]=' '.join((r2[start.end():start.end()+end.start()] if end else r2[start.end():]).split())
aux=(R/'manuscript_submission.aux').read_text()
loc={k:(int(line),int(page)) for k,line,page in re.findall(r'\\newlabel\{lin:([^}]+)\}\{\{(\d+)\}\{(\d+)\}',aux)}
map1={1:['results_intra','results_inter','results_ML'],2:['participants','groupmodels'],3:['groupmodels','ML','selection'],4:['efficiency','nulls'],5:['results_null','nulls'],6:['power'],7:['preprocessing','falff_provenance'],8:['states'],9:['sex'],10:['data']}
parts=re.split(r'(?m)(^## Reviewer [12]\n|^### [^\n]+\n)',s);reviewer=0;out=''
for part in parts:
 if part.startswith('## Reviewer '):reviewer=int(part.split()[2]);out+=part
 elif part.startswith('### '):
  out+=part+'\n'
  num=re.match(r'### (\d+)\.',part)
  key=('r'+str(reviewer)+'_'+num.group(1)) if num else 'sex'
  if key in comments:out+='**Comment:** '+comments[key]+'\n\n'
  labels=map1[int(num.group(1))] if reviewer==1 and num else (['sex'] if not num else {1:['falff_provenance'],2:['states'],3:['ML']}[int(num.group(1))])
  out+='**Location:** '+ '; '.join(f'p. {loc[k][1]}, line {loc[k][0]}' for k in labels)+'.\n\n'
 else:out+=part
out+='''
## Release status for the authors

The clean manuscript, illustrated reading copy, supporting PDF, data archive, code archive, and separate figures have been prepared. The historical fALFF filtering/denominator concern remains scientifically open; any needed verified recomputation must precede a claim that the revision fully resolves that point. The original externally hosted README has not been modified. Confirm portal declarations and the final release link if retaining that external link.
'''
(R/'response_to_reviewers.md').write_text(out)
def esc(t):
 t=t.replace('−','-').replace('–','--').replace('—','---').replace('≥',r'$\geq$').replace('±',r'$\pm$').replace('×',r'$\times$').replace('α',r'$\alpha$').replace('²',r'$^2$').replace('Mišić','Misic')
 return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#'}.get(ch,ch) for ch in t)
lines=[]
for para in out.split('\n\n'):
 para=' '.join(para.splitlines())
 if not para:continue
 m=re.match(r'^(#{1,3}) (.*)',para)
 if m:
  cmd={1:'section',2:'section',3:'subsection'}[len(m[1])];lines.append('\\'+cmd+'*{'+esc(m[2])+'}')
 else:
  t=esc(para);t=re.sub(r'\*\*(.*?)\*\*',r'\\textbf{\1}',t);t=re.sub(r'`([^`]+)`',r'\\texttt{\1}',t);lines.append(t+'\n')
tex=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{parskip}
\usepackage{lmodern}
\begin{document}
'''+ '\n\n'.join(lines)+'\n\\end{document}\n'
(R/'response_to_reviewers.tex').write_text(tex)
print('Response prepared with',len(comments),'verbatim reviewer comments and',len(loc),'resolved location labels')
