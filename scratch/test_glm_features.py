import pandas as pd, numpy as np, re
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')
sub = train[train['model'].isin(['glm-5.3', 'glm-5.3-flash'])].copy().reset_index(drop=True)
y = (sub['model'] == 'glm-5.3-flash').astype(int).values
groups = sub['prompt_id'].values

MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')

def extract(t):
    s = WRE.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\n' for c in t)/nt]
    base += [min(t.count(m)/n, 3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)

X_raw = np.stack([extract(t) for t in sub['text']])

# Also prompt-relative z-score within the pair:
pids = sub['prompt_id'].values
X_z = np.zeros_like(X_raw)
for p in np.unique(pids):
    idx = np.where(pids == p)[0]
    s = X_raw[idx]
    m = s.mean(axis=0)
    st = np.maximum(s.std(axis=0), 1e-4)
    X_z[idx] = (s - m) / st

X_feat = np.column_stack([X_raw, X_z])

gkf = GroupKFold(n_splits=5)
for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]:
    oof = np.zeros(len(sub))
    for tr, va in gkf.split(X_feat, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        clf.fit(X_feat[tr], y[tr])
        oof[va] = clf.predict_proba(X_feat[va])[:, 1]
    acc = accuracy_score(y, oof > 0.5)
    print(f'GLM vs GLM-Flash C={C}: CV Acc = {acc*100:.2f}%')

