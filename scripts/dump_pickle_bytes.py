from pathlib import Path
p=Path('fraud_model.pkl').read_bytes()
s=b'StandardScaler'
i=p.find(s)
print('index',i)
if i!=-1:
    start=max(0,i-120)
    end=min(len(p),i+120)
    print(p[start:end])
else:
    print('StandardScaler not found as raw bytes')
