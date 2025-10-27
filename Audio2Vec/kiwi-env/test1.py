# kiwi_test.py
from kiwipiepy import Kiwi

kiwi = Kiwi()
tokens = kiwi.tokenize("하늘을 나는 자동차")

for token in tokens:
    print(f"{token.form} ({token.tag})")
# 형태소 분석 결과를 '단어/태그' 형식으로 정리
#token_str = " ".join([f"{token.form}/{token.tag}" for token in tokens])
