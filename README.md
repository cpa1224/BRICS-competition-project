# RAG鏅鸿兘闂瓟绯荤粺

鍩轰簬Ollama鏈湴澶фā鍨嬨€丩angChain妗嗘灦鍜孲treamlit鏋勫缓鐨勬櫤鑳介棶绛旂郴缁燂紝鑳藉"瀛︿範"鎸囧畾鏂囨。骞跺洖绛旂浉鍏抽棶棰樸€?
## 椤圭洰鍔熻兘

- 鏀寔涓婁紶PDF銆丏OCX銆乀XT绛夋牸寮忔枃妗?- 鑷姩杩涜鏂囨。瑙ｆ瀽銆佹枃鏈垎鍧楀拰鍚戦噺鍖?- 浣跨敤Chroma鍚戦噺鏁版嵁搴撳瓨鍌ㄦ枃妗ｅ悜閲?- 鍩轰簬妫€绱㈠寮虹敓鎴愶紙RAG锛夋妧鏈繘琛岄棶绛?- 鏀寔澶氳疆瀵硅瘽锛屽叿鏈変細璇濊蹇嗗姛鑳?- 瀵规棤鍏抽棶棰樿兘姝ｇ‘鎷掔瓟

## 鐜瑕佹眰

- Python 3.8+
- Ollama锛堢敤浜庤繍琛屾湰鍦板ぇ妯″瀷锛?- 鑷冲皯8GB鍐呭瓨锛堟帹鑽?6GB浠ヤ笂锛?
## 瀹夎姝ラ

### 1. 瀹夎Ollama

璁块棶 [Ollama瀹樻柟缃戠珯](https://ollama.com/) 涓嬭浇骞跺畨瑁匫llama銆?
### 2. 涓嬭浇澶фā鍨?
```bash
# 涓嬭浇deepseek-r1:7b妯″瀷锛堟帹鑽愶級
ollama pull deepseek-r1:7b

# 鎴栬€呬笅杞絨wen2:7b妯″瀷
ollama pull qwen2:7b

# 涓嬭浇宓屽叆妯″瀷
ollama pull nomic-embed-text
```

### 3. 瀹夎Python渚濊禆

```bash
pip install -r requirements.txt
```

## 浣跨敤璇存槑

### 杩愯Web搴旂敤

```bash
streamlit run streamlit_app.py
```

### 浣跨敤娴佺▼

1. 鍦ㄦ祻瑙堝櫒涓墦寮€搴旂敤锛堥€氬父鏄?http://localhost:8501锛?2. 鍦ㄤ晶杈规爮涓婁紶PDF銆丏OCX鎴朤XT鏍煎紡鐨勬枃妗?3. 鐐瑰嚮"鏋勫缓鐭ヨ瘑搴?鎸夐挳锛岀瓑寰呮枃妗ｅ鐞嗗畬鎴?4. 鍦ㄩ棶绛斾氦浜掑尯杈撳叆闂锛岀偣鍑?鎻愰棶"鎸夐挳鑾峰彇绛旀
5. 鏀寔澶氳疆瀵硅瘽锛岀郴缁熶細璁颁綇瀵硅瘽鍘嗗彶

### 鍛戒护琛屾祴璇?
```bash
python test_rag.py
```

## 鍏抽敭鎶€鏈偣

### RAG娴佺▼

1. **鏂囨。鍔犺浇**锛氭敮鎸丳DF銆丏OCX銆乀XT绛夋牸寮忔枃妗ｇ殑璇诲彇
2. **鏂囨湰鍒嗗潡**锛氫娇鐢≧ecursiveCharacterTextSplitter杩涜鍒嗗潡锛坈hunk_size=1000, chunk_overlap=200锛?3. **鍚戦噺鍖?*锛氫娇鐢∣llama鐨刵omic-embed-text妯″瀷灏嗘枃鏈潡杞崲涓哄悜閲?4. **鍚戦噺瀛樺偍**锛氫娇鐢–hroma鍚戦噺鏁版嵁搴撳瓨鍌ㄥ拰妫€绱㈠悜閲?5. **闂瓟鐢熸垚**锛氫娇鐢–onversationalRetrievalChain杩炴帴妫€绱㈠櫒鍜屽ぇ妯″瀷

### 鎵€鐢ㄦā鍨?
- **澶ц瑷€妯″瀷**锛歞eepseek-r1:7b 鎴?qwen2:7b
- **宓屽叆妯″瀷**锛歯omic-embed-text

### 绯荤粺鎻愮ず璇嶈璁?
绯荤粺鎻愮ず璇嶈姹傛ā鍨嬶細
- 涓ユ牸鍩轰簬鎻愪緵鐨勫弬鑰冩枃妗ｅ洖绛旈棶棰?- 鑻ユ枃妗ｄ腑娌℃湁鐩稿叧淇℃伅锛屾槑纭"鏂囨。涓湭鎵惧埌鐩稿叧绛旀"
- 涓嶄娇鐢ㄦ枃妗ｅ鐨勭煡璇?- 淇濇寔鍥炵瓟绠€娲佺浉鍏?
## 椤圭洰缁撴瀯

```
RAG-QA-System/
鈹溾攢鈹€ streamlit_app.py      # Streamlit Web搴旂敤涓绘枃浠?鈹溾攢鈹€ document_processor.py # 鏂囨。澶勭悊妯″潡
鈹溾攢鈹€ rag_chain.py          # RAG闂瓟閾炬ā鍧?鈹溾攢鈹€ test_rag.py           # 鍛戒护琛屾祴璇曡剼鏈?鈹溾攢鈹€ requirements.txt      # 渚濊禆鍖呭垪琛?鈹溾攢鈹€ .gitignore            # Git蹇界暐閰嶇疆
鈹斺攢鈹€ documents/            # 绀轰緥鏂囨。鐩綍
    鈹溾攢鈹€ nlp_introduction.txt
    鈹溾攢鈹€ transformer.txt
    鈹溾攢鈹€ word_embedding.txt
    鈹溾攢鈹€ bert.txt
    鈹斺攢鈹€ text_classification.txt
```

## 宸茬煡闂涓庢敼杩涙柟鍚?
### 宸茬煡闂
- Ollama鏈嶅姟闇€瑕侀鍏堝惎鍔?- 妯″瀷涓嬭浇闇€瑕佷竴瀹氭椂闂村拰缃戠粶甯﹀
- 棣栨鏋勫缓鐭ヨ瘑搴撳彲鑳介渶瑕佽緝闀挎椂闂?
### 鏀硅繘鏂瑰悜
- 鏀寔鏇村鏂囨。鏍煎紡锛堝PPT銆丒xcel锛?- 娣诲姞鏂囨。绠＄悊鍔熻兘锛堝垹闄ゃ€佹洿鏂版枃妗ｏ級
- 鏀寔鎵归噺涓婁紶鏂囨。
- 娣诲姞澶滈棿妯″紡
- 鏀寔瀵煎嚭闂瓟璁板綍
- 浼樺寲闀挎枃鏈鐞嗚兘鍔?
## 璁稿彲璇?
MIT License