from typing_extensions import Self
from django.shortcuts import render
from django.views import View
# from .models import AI
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest

import numpy as np
import itertools
import torch

from konlpy.tag import Okt, Komoran
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from googletrans import Translator
from textrank import KeysentenceSummarizer
# from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# Create your views here.


class KoGPT2Comment(LightningModule):
    def __init__(self):
        super(KoGPT2Comment, self).__init__()
        self.kogpt2 = GPT2LMHeadModel.from_pretrained(
            'skt/kogpt2-base-v2')  # pretrained KoGPT2 model

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def comment(self, keysent):
        # KoGPT2 모델에서 사용하는 특수 토큰
        U_TKN = '<usr>'  # 일기 핵심 문장의 시작을 나타내는 특수 토큰
        S_TKN = '<sys>'  # 코멘트의 시작을 나타내는 특수 토큰
        BOS = '</s>'    # 문장의 시작을 나타내는 특수 토큰
        EOS = '</s>'    # 문장의 끝을 나타내는 특수 토큰
        UNK = '<unk>'   # 어휘에 없는 토큰을 나타내는 특수 토큰
        MASK = '<unused0>'  # 마스킹된 토큰을 나타내는 특수 토큰
        SENT = '<unused1>'  # 일기 핵심 문장의 끝을 나타내는 특수 토큰
        PAD = '<pad>'   # 토큰 배열을 동일한 크기로 만드는데 사용되는 특수 토큰
        tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                      bos_token=BOS, eos_token=EOS, unk_token=UNK,
                                                      pad_token=PAD, mask_token=MASK)
        sent = '0'

        with torch.no_grad():
            q = keysent
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(
                    U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
            return a.strip()


def komoran_tokenizer(sent):
    komoran = Komoran()
    words = komoran.pos(sent, join=True)
    words = [w for w in words if (
        '/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words


class aiView(View):
    def get_emotion(doc):
        classifier = pipeline('zero-shot-classification',
                              model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0)
        out = classifier(doc, ["love", "joy", "surprise", "anger",
                               "sadness", "fear", "neutral", "tired"])
        emotion = []

        for i in range(8):
            if out['scores'][i] > 0.4:
                emotion.append(out['labels'][i])

        if len(emotion) == 0:
            emotion.append(out['labels'][0])
            emotion.append(out['labels'][1])

        if(len(emotion) > 1):
            # 창을 띄워서 사용자가 감정을 선택하는 코드로 바꾸기
            print(emotion)
            emotion_idx = int(
                input('여러 개의 감정이 느껴지네요! 오늘을 대표하는 감정 1개를 선택해주세요(0, 1, 2, ...의 index로 입력):'))
        else:
            emotion_idx = 0

        emo = emotion[emotion_idx]
        # print(EMOTION)

        return emo

    def comment_emo(emotion):
        # 감정에 따른 문구 출력
        if (emotion == 'joy'):
            emotion_comment = '오늘은 행복한 하루였군요!'
        elif (emotion == 'love'):
            emotion_comment = '당신의 하루에서 사랑이 느껴지네요.'
        elif (emotion == 'surprise'):
            emotion_comment = '오늘은 놀라운 일이 있었군요.'
        elif (emotion == 'anger'):
            emotion_comment = '오늘은 화가 많이 났던 하루였군요.'
        elif (emotion == 'sadness'):
            emotion_comment = '오늘은 조금 슬픈 하루였군요.'
        elif (emotion == 'fear'):
            emotion_comment = '오늘은 조금 무서웠던 일이 있었군요.'
        elif (emotion == 'neutral'):
            emotion_comment = '당신의 하루에서 평온함이 느껴지네요.'
        else:
            emotion_comment = '오늘은 조금 지치는 하루였군요.'

        return emotion_comment

    def keyword_extract(doc):
        def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
            word_doc_similarity = cosine_similarity(
                candidate_embeddings, doc_embedding)
            word_similarity = cosine_similarity(candidate_embeddings)
            keywords_idx = [np.argmax(word_doc_similarity)]
            candidates_idx = [i for i in range(
                len(words)) if i != keywords_idx[0]]

            for _ in range(top_n - 1):
                candidate_similarities = word_doc_similarity[candidates_idx, :]
                target_similarities = np.max(
                    word_similarity[candidates_idx][:, keywords_idx], axis=1)

                mmr = (1-diversity) * candidate_similarities - \
                    diversity * target_similarities.reshape(-1, 1)
                mmr_idx = candidates_idx[np.argmax(mmr)]

                keywords_idx.append(mmr_idx)
                candidates_idx.remove(mmr_idx)
            return [words[idx] for idx in keywords_idx]

        okt = Okt()
        tokenized_doc = okt.pos(doc)
        tokenized_nouns = ' '.join([word[0]
                                    for word in tokenized_doc if word[1] == 'Noun'])
        n_gram_range = (1, 1)
        count = CountVectorizer(
            ngram_range=n_gram_range).fit([tokenized_nouns])
        candidates = count.get_feature_names_out()
        model = SentenceTransformer(
            'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)
        top_n = 5
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index]
                    for index in distances.argsort()[0][-top_n:]]

        keyword = mmr(doc_embedding, candidate_embeddings,
                      candidates, top_n=1, diversity=0.4)

        translator = Translator()
        key = translator.translate(keyword[0], dest='en').text

        return key

    def keySentence(doc):
        komoran = Komoran()
        summarizer = KeysentenceSummarizer(
            tokenize=komoran_tokenizer,
            min_sim=0.3,
            verbose=False
        )
        sents_list = doc.split('.' or '?' or '!')
        sents = []
        for sent in sents_list:
            if sent.strip() is not '':
                sents.append(sent.strip())

        bias = np.ones(len(sents))
        bias[-1] = 5

        keysents_list = summarizer.summarize(sents, topk=3, bias=bias)
        keysents = []
        for _, _, sent in keysents_list:
            keysents.append(sent+'.')

        return keysents

    def comment_moon(keysents):
        # KoGPT2에서 제공하는 토큰나이저 사용
        model = KoGPT2Comment()
        model.load_state_dict(torch.load("model_chp/comment_model.pth"))
        model.eval()

        moon_comment = ''
        max_comment = ''
        comment = ''
        for sent in keysents:
            comment = model.comment(sent)
            if(comment[-1] is not '?' and (not '.' or not '!')):
                comment = comment + '.'
            if(len(max_comment) <= len(comment)):
                max_comment = comment
            if(len(comment) > 8 and '?' not in comment and '저' not in comment):
                moon_comment = comment
                break
        if(moon_comment is ''):
            moon_comment = max_comment

        return moon_comment  # database에 moon_comment 저장하는 코드 추가하기

    def ai(self, request):
        doc = """
            인공지능 중간고사를 친 다음날, 할게 없어서 북한산에 등산을 하러 갔다.
            등산은 정말 오랜만이었는데, 가는 길은 험난했지만 백운대에 도착하니 너무 뿌듯하고 개운했다.
            방안에 틀어박히기 보다 나와서 운동을 했더니 기분이 너무 좋았다.
            앞으로 시간이 날 때마다 자주 산에 와야겠다.
        """
        if doc is None:
            return HttpResponseBadRequest()
        else:
            emotion = get_emotion(doc)
            comm_emo = comment_emo(emotion)
            keyW = keyword_extract(doc)
            keyS = keySentence(doc)
            comm_moon = comment_moon(keyS)
            comm = comm_emo + comm_moon
            print(emotion)
            print(comm)
            print(keyW)
            print(keyS)
