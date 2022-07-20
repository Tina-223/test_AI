import ai

doc = """
    인공지능 중간고사를 친 다음날, 할게 없어서 북한산에 등산을 하러 갔다.
    등산은 정말 오랜만이었는데, 가는 길은 험난했지만 백운대에 도착하니 너무 뿌듯하고 개운했다.
    방안에 틀어박히기 보다 나와서 운동을 했더니 기분이 너무 좋았다.
    앞으로 시간이 날 때마다 자주 산에 와야겠다.
"""

# doc = """
#     오늘은 내 생일이야. 정말 너무 행복한 날이야. 생일에 친구들에게 많은 축하를 받는다는게 너무 좋은 것 같아.
#     매일매일이 내 생일 같았으면 좋겠다.
# """

choice = int(input('1, 2, 3 골라'))

if choice == 1:
    emo, comm_emo = ai.run_emotion(doc)
    print(emo, comm_emo)
elif choice == 2:
    comm = ai.run_comment(doc)
    print(comm)
else:
    a, b = ai.run_pixray(doc)
    print(a, b)
