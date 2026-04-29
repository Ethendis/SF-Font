import json
def Chara_Match_Pairs():
    with open("gb2312_6763_结构部件.json", encoding='utf-8') as c:
        comp = json.load(c)

    with open("meta/ref.json", encoding='utf-8') as d:
        ref = json.load(d)
        reff = []
        for i in ref:
            for j in ref[i]:
              reff.append(j)#去掉字典，为每个汉字

    with open("meta/cref2.json", "w", encoding = 'utf8') as e:
        dict = {}     #内容映射表8-shot版本3
        for i in comp:#i为每个汉字
            dict.update({i: []})
            list1 = []#每个字的映射参考字
            decompose =[]#装一下映射字已有部件
            dict2 = {}
            for j in reff:#遍历参考字
                dict2.update({j: []})
                list2 = []  # 装每个参考字的公共部件
                for k in comp[i]:#第i个字的部件
                    for l in comp[j]:#第j个字的部件
                        if k == l:
                            list2.append(l)

                            # print(list2)
                            # print(len(list2))
                for y in range(len(list2)):
                    dict2[j].append(list2[y])# dict2:{字：部件}
                list2.clear()
            for key,value in dict2.items():
                Ref_Sub_Sel(value,list1,key,decompose,3)
                             # print(c,value)
                             # exit()#力 ['力', '女', '又', '奴']
                if len(list1) == 8:
                            break
            if len(list1) < 8:
                for key, value in dict2.items():
                    Ref_Sub_Sel(value,list1,key,decompose,2)
                    if len(list1) == 8:
                            break
            if len(list1) < 8:#保证每个部件都有参考字
                for key, value in dict2.items():
                    Ref_Sub_Sel(value,list1,key,decompose,1)

                    if len(list1) == 8:
                            break
            if len(list1) < 8:#实在没有就复制
                for h in range(8):
                    list1.append(list1[h])
                    if len(list1) == 8:
                        break
            for x in range(len(list1)):
                dict[i].append(list1[x])
            list1.clear()
            decompose.clear()
            dict2.clear()
        json.dump(dict, e, ensure_ascii=False)

def Ref_Sub_Sel(value,list1,key,decompose,bujiangeshu):
    if bujiangeshu == 3:
        if len(value) >= bujiangeshu:  # 一个的时候条件苛刻一点
            list1.append(key)
    else:
        if len(value) == bujiangeshu:  # 一个的时候条件苛刻一点
            list1.append(key)
    for c in value:
        decompose.append(c)
if __name__ == '__main__':
    Chara_Match_Pairs()