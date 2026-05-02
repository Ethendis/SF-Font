import json
def Chara_Match_Pairs():
    with open("gb2312_6763_结构部件.json", encoding='utf-8') as c:
        comp = json.load(c)

    with open("meta/ref.json", encoding='utf-8') as d:
        ref = json.load(d)
        reff = []
        for i in ref:
            for j in ref[i]:
              reff.append(j)

    with open("meta/cref2.json", "w", encoding = 'utf8') as e:
        dict = {}
        for i in comp:
            dict.update({i: []})
            list1 = []
            decompose =[]
            dict2 = {}
            for j in reff:
                dict2.update({j: []})
                list2 = []
                for k in comp[i]:
                    for l in comp[j]:
                        if k == l:
                            list2.append(l)

                            
                for y in range(len(list2)):
                    dict2[j].append(list2[y])
                list2.clear()
            for key,value in dict2.items():
                Ref_Sub_Sel(value,list1,key,decompose,3)
                if len(list1) == 8:
                            break
            if len(list1) < 8:
                for key, value in dict2.items():
                    Ref_Sub_Sel(value,list1,key,decompose,2)
                    if len(list1) == 8:
                            break
            if len(list1) < 8:
                for key, value in dict2.items():
                    Ref_Sub_Sel(value,list1,key,decompose,1)

                    if len(list1) == 8:
                            break
            if len(list1) < 8:
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
        if len(value) >= bujiangeshu:
            list1.append(key)
    else:
        if len(value) == bujiangeshu:
            list1.append(key)
    for c in value:
        decompose.append(c)
if __name__ == '__main__':
    Chara_Match_Pairs()