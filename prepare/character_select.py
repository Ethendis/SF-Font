import json

def Read_component():
    with open("gb2312_6763_结构部件.json", encoding='utf-8') as c:
        comp = json.load(c)
        print("Length : %d" % len(comp))
    return comp

def Compare_component(comp):
    with open("meta/ref.json", "w", encoding='utf-8') as e:
     with open("meta/decompose.json", "w", encoding='utf-8') as f:
        ref = []
        decompose = []


        for i in comp:
            if len(comp[i]) >= 2:
                print(i)

            for j in comp[i]:
                if j not in decompose:
                   ref.append(i)
                   decompose.append(j)
                decompose = list(set(decompose))
                ref = list(set(ref))

        print(len(ref))
        print(len(decompose))
    return ref,decompose

def Save_R(dict,dec):
    with open("meta/ref.json", "w", encoding='utf-8') as e:
        with open("meta/decompose.json", "w", encoding='utf-8') as f:
            dict = {"ref": ref}
            json.dump(dict, e ,ensure_ascii=False)
            dec = {"decompose": decompose}
            json.dump(dec, f)



if __name__ == '__main__':
    comp = Read_component()
    ref,decompose = Compare_component(comp)
    Save_R(ref,decompose)


