import json
from typing import List, Set, Dict

def load_char_components(json_path: str) -> Dict[str, Set[str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    char_to_comp = {}
    for char, comps in data.items():
        if isinstance(comps, str):
            comps = [comps]
        char_to_comp[char] = set(comps)
    return char_to_comp

def greedy_set_cover(char_to_comp: Dict[str, Set[str]]) -> List[str]:
    all_components = set()
    for comps in char_to_comp.values():
        all_components.update(comps)

    uncovered = set(all_components)
    selected_chars = []

    while uncovered:
        best_char = None
        best_cover = set()
        for char, comps in char_to_comp.items():
            cover = comps & uncovered
            if len(cover) > len(best_cover):
                best_cover = cover
                best_char = char
        if not best_char:
            break
        selected_chars.append(best_char)
        uncovered -= best_cover
    return selected_chars

def exact_set_cover_pulp(char_to_comp: Dict[str, Set[str]]) -> List[str]:
    try:
        import pulp
    except ImportError:
        raise ImportError("pip install pulp")

    all_components = set()
    for comps in char_to_comp.values():
        all_components.update(comps)
    comp_list = list(all_components)

    prob = pulp.LpProblem("MinCharCover", pulp.LpMinimize)
    char_vars = {char: pulp.LpVariable(f"x_{char}", cat='Binary') for char in char_to_comp}
    prob += pulp.lpSum(char_vars.values())
    for comp in comp_list:
        covering_chars = [char_vars[char] for char, comps in char_to_comp.items() if comp in comps]
        prob += pulp.lpSum(covering_chars) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [char for char, var in char_vars.items() if var.varValue == 1]
    return selected

if __name__ == "__main__":
    json_file = "gb2312_6763_结构部件.json"
    use_exact = True

    char_to_comp = load_char_components(json_file)

    if use_exact:
        try:
            ref_list = exact_set_cover_pulp(char_to_comp)
        except ImportError:
            ref_list = greedy_set_cover(char_to_comp)
    else:
        ref_list = greedy_set_cover(char_to_comp)


    # 保存 ref 列表为 JSON
    ref_output = {"ref": ref_list}
    with open("ref.json", "w", encoding="utf-8") as f:
        json.dump(ref_output, f, ensure_ascii=False, indent=2)


    # 保存 ref 列表为纯文本（直接拼接字符）
    with open("ref.txt", "w", encoding="utf-8") as f:
        f.write("".join(ref_list))