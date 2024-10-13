import os

_id2label = {
    0: "отказ",
    1: "отмена",
    2: "подтверждение",
    3: "начать осаживание",
    4: "осадить на (количество) вагон",
    5: "продолжаем осаживание",
    6: "зарядка тормозной магистрали",
    7: "вышел из межвагонного пространства",
    8: "продолжаем роспуск",
    9: "растянуть автосцепки",
    10: "протянуть на (количество) вагон",
    11: "отцепка",
    12: "назад на башмак",
    13: "захожу в межвагонное,пространство",
    14: "остановка",
    15: "вперед на башмак",
    16: "сжать автосцепки",
    17: "назад с башмака",
    18: "тише",
    19: "вперед с башмака",
    20: "прекратить зарядку тормозной магистрали",
    21: "тормозить",
    22: "отпустить",
}

def id2label(id: int) -> str:
    global _id2label
    return _id2label[id]

def get_labels():
    return list(_id2label.keys())

def get_path(*args):
    """Return the path to a file in the main directory."""
    if not args or not all(isinstance(arg, str) for arg in args):
        raise ValueError("At least one string argument is required")
    
    return os.path.join(os.path.dirname(__file__)[:-4], *args)
