from .cityscapes_classes_labels import labels

_id2name = {label.id: label.name for label in labels}
_color2name = {label.color: label.name for label in labels}
_color2id = {label.color: label.id for label in labels}
_color2trainid = {label.color: label.train_id for label in labels}
_trainid2name = {label.train_id: label.name for label in labels}
_id2color = {label.id: label.color for label in labels}


def id_to_name(id): return _id2name[id]


def color_to_name(color): return _color2name[tuple(color)]


def color_to_id(color): return _color2id[tuple(color)]


def color_to_train_id(color): return _color2trainid[tuple(color)]


def id_to_color(id): return _id2color[id]


def get_number_of_labels(): return len(
    [label for label in labels if label.id >= 0])


def get_number_of_train_labels():
    train_ids = [label.train_id for label in labels]
    return len(set(train_ids))


def get_train_labels():
    labels = [_trainid2name[key] for key in sorted(_trainid2name.keys())]
    return labels[1:]
