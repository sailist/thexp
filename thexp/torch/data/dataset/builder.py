from typing import Callable

from torch.utils.data import Dataset


class placehold:
    def __init__(self, transform=None,
                 reid_func=None,
                 name=None,
                 value=None, source=None):
        self.transform = transform
        self.reid_func = reid_func
        self.name = name
        self.value = value
        self.source = source


class DatasetBuilder(Dataset):
    def __init__(self, xs=None, ys=None, indices=None):
        self.dataset_len = None
        self.ipts = {}
        self.zip = False
        self.indices = indices
        self.placeholds = {'x': [], 'y': []}
        self._name = 'Dataset'
        self._add_id = False
        if xs is not None:
            self.add_inputs(xs)
        if ys is not None:
            self.add_labels(ys)

    def __str__(self):
        if self.zip:
            from thexp.base_classes.attr import attr
            res = attr()
            if self._add_id:
                res['_indexs'] = 'id'
            for ph in self.placeholds['x']:
                res[ph.name] = 'x'
            for ph in self.placeholds['y']:
                res[ph.name] = 'y'

            return str(res)
        else:
            xs = []
            ys = []
            if self._add_id:
                xs.append('indexs')
            for ph in self.placeholds['x']:
                xs.append(ph.name)
            for ph in self.placeholds['y']:
                ys.append(ph.name)
            return str("{}: ({}, {})".format(self._name, ', '.join(xs), ', '.join(ys)))

    __repr__ = __str__

    def name(self, name: str):
        self._name = name

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(list(self.ipts.values())[0])

    def __getitem__(self, index: int):
        if self.indices is not None:
            index = self.indices[index]

        for ph in self.placeholds['x']:  # type:placehold
            source = self.ipts[ph.source]
            if ph.reid_func is not None and callable(ph.reid_func):
                index = ph.reid_func(index, source[index], self)
            x = source[index]
            if ph.transform is not None:
                x = ph.transform(x)
            ph.value = x
        for ph in self.placeholds['y']:
            source = self.ipts[ph.source]
            if ph.reid_func is not None and callable(ph.reid_func):
                index = ph.reid_func(index, source[index], self)
            y = source[index]
            if ph.transform is not None:
                y = ph.transform(y)
            ph.value = y

        if self.zip:
            from thexp.base_classes.attr import attr
            res = attr()
            if self._add_id:
                res['_indexs'] = index

            for ph in self.placeholds['x']:
                res[ph.name] = ph.value
            for ph in self.placeholds['y']:
                res[ph.name] = ph.value

            return res
        else:
            xs = []
            ys = []
            if self._add_id:
                xs.append(index)
            for ph in self.placeholds['x']:
                xs.append(ph.value)
            for ph in self.placeholds['y']:
                ys.append(ph.value)

            return [*xs, *ys]

    def _check_len(self, values):
        ilen = len(values)
        if self.dataset_len is None:
            self.dataset_len = ilen
        else:
            assert self.dataset_len == ilen

    def add_inputs(self, inputs, name='xs'):
        self._check_len(inputs)
        self.ipts[name] = inputs
        return self

    def add_labels(self, labels, name='ys'):
        self._check_len(labels)
        self.ipts[name] = labels
        return self

    def toggle_id(self, toggle=None):
        if toggle is None:
            toggle = not self._add_id
        self._add_id = toggle

    def add_x(self, transform=None, name=None, reid_func: Callable = None, source='xs'):
        if name is None:
            name = 'input_{}'.format(len(self.placeholds['x']))

        self.placeholds['x'].append(placehold(transform, reid_func, name, None, source=source))
        return self

    def add_y(self, transform=None, name=None, reid_func: Callable = None, source='ys'):
        if name is None:
            name = 'label_{}'.format(len(self.placeholds['y']))

        self.placeholds['y'].append(placehold(transform, reid_func, name, None, source=source))
        return self

    def zip_mode(self):
        self.zip = True
        return self

    def chain_mode(self):
        self.chain = True
        return self

    def subset(self, indices):
        self.indices = indices
        return self

    @staticmethod
    def DataLoader_from_dataset(dataset: Dataset, batch_size=1, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=0, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None, multiprocessing_context=None):
        from torch.utils.data import DataLoader
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          sampler=sampler,
                          batch_sampler=batch_sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          timeout=timeout,
                          worker_init_fn=worker_init_fn,
                          multiprocessing_context=multiprocessing_context)

    def DataLoader(self, batch_size=1, shuffle=False, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None, multiprocessing_context=None):
        from torch.utils.data import DataLoader
        return DataLoader(dataset=self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          sampler=sampler,
                          batch_sampler=batch_sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          timeout=timeout,
                          worker_init_fn=worker_init_fn,
                          multiprocessing_context=multiprocessing_context)
