from thexp import Params

p = Params().from_json('pj')
print(p)
print(dict(p.inner_dict.raw_items()))