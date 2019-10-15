'''
Format:
'id'
'annotation_id'
'per_annotation_id'
'persona'
'input.keywords'
'input.images'
'input.sentences'
'output.sentences'
'''

import os
import glob
import json
import pprint
from collections import defaultdict,Counter



# show per-persona sentences
def aggregate_data(data, STYLE_ORDER = ['gender', 'country', 'age', 'ethnic', 'education', 'politics', 'tod'], verbose=False):
    # persona dictionaries
    combined_dict = defaultdict(lambda: defaultdict(list))
    controlled_dict = defaultdict(list)

    for pid,obj in enumerate(data):
        sents = obj['output.sentences']
        sents = [sents]
        styles = obj['persona']

        # (1) combined setting dictionary
        for s,v in styles.items():
            if v.strip() == '':
                v = 'Empty'
            combined_dict[s][v] += sents

        # (2) controlled setting dictionary
        style_ordered_tuple = tuple([styles[s] for s in STYLE_ORDER])
        assert len(style_ordered_tuple) == len(STYLE_ORDER)
        controlled_dict[style_ordered_tuple] += sents

    if verbose:
        # combined setting
        print ('Combined dictionary',combined_dict.keys())
        print ('Combined Combinations',' '.join(
            ['%s:%d'%(k,len(v)) for k,v in combined_dict.items()]))
        for style, value_dic in combined_dict.items():
            print ('\t target STYLE: %s' % (style))
            print ('\t\t unique value types: %d'%(len(value_dic)))
            print ('\t\t value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()]))
        print()

        # controlled setting
        print ('Controlled dictionary: total combination of styles=',len(controlled_dict))
        for style_order, value_dic in list(controlled_dict.items())[:5]:
            print('\t combination:',style_order, len(value_dic))
        print('\t ...')
    return combined_dict, controlled_dict


def read_dataset(data_dir, level):
    data = []
    for dt in ['train','valid','test']:
        d = []
        files = glob.glob(os.path.join(data_dir, level, dt)+'/*.json')
        for file in files:
            with open(file) as fin:
                obj = dict(json.load(fin))
                obj['filename'] = os.path.basename(file)
                d.append(obj)
        print('Number of files/loaded-data: {}/{} in {}'.format(len(files),len(d), dt))
        data.append(d)

    train,valid,test = tuple(data)
    return train, valid, test

def write_dataset(data_dir, level, data):
    assert len(data) == 3
    for data_per_type, datatype in zip(data, ['train','valid','test']):
        dir = os.path.join(data_dir, level, datatype)
        if not os.path.exists(dir):
            os.makedirs(dir)

        for obj in data_per_type:
            with open(os.path.join(dir , obj['filename']), 'w') as f:
                json.dump(obj, f)

        print('Number of files written: {} in {}'.format(len(data_per_type), dir))
    return



def main(data_dir = '../../data/v2/', level = 'sentences', verbose=False):
    train, valid, test = read_dataset(data_dir, level)

    # show some samples
    if verbose:
        print(train[-1])

    combined_dict, controlled_dict = aggregate_data(train+valid+test, verbose=True)



if __name__ == '__main__': main()

