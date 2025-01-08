from collections import namedtuple

# from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# 定义基础运算符
PRIMITIVES = [
    # 'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',  # 3x3的分离卷积
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

eg_nas_cifar10 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2),
            ('skip_connect', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
eg_nas_imagenet = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 2),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3),
            ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

eg_nas_cifar10 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3),
            ('max_pool_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
eg_nas_imagenet = Genotype(
    normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1),
            ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

cif10_ps25 = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0),
            ('skip_connect', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

cif10_ps50 = Genotype(
    normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 3),
            ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

cif10_ps70 = Genotype(
    normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2),
            ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))

cif10_ps501 = Genotype(
    normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

cif10_ps502 = Genotype(
    normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 0),
            ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
            ('sep_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

cif100_1 = Genotype(
    normal=[('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1),
            ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2),
            ('avg_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

cif100_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2),
            ('skip_connect', 3), ('avg_pool_3x3', 4), ('dil_conv_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

l31 = Genotype(
    normal=[('dil_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
l32 = Genotype(
    normal=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3),
            ('dil_conv_3x3', 2), ('avg_pool_3x3', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

l33 = Genotype(
    normal=[('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 1), ('avg_pool_3x3', 4), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

l34 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('skip_connect', 3), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l21 = Genotype(
    normal=[('dil_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0),
            ('skip_connect', 3), ('avg_pool_3x3', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l22 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 1), ('dil_conv_5x5', 1),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1),
            ('dil_conv_5x5', 3), ('avg_pool_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l23 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1),
            ('max_pool_3x3', 3), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

l24 = Genotype(
    normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('skip_connect', 3), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2),
            ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
l11 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 3),
            ('sep_conv_5x5', 2), ('avg_pool_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

l12 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l13 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

l14 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 3),
            ('sep_conv_5x5', 1), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

l41 = Genotype(
    normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3),
            ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

l42 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 3),
            ('dil_conv_5x5', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('dil_conv_5x5', 0), ('max_pool_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l43 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3),
            ('sep_conv_5x5', 2), ('avg_pool_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

l44 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 2), ('avg_pool_3x3', 4), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1),
            ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

eg_nas = eg_nas_cifar10
