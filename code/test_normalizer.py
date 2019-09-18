# %load_ext autoreload
# %autoreload 2
import sys
import numpy as np
import pandas as pd
from normalizer import fit_standard_scaler_to_sequence_batch, normalize_sequence_batch

def test_fit_standard_scaler_to_sequence_batch():

    X = np.array([[[0.9180555898766518, -0.1124999994242935, 0.5097222514293852],
            [0.9111111304603812, -0.09305556168259389, 0.5375000404706096],
            [0.8819444981597608, -0.0861111144222878, 0.5138889270791476],
            [0.8819444981597608, -0.0861111144222878, 0.5138889270791476],
            [0.8791667143932526, -0.1000000028649177, 0.5055555757796229],
            [0.8791667143932526, -0.1000000028649177, 0.5055555757796229],
            [0.8888889575760315, -0.105555564319952, 0.5125000351958935],
            [0.8625000117942031, -0.1013888947481719, 0.5097222514293852],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8541666604946784, -0.1083333359304957, 0.5277777972878307],
            [0.8513888767281701, -0.1013888947481719, 0.5527778025625468]],

           [[0.8583333361444407, -0.1083333359304957, 0.5624999971214676],
            [0.8541666604946784, -0.1152777831908018, 0.5486111269127845],
            [0.8513888767281701, -0.1097222278137498, 0.5500000187960387],
            [0.8513888767281701, -0.1097222278137498, 0.5500000187960387],
            [0.8708333630937278, -0.08750000022755966, 0.5680555646544842],
            [0.8583333361444407, -0.0791666671619817, 0.5708333484209924],
            [0.8319444876103286, -0.08472222253903368, 0.5597222619788175],
            [0.8208333525442956, -0.1041666724366978, 0.5263889054045766],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1013888947481719, 0.577777807837263],
            [0.8291667038438203, -0.1111111196970039, 0.5847222672535336]],

           [[0.8430556226763616, -0.1069444440472416, 0.5861111105129296],
            [0.8347222713768369, -0.1013888947481719, 0.5624999971214676],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8277778119605662, -0.1041666724366978, 0.5708333484209924],
            [0.8333333794935827, -0.1111111196970039, 0.5750000240707548],
            [0.8319444876103286, -0.1013888947481719, 0.5624999971214676],
            [0.8333333794935827, -0.1013888947481719, 0.5583333700955634],
            [0.8305555957270745, -0.1138888913075477, 0.5583333700955634],
            [0.8250000281940579, -0.1111111196970039, 0.5694444565377382]]])

    means_expected = np.array([ 0.8482194 , -0.10306268,  0.54405273])
    variances_expected = np.array([6.7752e-04, 6.9760e-05, 6.3776e-04])

    means, variances = fit_standard_scaler_to_sequence_batch(X)
    assert((means_expected == np.round(means, 8)).all())
    assert((variances_expected == np.round(variances, 8)).all())

def test_normalize_sequence_batch():
    X = np.array([[[0.9180555898766518, -0.1124999994242935, 0.5097222514293852],
            [0.9111111304603812, -0.09305556168259389, 0.5375000404706096],
            [0.8819444981597608, -0.0861111144222878, 0.5138889270791476],
            [0.8819444981597608, -0.0861111144222878, 0.5138889270791476],
            [0.8791667143932526, -0.1000000028649177, 0.5055555757796229],
            [0.8791667143932526, -0.1000000028649177, 0.5055555757796229],
            [0.8888889575760315, -0.105555564319952, 0.5125000351958935],
            [0.8625000117942031, -0.1013888947481719, 0.5097222514293852],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8611111199109489, -0.1041666724366978, 0.5013889001298605],
            [0.8541666604946784, -0.1083333359304957, 0.5277777972878307],
            [0.8513888767281701, -0.1013888947481719, 0.5527778025625468]],

           [[0.8583333361444407, -0.1083333359304957, 0.5624999971214676],
            [0.8541666604946784, -0.1152777831908018, 0.5486111269127845],
            [0.8513888767281701, -0.1097222278137498, 0.5500000187960387],
            [0.8513888767281701, -0.1097222278137498, 0.5500000187960387],
            [0.8708333630937278, -0.08750000022755966, 0.5680555646544842],
            [0.8583333361444407, -0.0791666671619817, 0.5708333484209924],
            [0.8319444876103286, -0.08472222253903368, 0.5597222619788175],
            [0.8208333525442956, -0.1041666724366978, 0.5263889054045766],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1097222278137498, 0.5375000404706096],
            [0.8125000498686289, -0.1013888947481719, 0.577777807837263],
            [0.8291667038438203, -0.1111111196970039, 0.5847222672535336]],

           [[0.8430556226763616, -0.1069444440472416, 0.5861111105129296],
            [0.8347222713768369, -0.1013888947481719, 0.5624999971214676],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8305555957270745, -0.1041666724366978, 0.554166694445801],
            [0.8277778119605662, -0.1041666724366978, 0.5708333484209924],
            [0.8333333794935827, -0.1111111196970039, 0.5750000240707548],
            [0.8319444876103286, -0.1013888947481719, 0.5624999971214676],
            [0.8333333794935827, -0.1013888947481719, 0.5583333700955634],
            [0.8305555957270745, -0.1138888913075477, 0.5583333700955634],
            [0.8250000281940579, -0.1111111196970039, 0.5694444565377382]]])

    means = np.array([ 0.8482194 , -0.10306268,  0.54405273])
    variances = np.array([6.7752e-04, 6.9760e-05, 6.3776e-04])

    X_norm = normalize_sequence_batch(X, means, variances)

    X_norm_expected = np.array([[[ 2.68299261, -1.12991408, -1.35941237],
        [ 2.4161978 ,  1.19813512, -0.25947227],
        [ 1.29566332,  2.02958189, -1.19442107],
        [ 1.29566332,  2.02958189, -1.19442107],
        [ 1.18894539,  0.36668908, -1.52440368],
        [ 1.18894539,  0.36668908, -1.52440368],
        [ 1.56245813, -0.29846877, -1.24941817],
        [ 0.54863783,  0.20039944, -1.35941237],
        [ 0.49527887, -0.13217912, -1.68939498],
        [ 0.49527887, -0.13217912, -1.68939498],
        [ 0.49527887, -0.13217912, -1.68939498],
        [ 0.22848406, -0.6310466 , -0.64445198],
        [ 0.12176613,  0.20039944,  0.34549392]],

       [[ 0.38856095, -0.6310466 ,  0.7304717 ],
        [ 0.22848406, -1.46249336,  0.18050262],
        [ 0.12176613, -0.79733624,  0.23549972],
        [ 0.12176613, -0.79733624,  0.23549972],
        [ 0.86879161,  1.86329297,  0.95046011],
        [ 0.38856095,  2.86102865,  1.06045431],
        [-0.62525561,  2.19587153,  0.62047943],
        [-1.05212732, -0.13217912, -0.69944908],
        [-1.37227923, -0.79733624, -0.25947227],
        [-1.37227923, -0.79733624, -0.25947227],
        [-1.37227923, -0.79733624, -0.25947227],
        [-1.37227923,  0.20039944,  1.33543982],
        [-0.73197354, -0.96362589,  1.61042533]],

       [[-0.19838391, -0.46475695,  1.6654205 ],
        [-0.51853769,  0.20039944,  0.7304717 ],
        [-0.67861458, -0.13217912,  0.40049102],
        [-0.67861458, -0.13217912,  0.40049102],
        [-0.67861458, -0.13217912,  0.40049102],
        [-0.67861458, -0.13217912,  0.40049102],
        [-0.67861458, -0.13217912,  0.40049102],
        [-0.7853325 , -0.13217912,  1.06045431],
        [-0.57189665, -0.96362589,  1.22544562],
        [-0.62525561,  0.20039944,  0.7304717 ],
        [-0.57189665,  0.20039944,  0.56548233],
        [-0.67861458, -1.29620372,  0.56548233],
        [-0.89205043, -0.96362589,  1.00545721]]])

    assert((X_norm_expected == np.round(X_norm, 8)).all())