import os
import math
import numpy as np
import pandas as pd
from models.predictor import predict
from models.sub_predictor import sub_predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter


def execute(train, validation, test, rtrain, params, model, source=None, problem=None, measure='Cosine', gpu_on=True,
            analytical=False, folder='latent', scene='r', is_topK=False):
    progress = WorkSplitter()

    columns = ['model', 'rank', 'alpha', 'lambda', 'lambda2', 'iter', 'similarity', 'corruption', 'root',
               'batch_size', 'learning_rate', 'source']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(source, str):
        if os.path.isfile('{2}/{3}_U_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):

            RQ = np.load('{2}/{3}_U_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            Y = np.load('{2}/{3}_V_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))

            if os.path.isfile(
                    '{2}/{3}_B_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):
                yBias = np.load(
                    '{2}/{3}_B_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            else:
                yBias = None

            if os.path.isfile(
                    '{2}/{3}_tU_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):
                X = np.load(
                    '{2}/{3}_tU_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            else:
                X = None

            if os.path.isfile(
                    '{2}/{3}_tV_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):
                xBias = np.load(
                    '{2}/{3}_tV_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            else:
                xBias = None
        else:
            RQ, X, xBiast, Yt, yBias = model(train,
                                             validation,
                                             embeded_matrix=np.empty(0),
                                             matrix_utrain=rtrain,
                                             iteration=params['iter'],
                                             rank=params['rank'],
                                             batch_size=params['batch_size'],
                                             learning_rate=params['learning_rate'],
                                             lam=params['lambda'],
                                             lam2=params['lambda2'],
                                             alpha=params['alpha'],
                                             corruption=params['corruption'],
                                             root=params['root'],
                                             source=source,
                                             problem=problem,
                                             gpu_on=gpu_on,
                                             scene=scene,
                                             is_topK=is_topK)
            Y = Yt.T

            np.save('{2}/{3}_U_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), RQ)
            np.save('{2}/{3}_V_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), Y)
            if yBias is not None:
                np.save('{2}/{3}_B_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), yBias)
            if X is not None:
                np.save('{2}/{3}_tU_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), X)
            if xBiast is not None:
                xBias = xBiast.T
                np.save('{2}/{3}_tV_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), xBias)

    else:
        if os.path.isfile('{2}/U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):

            RQ = np.load('{2}/U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
            Y = np.load('{2}/V_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))

            if os.path.isfile('{2}/B_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):
                yBias = np.load('{2}/B_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
            else:
                yBias = None

            if os.path.isfile('{2}/tU_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):
                X = np.load('{2}/tU_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
            else:
                X = None

            if os.path.isfile('{2}/tV_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):
                xBias = np.load('{2}/tV_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
            else:
                xBias = None
        else:
            RQ, X, xBiast, Yt, yBias = model(train,
                                             validation,
                                             embeded_matrix=np.empty(0),
                                             matrix_utrain=rtrain,
                                             iteration=params['iter'],
                                             rank=params['rank'],
                                             batch_size=params['batch_size'],
                                             learning_rate=params['learning_rate'],
                                             lam=params['lambda'],
                                             lam2=params['lambda2'],
                                             alpha=params['alpha'],
                                             corruption=params['corruption'],
                                             root=params['root'],
                                             source=source,
                                             problem=problem,
                                             gpu_on=gpu_on,
                                             scene=scene,
                                             is_topK=is_topK)
            Y = Yt.T

            np.save('{2}/U_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), RQ)
            np.save('{2}/V_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), Y)
            if yBias is not None:
                np.save('{2}/B_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), yBias)
            if X is not None:
                np.save('{2}/tU_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), X)
            if xBiast is not None:
                xBias = xBiast.T
                np.save('{2}/tV_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), xBias)

    progress.subsection("Prediction")

    if params['model'] not in ['BUNV-MF', 'BUNV-MF_wo_e']:
        rating_prediction, topk_prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure,
                                                     bias=yBias, topK=params['topK'][-1], matrix_Train=train,
                                                     matrix_Test=test, gpu=gpu_on, is_topK=True)
    else:
        rating_prediction, topk_prediction = sub_predict(matrix_U=RQ, matrix_V=Y, tmatrix_U=X, tmatrix_V=xBias,
                                                         measure=measure, bias=yBias, topK=params['topK'][-1],
                                                         matrix_Train=train, matrix_Test=test, gpu=gpu_on, is_topK=True)

    progress.subsection("Evaluation")

    result = evaluate(rating_prediction, topk_prediction, test, params['metric'], params['topK'], analytical=analytical,
                      is_topK=True)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df
