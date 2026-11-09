import numpy as np
import itertools
import math


def _fit_polynomial_surface(x1, x2, y):
    design = np.column_stack([np.ones(len(x1)), x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _predict_polynomial_surface(coefficients, x1, x2):
    c0, c1, c2, c3, c4, c5 = coefficients
    return c0 + c1 * x1 + c2 * x2 + c3 * x1 * x2 + c4 * (x1 ** 2) + c5 * (x2 ** 2)


def _update_top_candidates(candidates, losses, values, loss, limit):
    if len(candidates) < limit:
        candidates.append(values)
        losses.append(loss)
        return
    worst_index = int(np.argmax(losses))
    if loss < losses[worst_index]:
        candidates[worst_index] = values
        losses[worst_index] = loss


def RIBG_impute(j, imputed):
    queue_x = []
    para1 = 0.5
    split = int(imputed.shape[0] * para1)
    for index in range(imputed.shape[1]):
        if index == j:
            continue
        queue_x.append(np.asarray(imputed[:, index], dtype=float))
    queue_aim = np.asarray(imputed[:, j], dtype=float)
    flag = True
    mini = math.inf
    while flag:
        F = 50
        next_queue = []
        loss_queue = []
        for xt1, xt2 in itertools.combinations(queue_x, 2):
            x1_head = xt1[:split]
            x2_head = xt2[:split]
            y_head = queue_aim[:split]
            coef_head = _fit_polynomial_surface(x1_head, x2_head, y_head)
            yC = _predict_polynomial_surface(coef_head, xt1, xt2)
            loss = float(np.sum((y_head - yC[:split]) ** 2))

            x1_tail = xt1[split:]
            x2_tail = xt2[split:]
            y_tail = queue_aim[split:]
            coef_tail = _fit_polynomial_surface(x1_tail, x2_tail, y_tail)
            yB = _predict_polynomial_surface(coef_tail, xt1, xt2)
            loss += float(np.sum((y_tail - yB[split:]) ** 2))
            loss += float(np.sum((yC - yB) ** 2))

            coef_full = _fit_polynomial_surface(xt1, xt2, queue_aim)
            Z = _predict_polynomial_surface(coef_full, xt1, xt2)
            _update_top_candidates(next_queue, loss_queue, np.asarray(Z, dtype=float), loss, F)

        if not next_queue:
            break
        ordered = sorted(zip(next_queue, loss_queue), key=lambda item: item[1])
        queue_x = [values for values, _ in ordered]
        loss_queue = [loss for _, loss in ordered]
        if len(queue_x) == 1 or loss_queue[0] >= mini:
            flag = False
        else:
            mini = loss_queue[0]
    return queue_x[-1]
def RIBG(df):
    data = df.to_numpy(dtype=float, copy=True)
    mean = np.nanmean(data, axis=0)
    imp = np.where(np.isnan(data), mean, data)
    imputed = np.asarray(imp, dtype=float)
    for j in range(df.shape[1]):
        if np.isnan(data[:, j]).any():
            queue = RIBG_impute(j, imputed)
            for i in range(df.shape[0]):
                if np.isnan(data[i, j]):
                    data[i, j] = queue[i]
    return data
