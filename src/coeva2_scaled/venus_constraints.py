import numpy as np
import autograd.numpy as anp


def evaluate(x_ml, encoder):

    # ----- PARAMETERS

    tol = 1e-3

    # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
    calculated_installment = (
        np.ceil(
            100
            * (x_ml[:, 0] * (x_ml[:, 2] / 1200) * (1 + x_ml[:, 2] / 1200) ** x_ml[:, 1])
            / ((1 + x_ml[:, 2] / 1200) ** x_ml[:, 1] - 1)
        )
        / 100
    )
    g41 = np.absolute(x_ml[:, 3] - calculated_installment)

    # open_acc <= total_acc
    g42 = x_ml[:, 10] - x_ml[:, 14]

    # pub_rec_bankruptcies <= pub_rec
    g43 = x_ml[:, 16] - x_ml[:, 11]

    # term = 36 or term = 60
    g44 = np.absolute((36 - x_ml[:, 1]) * (60 - x_ml[:, 1]))

    # ratio_loan_amnt_annual_inc
    g45 = np.absolute(x_ml[:, 20] - x_ml[:, 0] / x_ml[:, 6])

    # ratio_open_acc_total_acc
    g46 = np.absolute(x_ml[:, 21] - x_ml[:, 10] / x_ml[:, 14])

    # diff_issue_d_earliest_cr_line
    g47 = np.absolute(
        x_ml[:, 22]
        - (_date_feature_to_month(x_ml[:, 7]) - _date_feature_to_month(x_ml[:, 9]))
    )

    # ratio_pub_rec_diff_issue_d_earliest_cr_line
    g48 = np.absolute(x_ml[:, 23] - x_ml[:, 11] / x_ml[:, 22])

    # ratio_pub_rec_bankruptcies_pub_rec
    g49 = np.absolute(x_ml[:, 24] - x_ml[:, 16] / x_ml[:, 22])

    # ratio_pub_rec_bankruptcies_pub_rec
    ratio_mask = x_ml[:, 11] == 0
    ratio = np.empty(x_ml.shape[0])
    ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
    ratio[~ratio_mask] = x_ml[~ratio_mask, 16] / x_ml[~ratio_mask, 11]
    ratio[ratio == np.inf] = -1
    ratio[np.isnan(ratio)] = -1
    g410 = np.absolute(x_ml[:, 25] - ratio)

    constraints = anp.column_stack([g41, g42, g43, g44, g45, g46, g47, g48, g49, g410])
    constraints[constraints <= tol] = 0.0
    scaled_constraints = encoder.constraint_scaler.transform(constraints)

    return scaled_constraints


def _date_feature_to_month(feature):
    return np.floor(feature / 100) * 12 + (feature % 100)
