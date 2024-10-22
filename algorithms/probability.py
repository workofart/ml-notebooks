import numpy as np
# Bayes Rule
# P(A|B) = P(B|A) * P(A) / P(B)
# Where
# P(A|B) = Posterior Probability
# P(B|A) = Likelihood
# P(A) = Prior Probability
# P(B) = Evidence


def compute_bayes_rule(
    p_cancer, p_test_positive_given_no_cancer, p_test_positive_given_cancer
):
    # We need to add up the two cases that make up the positive test
    # p(test positive) = p(test positive | cancer) * p_cancer + p(test positive | no cancer) * p_no_cancer
    p_test_positive = (p_test_positive_given_cancer * p_cancer) + (
        p_test_positive_given_no_cancer * (1 - p_cancer)
    )

    # Simulating negative tests as well
    # p(test negative) = p(test negative | cancer) + p(test negative | no cancer)
    # p(test negative | cancer) = 1 - p(test positive | cancer)
    # p(test negative | no cancer) = 1 - p(test positive | no cancer)
    p_test_negative = ((1 - p_test_positive_given_cancer) * p_cancer) + (
        (1 - p_test_positive_given_no_cancer) * (1 - p_cancer)
    )

    # Ultimately, as we know more information from the diagnoses, we want to know how our p(cancer) belief changes
    # This p(cancer), based on the Bayes rule can either come from p(cancer | test positive) or p(cancer | test negative)
    # Posterior Updates
    # p(cancer | test positive) = p(test positive | cancer) * p(cancer) / p(test positive)
    p_cancer_given_test_positive = (
        p_test_positive_given_cancer * p_cancer / p_test_positive
    )
    # p(cancer | test negative) = p(test negative | cancer) * p(cancer) / p(test negative)
    #                           = (1 - p(test positive | cancer)) * p(cancer) / p(test negative)
    p_cancer_given_test_negative = (
        (1 - p_test_positive_given_cancer) * p_cancer / p_test_negative
    )
    return {
        "p_cancer_given_test_positive": round(p_cancer_given_test_positive, 2),
        "p_cancer_given_test_negative": round(p_cancer_given_test_negative, 2),
        "p_test_negative": round(p_test_negative, 2),
        "p_test_positive": round(p_test_positive, 2),
    }


if __name__ == "__main__":
    probs = compute_bayes_rule(
        # We have 1 person with p(cancer) of having cancer, this could be prior knowledge or population statistics
        # I will just randomly guess 0.5%
        p_cancer=0.005,  # initial belief 0.5%
        # accuracy = p(test positive | cancer)
        p_test_positive_given_cancer=round(np.random.uniform(0.6, 0.9), 2),
        # p(test positive | no cancer)
        p_test_positive_given_no_cancer=round(np.random.uniform(0.01, 0.05), 2),
    )
    print(probs)
