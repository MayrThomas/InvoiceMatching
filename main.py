# imports
from enum import Enum
from input import *
from reference import *
from matching import *


class Metric(Enum):
    JARO = 1,
    JARO_WINKLER = 2,
    LEVENSHTEIN = 3,
    JACCARD = 4,
    MONGE_ELKAN = 5,
    TFIDF = 6,
    SOFT_TFIDF = 7,


def __print_matches(matches, type: Metric):
    for match in matches:
        print("{} match for {}:".format(type.name, match[0].name))
        print("---------------------------------------------")
        for product in match[1]:
            print("product: {}; similarity: {}".format(product[0].name, product[1]))
        print()
    print()


x = '{"name": "Avocados 500g", "quantity": null, "unitPrice": null, "totalPrice": {"value": 1.46, "currency": "EUR", ' \
    '"convertedValue": null, "convertedCurrency": null, "convertTime": null}} '

if __name__ == '__main__':
    reference_products = prepare_reference_data()

    result = read_invoice(x)
    invoices = read_invoices(open_invoice_file())
    test_invoices = invoices[1:10]

    jaro_winkler_match = jaro_winkler_bulk_matcher(reference_products, test_invoices, 0.7, True)
    __print_matches(jaro_winkler_match, Metric.JARO_WINKLER)
    print()

    levenshtein_match = levenshtein_bulk_matcher(reference_products, test_invoices, 10, True)
    __print_matches(levenshtein_match, Metric.LEVENSHTEIN)
    print()

    jaro_match = jaro_bulk_matcher(reference_products, test_invoices, 0.7, True)
    __print_matches(jaro_match, Metric.JARO)
    print()

    tfidf_result = tfidf_bulk_matcher(reference_products, test_invoices, 0.4, True)
    __print_matches(tfidf_result, Metric.TFIDF)
    print()

    jaccard_match = jaccard_bulk_matcher(reference_products, test_invoices, 0.7, True)
    __print_matches(jaccard_match, Metric.JACCARD)
    print()

    monge_elkan_match = monge_elkan_bulk_matcher(reference_products, test_invoices, 0.7, True)
    __print_matches(monge_elkan_match, Metric.MONGE_ELKAN)
    print()

    soft_tfidf_match = soft_tfidf_bulk_matcher(reference_products, test_invoices, 0.4, True)
    __print_matches(soft_tfidf_match, Metric.SOFT_TFIDF)
