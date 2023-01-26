# imports
from enum import Enum

import pandas

from input import *
from reference import *
from matching import *
import pandas as pd


class Metric(Enum):
    JARO = 1,
    JARO_WINKLER = 2,
    LEVENSHTEIN = 3,
    JACCARD = 4,
    MONGE_ELKAN = 5,
    TFIDF = 6,
    SOFT_TFIDF = 7,
    LSTM = 8


def __print_matches(matches, type: Metric):
    for match in matches:
        print("{} match for {}:".format(type.name, match[0].name))
        print("---------------------------------------------")
        for product in match[1]:
            print("product: {}; similarity: {}; should be: {}; {}".format(
                product[0].name,
                product[1],
                match[0].retailerProductName,
                product[0].name == match[0].retailerProductName
                )
            )
        print()
    print()


def __create_vocabulary(ref: list[Product], inv):
    vocabulary = []
    [vocabulary.append(rp.name) for rp in ref]
    [vocabulary.append(i.name) for i in inv]

    with open("vocabulary.json", "w", encoding='utf8') as file:
        json.dump(vocabulary, file, ensure_ascii=False)


if __name__ == '__main__':
    reference_products = prepare_reference_data()

    invoices = read_invoices(open_invoice_file())

    # __create_vocabulary(reference_products, invoices)

    load_model()

    # jaro_winkler_match = jaro_winkler_bulk_matcher(reference_products, invoices, 0.7, True)
    # __print_matches(jaro_winkler_match, Metric.JARO_WINKLER)
    # print()

    # levenshtein_match = levenshtein_bulk_matcher(reference_products, invoices, 10, True)
    # __print_matches(levenshtein_match, Metric.LEVENSHTEIN)
    # print()

    # jaro_match = jaro_bulk_matcher(reference_products, invoices, 0.7, True)
    # __print_matches(jaro_match, Metric.JARO)
    # print()

    # tfidf_result = tfidf_bulk_matcher(reference_products, invoices, 0.4, True)
    # __print_matches(tfidf_result, Metric.TFIDF)
    # print()

    # jaccard_match = jaccard_bulk_matcher(reference_products, invoices, 0.4, True)
    # __print_matches(jaccard_match, Metric.JACCARD)
    # print()

    # monge_elkan_match = monge_elkan_bulk_matcher(reference_products, invoices, 0.7, True)
    # __print_matches(monge_elkan_match, Metric.MONGE_ELKAN)<
    # print()

    # soft_tfidf_match = soft_tfidf_bulk_matcher(reference_products, invoices, 0.4, True)
    # __print_matches(soft_tfidf_match, Metric.SOFT_TFIDF)
    # print()

    lstm_match = lstm_bulk_matcher(reference_products, invoices, 0.7, False)
    __print_matches(lstm_match, Metric.LSTM)
    print()

    # fileData = open("stringTestData.json", "r", encoding="utf-8")
    # test_data = json.load(fileData, object_hook=lambda d: SimpleNamespace(**d))
    # matching_results = test_matchers(test_data)
    #
    # for result in matching_results:
    #     print(result)

    # model_results = test_model(test_data)
    # for index, result in enumerate(model_results):
    #     # print("prediction: {} ; should be {}".format(result, test_data[index].similar, ))
    #     print(result)
