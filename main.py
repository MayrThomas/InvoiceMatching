# imports
from input import *
from reference import *
from matching import *

x = '{"name": "Avocados 500g", "quantity": null, "unitPrice": null, "totalPrice": {"value": 1.46, "currency": "EUR", ' \
    '"convertedValue": null, "convertedCurrency": null, "convertTime": null}} '

if __name__ == '__main__':
    reference_products = prepare_reference_data()

    result = read_invoice(x)
    print("extracted product name: -- ", result)

    print()
    print("Jaro-Winkler match for --", result.name)
    print("---------------------------------------------")
    jaro_winkler_match = jaro_matcher(reference_products, result, True)

    print("product: --", jaro_winkler_match[0].name)
    print("similarity: --", jaro_winkler_match[1])
    print()
    print("Levenshtein match for --", result.name)
    print("---------------------------------------------")

    levenshtein_match = levenshtein_matcher(reference_products, result)

    print("product: --", levenshtein_match[0].name)
    print("distance: --", levenshtein_match[1])
    print()
    print("Jaro match for --", result.name)
    print("---------------------------------------------")

    jaro_match = jaro_matcher(reference_products, result)

    print("product: --", jaro_match[0].name)
    print("similarity: --", jaro_match[1])

    print()
    print("TF-IDF cosine match for --", result.name)
    tfidf_result = tfidf_matcher(reference_products, result)
    print("product(s): --")
    [print(x.name) for x in tfidf_result[0]]
    print("similarity: --", tfidf_result[1])

    print()
    print("Jaccard match for --", result.name)
    jaccard_match = jaccard_matcher(reference_products, result)
    print("product: --", jaccard_match[0].name)
    print("similarity: --", jaccard_match[1])

    print()
    print("Monge-Elkan match for --", result.name)
    monge_elkan_match = monge_elkan_matcher(reference_products, result)
    print("product: --", monge_elkan_match[0].name)
    print("similarity: --", monge_elkan_match[1])

    print()
    print("Soft TF-IDF match for --", result.name)
    soft_tfidf_match = soft_tfidf_matcher(reference_products, result)
    print("product: --", soft_tfidf_match[0].name)
    print("similarity: --", soft_tfidf_match[1])

